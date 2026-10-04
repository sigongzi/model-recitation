#import "../template/template.typ": *

#show: docstyle 
= llama-2-7b

#link("https://modelscope.cn/models/LLM-Research/llama-2-7b")[model_scope: llama-2-7b]

== 关键参数

=== attention_bias: false

Q/K/V/O 不带投影

=== attention_dropout: 0.0

注意力dropout概率，推理时一般不启用

=== token id 

序列开始

bos_token_id: 1 

序列结束

eos_token_id: 2

填充token

pad_token_id: 0

=== hidden_act: "silu"

激活函数是silu

=== "hidden_size": 4096,
  
每个token的向量维度
  
=== "initializer_range": 0.02,

权重初始化标准差范围，也就是每一个参数在训练时初始化是从$N(0, 0.02)$里采样得到的

=== "intermediate_size": 11008,

FFN 中间层维度，Llama-2 的 SwiGLU FFN 常用这个值 (大约是 8/3 hidden_size)
=== "max_position_embeddings": 4096,

最大上下文长度，即模型训练时支持的最长序列

=== "mlp_bias": false,

FFN 线性层是否带 bias，这里不带


=== "num_hidden_layers": 32

Transformer 层数，决定模型深度
=== 多头注意力


*"num_attention_heads": 32*

注意力头数，每个头维度为 4096 / 32 = 128
  
*"num_key_value_heads": 32* 和 *num_attention_heads* 相同说明是标准MHA

=== "pretraining_tp": 1

未启用tp


=== "rms_norm_eps": 1e-05,

	RMSNorm 的 epsilon

=== rope参数

*"rope_scaling": null*
RoPE 缩放配置；为 null 表示没有启用长上下文外推缩放

*"rope_theta": 10000.0* RoPE 旋转位置编码的基频参数，Llama-2 默认是 10000

=== "tie_word_embeddings": false

输入嵌入和输出 lm_head 是否共享权重，这里不共享

=== "use_cache": true

推理时是否启用 KV cache，开启后可加速生成

=== "vocab_size": 32000

词表大小

= 组件

== RMSNorm

LLaMA版 RMSNorm

*注意这个函数的输入一定要强制转换成float32, 输出再转换为原类型*

只有一个参数 self.weight， 大小和hidden size一致

公式是

$
  "RMSNorm"(x_i) = x_i / sqrt(1/n sum_(k = 1)^n x_k^2 + epsilon) dot gamma_i 
$

```py
@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
```

== RoPE

LLaMA版 RoPE

计算 `inv_freqs`

```python
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
```

得到$"inv_freq"_i = theta^(-(2i)/dim)$

然后和`position_ids`做矩阵乘法

```python
inv_freq_expanded = (
    self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1).to(dtype=torch.float, device=x.device)
) # 把 inv_freq 从 [dim//2] 扩展成 [batch, dim//2, 1]，方便和位置索引做矩阵乘法。

position_ids_expanded = position_ids[:, None, :].float()  # 把位置索引从 [batch, seq_len] 变成 [batch, 1, seq_len]，并转为浮点数。
```

对于cos和sin的运算要关闭混合精度， 默认是float32 （因为inv_freqs生成的时候就是float32）
（在这里self.attention_scaling=1）
```python
with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2) # [batch, seq_len, dim // 2]
            emb = torch.cat((freqs, freqs), dim=-1) # 第m个位置是 [m theta_0, m theta_1, cdots, m theta_(dim / 2), m theta_0, m theta_1, cdots, m theta_(dim / 2)]
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
```

这里的配对相当于$x_i$和$x_(i + dim / 2)$配对

考虑旋转矩阵乘法 (相当于 $x_i + x_(i + dim / 2) upright(i)$和$cos theta + sin theta upright(i)$ 复数乘法的实部和虚部)

$
R(theta) mat(delim:"[", x_i; x_(i + dim / 2)) = mat(delim:"[", x_i cos theta - x_(i + dim / 2) sin theta; x_i sin theta + x_(i + dim / 2) cos theta)
$

== MLP

SwiGLU的标准写法

一般形式为

$
"SwiGLU"(x) = W_"down" "SiLU"(W_"gate" x dot.o W_"up" x)
$

gate_proj 负责判断每个维度有多重要，up_proj 负责提供候选特征。两者逐元素相乘后，模型就能根据当前输入动态调节信息流。

down_proj 就是 把门控后的中间表示再投影回原始 hidden_size。

intermediate_size = 11008, hidden_size = 4096

```py
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

== Attention

是最普通的MHA

head_dim 是通过 `hidden_size //  num_attention_heads` 算的， 代入一下是 128



```py
self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        ) # [4096, 32 * 128]
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        ) # [4096, 32 * 128]
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        ) # [4096, 32 * 128]
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        ) # [32 * 128, 4096]


# 其实都是[4096, 4096]
```


还有scaling 参数

```py
self.scaling = self.head_dim**-0.5
```

具体相乘部分：

这里 Q的维度是 `[batch, 32, seq_len, 128]`

K.transpose(1,2)是`[batch, 32, 128, seq_len]`

V是 `[batch, 32, seq_len, 128]`
公式：

$
  "softmax"((Q K^T) / sqrt("hidden_size")) V
$
```py
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```

== 拼接

=== DecoderLayer

每一层：

R1=hidden_state RMSNorm -> Casual Attention -> +R1 residual =  R2 -> RMSNorm -> FFN -> +R2 residual
```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    use_cache: bool | None = False,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states

```

=== 模型数据流

输入：input_ids

通过embed_tokens转换为 token，形状`[batch, seq_len, 4096]`

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx) #[32000, 4096]
```

通过32层Decoder Layer， 变换后形状不发生改变`[batch, seq_len, 4096]`

输出额外通过一个RMSNorm，形状不变 `[batch, seq_len, 4096]`

由于llama的embed_tokens（把离散的标号变成4096维向量）和lm_head（把4096维向量变成一个32000的概率分布）是分开的

最后会通过一个lm_heads

```python
logits = self.lm_head(hidden_states[:, slice_indices, :]) # [batch, seq_len, 32000]
```

然后计算loss

在这里pad只传两个数表示`(pad_left, pad_right)`，只填充最后一维，这里表示在最后填上一个`ignore_index`，然后把labels的第一个删掉

就像一个序列`A B C D`，现在变成`B C D <ignore_index>`

因为每个token自己经过lm_heads出的logits，会变成自己下一个token的概率分布


```python
def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss
```


== 计算题

=== 每一层的参数量

两个Norm：$4096 times 2 = 8192$ $2d$

MLP: $4096 times 11008 times 2 ("gate + up") + 11008 times 4096 = 135,266,304$ 大约是 $8d^2$

Attention $4096 times 4096 times 4 ("q_proj + k_proj + v_proj + o_proj") = 67,108,864$ 大约是$4d^2$

所以估算可以是$12d^2 + 2d = 201,334,784$ 大约201 M 

实际是 $8192 + 135,266,304 + 67,108,864 = 202,383,364$ 准确是202 M


32层的估算结果是 6.432B

实际计算结果 $202,383,364 times 32 = 6,476,267,648$


零碎权重

最后一层Norm $4096$

一个token_embedding，一个lm_heads：$4096 times 32000 times 2 = 262,144,000$ 大约是262 MB

估算结果：6.694 B 

实际计算 $6,476,267,648 + 4096 + 262,144,000 = 6,738,415,744$
