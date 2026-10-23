#import "../template/template.typ": *

#show: docstyle

= Qwen/Qwen3-Next-80B-A3B-Instruct

#link("https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct")[huggingface: Qwen/Qwen3-Next-80B-A3B-Instruct]


这个模型没有发报告，发了一个blog：#link("https://qwen.ai/blog?from=research.latest-advancements-list&id=4074cca80393150c248e508aa62983f9cb7d27cd&")[blog: Qwen3-Next-80B-A3B-Instruct]

比较大的改动是把3/4的attention换成了更省空间和计算的delta net， 类似mamba那种SSM 

#image("/assets/qwen3-next.png")

= 参数


== "attention_dropout": 0.0,

推理的时候attention不会dropout，训练的时候不知道

== token_id 

词表相关
"bos_token_id": 151643,
"eos_token_id": 151645
"vocab_size": 151936

== "hidden_size": 2048

token 维度


== "decoder_sparse_step": 1

每一层都是`Qwen3NextSparseMoeBlock`



== "full_attention_interval": 4

每四层用一个full_attention
  
== "hidden_act": "silu",

经典silu
  
== "initializer_range": 0.02

权重初始化标准差范围，也就是每一个参数在训练时初始化是从$N(0, 0.02)$里采样得到的
== "intermediate_size": 5120

普通情况下MLP的中间层维度，其实没用上，每一层都是`Qwen3NextSparseMoeBlock`

== GatedDeltaNet相关参数

"linear_conv_kernel_dim": 4,

这个表示每个token在一个短卷积里混合多少个自己之前token（包括自己）的信息（每个维度平均池化）

"linear_key_head_dim": 128,
（key的head dim）

"linear_num_key_heads": 16,
key的num heads和query的num heads一致

"linear_num_value_heads": 32,
GQA是query的头多，这里的value的头比较多，是key头的整数倍

"linear_value_head_dim": 128,
value的head dim，可以和k的不同，但是这里相同

== attention相关参数


"head_dim": 256,
"num_attention_heads": 16,
"num_key_value_heads": 2

"attention_bias": false
config里甚至没显式写这个键，qkvo投影全都不带bias，整个checkpoint里也确实一个`.bias`张量都没有（235B那篇专门强调过的坑）

== "max_position_embeddings": 262144,

训练时的最长上下文

== moe相关参数


"moe_intermediate_size": 512
每个专家的维度


"num_experts": 512,
"num_experts_per_tok": 10,
512个专家选10个

"norm_topk_prob": true
10个专家的softmax概率重新归一化一下，mixtral是默认操作，这里加了个开关，这次是明着写了true

"output_router_logits": false
训练的时候应该会开

"router_aux_loss_coef": 0.001
防止专家塌缩计算的loss，这里是乘在上面的系数

"shared_expert_intermediate_size": 512
Qwen2.5模仿deepseek引入，Qwen 3删了，Qwen3Next又回来了
== "num_hidden_layers": 48,

层数

== rope 

"partial_rotary_factor": 0.25,
这个表示只对部分dim加上RoPE，剩下的维度保持不变。这里full attention的head_dim = 256，0.25表示只有64维要加RoPE，然后又因为RoPE两两配对，只有32个inv_freqs，每个inv_freqs $theta_i = 10000000^(-(2i)/64)$（RoPE只有full attention层用，GDN那边压根没有位置编码这回事）

"rope_scaling": null,
"rope_theta": 10000000 （7个0）


  
== "rms_norm_eps": 1e-06

防止rms除0加的最小值

  
== "tie_word_embeddings": false,

lm_head和token_embedding不用同一块权重

== "use_cache": true

推理开kv cache
== "use_sliding_window": false,

他们终于发现max_window_layers是个死参数了

但是这个作为看上去用了但实际上全关了的SWA遗址还留着

= 组件

在说新的组件之前，先说大概讲一下如何拼接的，每一层的接口还是`Qwen3NextDecoderLayer`，属于Attention的部分会根据`self.block_type`是`full_attention`，就用`Qwen3NextAttention`，如果是`linear_attention`就用 `Qwen3NextGatedDeltaNet`

输入输出的形状经过两个block都不改变，都是`[batch, seq_len, hidden_size]`这一套，所以这两个块可以替代彼此的职责

== RMSNorm 

Qwen3Next的Norm换成了Gemma版，有两个小差别

```py
class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
```

第一个是weight不再直接乘在输出上，乘的其实是`(1 + weight)`，系数从1开始学

因为 Llama：weight 初始化为全 1，前向是 `weight * x_norm`；
Gemma：weight 初始化为全 0，前向是 `(1.0 + weight) * x_norm`。

这样可以避免一开始1就会收到参数上的正则化系数的惩罚而被迫向0移动（weight_decay)

二是精度路径

Llama 是 `x.to(fp16) * w`（归一化后先降回低精度，再做乘法，两次舍入）；Gemma 是 `(x * w).to(fp16)` 乘法整个留在 fp32 里，最后才降精度一次。也就是说llama乘法在float 16中进行，但是gemma在float 32中进行，最后才降精度，精度更高一些

== Qwen3NextSparseMoeBlock
Qwen 3-Next用的是Qwen2MoE而不是Qwen3MoE，所以还有shared_expert

```py
class Qwen3NextSparseMoeBlock(Qwen2MoeSparseMoeBlock):
    pass
```

放一下代码

这是Qwen2的

```py
class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Qwen2MoeTopKRouter(config)
        self.experts = Qwen2MoeExperts(config)
        self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output
```

这是Qwen3的

```py
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = Qwen3MoeExperts(config)
        self.gate = Qwen3MoeTopKRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
```


== Qwen3NextAttention

Qwen3-Next Attention的组件也有所修改

除了从Qwen3MoE那里继承来的qk norm，还多加了一个门控

门控加的方式有点耦合，我觉得这个代码写的不好

就是把`q_proj`拓宽了两倍

```py
@no_inherit_decorator
class Qwen3NextAttention(Qwen3MoeAttention):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        del self.sliding_window

```

然后输出的会得到一个gate向量

```py
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
)
gate = gate.reshape(*input_shape, -1)
```

然后在output之前，经过一个sigmoid函数，乘在attention的输出上


复习一下sigmoid函数长什么样子

#image("/assets/sigmoid.png")

这个函数关于(0,0.5)中心对称，值域被压在(0,1)之间
```py
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = attn_output * torch.sigmoid(gate)

attn_output = self.o_proj(attn_output)
```

为什么要加这个门控

主要来自论文#link("https://arxiv.org/pdf/2505.06708")[Gated Attention for Large Language Models: Non-linearity, Sparsity,
and Attention-Sink-Free]

原论文加了5个位置，在 Q、K、V 投影之后（G2/G3/G4）、SDPA 输出之后（G1）、最终 dense 输出之后（G5）。

SDPA 输出位置的 gating（G1）效果最好，能带来约 0.2 PPL 的下降和 MMLU 上约 2 分的提升；value 输出位置的 gating（G2）也有明显收益，但整体不如 G1。

然后head-specific gating很重要

作者把 gating 有效的原因归结为两点：一是它在 $W_V$ 和 $W_O$ 这两个连续线性映射之间引入了非线性，缓解了低秩映射的表达能力限制；二是它引入了输入依赖的稀疏性，能够过滤与当前 query 无关的上下文信息。这种稀疏 gating 还能消除 attention sink：baseline 中平均 46.7% 的注意力集中在第一个 token，而 SDPA gating 后降到 4.8%。

*attention sink* 是指：在 softmax attention 中，某些特定 token（尤其是序列开头的第一个 token）会不成比例地获得大量注意力分数。论文给出的具体例子是，baseline 模型中平均有 46.7% 的注意力分数集中在第一个 token 上，某些层甚至达到 83%。

论文对 attention sink 的影响主要讨论了两个方面：

1. 对长上下文扩展的影响论文发现，在训练长度（32k）以内，attention sink 并不会明显损害模型的长上下文性能；gated 模型和 baseline 在 32k 设置下表现接近。但当用 YaRN 把上下文扩展到 128k 时，baseline 的性能下降比 gated 模型严重得多。作者的假设是：baseline 模型依赖 attention sink 来调节注意力分布，而 YaRN 修改 RoPE base 后(应该是没有微调的对比？），这种 attention sink 模式难以在无训练的情况下自适应，导致性能明显下降。相比之下，gated 模型主要依靠输入依赖的 gating score 来控制信息流，因此对 RoPE 修改更鲁棒。

2. 与 massive activations 的关系论文还分析了 attention sink 和 massive activations（隐藏状态中的极大激活值）之间的关系。一个关键发现是：massive activations 并不是 attention sink 的必要条件。当只在 value 层输出位置（G2）加 gating 时，模型虽然消除了 massive activations，但 attention sink 仍然存在。这说明两者并非简单的因果关系。

== GDN Gated Delta net

这个思路来源于RNN，朴素写法是对所有历史累加：$S_t = S_(t - 1) + v_t dot k_t^T$

DeltaNet = 对记忆做一步在线梯度下降。 把 S 看成 fast weight（快权重矩阵），每个 token 定义回归目标，走一步 SGD


先把维度钉死（单个 v 头的视角）：$k_t, v_t$ 都是 $RR^128$ 的向量，$S in RR^(128 times 128)$（数学取向是 $d_v times d_k$，代码里存成 $d_k times d_v$，每个 v 头一份，一共 32 份），$alpha_t$和$beta_t$是每个 v 头一个的标量；k 头只有 16 个，会`repeat_interleave`成 32 份来对齐 v 头

$

  L(S) &= 1/2 || S k_t - v_t ||^2 \
    \
  S_t &= S_(t - 1) - beta_t nabla L(S_(t - 1)) \
  &= S_(t - 1) - beta_t (S_(t - 1) k_t - v_t) k_t^T \
  &= S_(t - 1)(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
$

这里$S_t$ 被叫做快权重，是因为它并不是静态的在`nn.Module`的时候被注册的权重，而是在计算的时候生成的

而且它还是一个在线梯度下降，因为它有一个损失函数 $ L(S) &= 1/2 || S k_t - v_t ||^2$ (就是$S k_t$需要能查出来$v_t$)，然后还有一个显式学习率$beta_t$


然后到了Gated DeltaNet，就是再上面又多加了一个参数

加 adaptive weight decay $alpha_t$

$

  S_t = S_(t - 1)(alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T
$

这个在在线梯度下降里有一个视角是

先把整个式子展开：

$

S_t = alpha_t S_(t - 1)  + beta_t (v_t - alpha_t S_(t - 1) k_t)k_t^T   
  
$

然后构造一个Loss函数（长得有点奇怪，不过论文作者肯定是怎么自圆其说怎么构造的啦）

$
  || S_t - alpha_t S_(t-1)||^2_F - 2 chevron.l S_t k_t , beta_t (v_t - alpha_t S_(t - 1) k_t) chevron.r
$


然后记$S = S_t$(这个就是我们主要要优化的快权重)，$hat(S) = S_(t-1)$ (这个是类似于梯度上的系数)

$
  L(S) = ||S||^2_F - 2 alpha chevron.l S, hat(S) chevron.r_F - 2 beta chevron.l S k_t, v_t chevron.r + 2 alpha beta chevron.l S k_t, hat(S) k_t chevron.r + "const"
$

然后对$S$求导

$
  nabla L(S) &= 2 S - 2 alpha hat(S) - 2 beta v_t k_t^T + 2 alpha beta (hat(S) k_t) k_t^T \
  &= 2[S - alpha hat(S) - beta(v_t - alpha hat(S) k_t) k_t^T]
$

因为 Frobenius 项 $||S − α hat(S)||^2$ 使 L 的曲率（Hessian）恰好是 2I，最小值点可以一步解出

$
  2[S - alpha hat(S) - beta(v_t - alpha hat(S) k_t) k_t^T] &= 0 \
  alpha hat(S) + beta(v_t - alpha hat(S) k_t)k_t^T &= S
$

这个我感觉怪怪的，主要是因为它本身是在一个做过一步随机梯度下降的式子里加一个gate，然后又在这个做过一步随机梯度下降梯度的式子做了一个求最优化的操作。这就是做了两次梯度下降。

以后做别的类似的线性注意力的时候会来再和GDN对比一下的

然后来重新看一下参数

- "linear_conv_kernel_dim": 4,
- "linear_key_head_dim": 128,
- "linear_num_key_heads": 16,
- "linear_num_value_heads": 32,
- "linear_value_head_dim": 128,


看一下代码

*第一步* 还是从hidden states投影出这些参数 （话说，hidden states糅合那么多知识不会很累吗）

```py
# forward 主干代码
projected_states_qkvz = self.in_proj_qkvz(hidden_states) 

projected_states_ba = self.in_proj_ba(hidden_states)


query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

```

其中`in_proj_qkvz`的权重形状是`[12288, 2048]`，这里面包括了`q(2048)+k(2048) + v(4096) + z(4096)`

然后`in_proj_ba`权重形状是`[64, 2048]`，这里面是`a(32) + b(32)`

这里面拆开每个tensor的函数长这样

```py
# 分支函数
def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
    """
    Derives `query`, `key` and `value` tensors from `mixed_qkvz` and `mixed_ba`.
    """

    new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
        self.num_k_heads,
        2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
    )
    new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)

    mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
    mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
    split_arg_list_qkvz = [
        self.head_k_dim,
        self.head_k_dim,
        (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        (self.num_v_heads // self.num_k_heads * self.head_v_dim),
    ]
    split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
    query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
    b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
    # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
    value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
    z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
    b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
    a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
    return query, key, value, z, b, a
```

其中第一个形状是

```py
new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
        self.num_k_heads,
        2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
    )
```

意思是丢掉最后一维12288，然后把12288拆开变成`[16, 128 + 128 + 32 //16 * 128 + 32 // 16 * 128]`

```py
new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)
```

这一步是计算ba，形状是`[16, 32 // 16 + 32 // 16]`

也就是分组的时候每个v的头对应k的头(一组中两个)，乘上的$alpha$和$beta$也是不同的

后面的就是纯切分了

```py
# forward主干代码
query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))
```

切分出来又把他们都第三维和第四维展平了， 为了之后的conv做准备，此时的形状各是
```
query: [B, seq_len, 16, 128] → [B, seq_len, 2048]     # 16×128 摊平
key:   [B, seq_len, 16, 128] → [B, seq_len, 2048]
value: [B, seq_len, 32, 128] → [B, seq_len, 4096]     # 32×128 摊平
```

然后会把它们都拼起来
```py
mixed_qkv = torch.cat((query, key, value), dim=-1)
mixed_qkv = mixed_qkv.transpose(1, 2)
```

组成一个`[B, seq_len, 8192]`的大向量，更改顺序换成`[B, 8192, seq_len]`

*第二步* 是会通过一个shortconv，这个shortconv目的是把某个token的信息和之前的3个token混合一下
```py
# forward主干代码
if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
    conv_state = cache_params.layers[self.layer_idx].conv_states[0]
    # Single-token cached decode: the fused per-step kernel updates the conv state in-place.
    mixed_qkv = causal_conv1d_update(
        mixed_qkv,
        conv_state,
        self.conv1d.weight.squeeze(1),
        self.conv1d.bias,
        self.activation,
    )
else:
    if cache_params is not None:
        mixed_qkv = cache_params.update_conv_state(
            mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
        )

    mixed_qkv = causal_conv1d_fn(
        mixed_qkv,
        self.conv1d.weight.squeeze(1),
        self.conv1d.bias,
        activation=self.activation,
        **kwargs,
    )

    # Drop the additional previous states
    if cache_params is not None:
        mixed_qkv = mixed_qkv[:, :, -seq_len:]
```

这段代码比较confusing，主要就是if分支是99%的decode会走的路径，每次只存前3个token的信息，然后混合进当前的token

else是训练和其他情况会走的分支，主要看训练

```py 
# 在init注册的weight向量
self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        ) # Cin = 8192, Cout=8192
# 分支函数
@use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")
def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
    **kwargs,
):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1

    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)

```

复习一下conv1d是对一个大小为`[B, C_in, L_in]` 到`[B, C_out, L_out]`的变换

这里就是`[B, 8192, seq_len]`到`[B, 8192, seq_len]`的变换，这里没设置bias，设置kernel_size=4是在最后一维上按照4的窗口大小滑动，groups和in_out_channels都相等，就是说输出的每个通道只关注对应的1个通道。 weight的形状是`[8192, 1, 4]`

padding实际上是左右都补3个，但是看最后out后面跟着一个`[:, :, : seq_len]`就是为了截断，等效于只有左边补（那么为什么你不写`padding=(3,0)`捏）

做的操作是把每个token映射到8192维的信息，根据这个kernel（大小为4）的系数，和前3个token还有它自己加权平均一下，这样每个token里就混入自己前面的token的信息了

注意这不是串行的，所以并不是前面混合完了我继续混入，是一下子把干净的token投影 全部转换为 混合之前信息的token投影

转换之后，又会把整个tensor转换回原来的多头形式

```py
# forward主干
mixed_qkv = mixed_qkv.transpose(1, 2)
query, key, value = torch.split(
    mixed_qkv,
    [
        self.key_dim,
        self.key_dim,
        self.value_dim,
    ],
    dim=-1,
)
query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim) # [B, L, 16, 128]
key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim) # [B, L, 16, 128]
value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim) # [B, L, 32, 128]
```

之后会计算一下这个公式里面的系数$alpha$和$beta$

$S_(t - 1)(alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T$

这里`beta = b.sigmoid()` 是一个很自然的把$RR$上的数规范到`(0,1)`的做法

然后`g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)`是Mamba2里对$alpha$的计算的一部分

```
A_log:  [32]     每 V 头一个可学习基准衰减，存 log 保证 -exp(A_log) < 0 恒成立
a:      [B,L,32] 输入依赖的"时间步"（in_proj_ba 投影出来）
dt_bias:[32]     可学习偏置，控制初始时间步
softplus(·)      保证 Δ > 0
g = A·Δ < 0  →  α = exp(g) ∈ (0,1)   ← 遗忘因子，构造上就合法
```

```py
# forward主干

beta = b.sigmoid()
# If the model is loaded in fp16, without the .float() here, A might be -inf
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```

然后会把k q头和v对齐一下（MQA是q和v要去对k，GDN这里是v头更多一些）

这里`repeat_interleave`表示把同一个值重复几遍（名字有点容易歧义），所以操作之后序列是：

```py
原来:  [k0, k1, k2, k3, ...]        # num_k_heads = 16
之后:  [k0, k0, k1, k1, k2, k2, ...] # num_v_heads = 32
        └v0┘    └v1┘ ...
```
```py
# forward主干
if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
```

之后会开始最主要的部分

```py
recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
if use_precomputed_states and seq_len == 1:
    core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
        **kwargs,
    )
else:
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
        **kwargs,
    )

# Update cache
if cache_params is not None:
    cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
```


其中`recurrent_state`：如果在增量推理（`use_precomputed_states` = 有 cache 且该层已有历史状态），就从 cache 里读上一步的递归状态作为初值；否则（prefill / 训练）为 None，从零状态开始。

分支 1：torch_recurrent_gated_delta_rule — 单 token 逐步递推
条件：use_precomputed_states and seq_len == 1，即带缓存的常规自回归 decode，每次只喂 1 个新 token。走纯递推形式：用旧状态 + 当前 token 的 q/k/v/beta/g 直接更新一步，O(1) 地算出新状态和输出。decode阶段会进行的操作

分支 2：torch_chunk_gated_delta_rule — 分块并行
负责其余所有情况：

- prefill：处理整段 prompt（无初始状态）；
- 一次进多个 token 的增量调用（如投机解码的验证步、batched generate 中某些序列同时前进）——虽然是增量，但 seq_len > 1，逐 token 递推太慢；
- 无 cache 的前向（训练、plain forward）。


这个函数把序列切成 chunk 并行计算，数学上和逐步递推等价，但能利用矩阵乘的并行度，适合长序列。

收尾会写回cache

先看一眼*推理分支*

```py
# 推理分支函数： torch_recurrent_gated_delta_rule
@use_kernel_func_from_hub_with_fallback("fused_recurrent_gated_delta_rule", "fla")
def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes linear attention using the gated delta rule, by iterating over each token in the sequence dimension.
    Same args and return value as torch_chunk_gated_delta_rule, except for `chunk_size` because the sequence dim is not
    chunked."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # If enabled, normalize query and key vectors (done once in fp32 for better accuracy)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # And always normalize queries by the head dimension
    query = query / (query.shape[-1] ** 0.5)

    # Create the storage for the last recurrent state, which will be updated in place. If a previous state is provided,
    # it is the starting point, otherwise start with a zeroed buffer.
    if initial_state is None:
        recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)

    # Loop over each token and update the recurrent state
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        # Decay the recurrent state
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t
        # Update the recurrent state with the current token
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # And use it to compute the attention output for the current token
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
```

这里面的向量大小：

`query`, `key`, `value`是`[B, seq_len, 32, 128]`

然后`beta`, `g`是`[B, seq_len, 32]`

不过这里是`seq_len = 1`的分支，所以可以认为`seq_len`都是1

然后会把他们的1，2维转置，也就是

`query`, `key`, `value`是`[B, 32, seq_len, 128]`

`beta`, `g`是`[B, 32, seq_len]`

```py
decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

# Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]

query, key, value, beta, decay = [
    x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
    for x in (query, key, value, beta, decay)
]
```

然后会对query和key进行一个l2norm(没有缩放系数，单纯除一下rsqrt)

再把query的方差缩小到1，和attention一样，会乘上一个$1/sqrt("head_dim")$

```py

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

# 推理分支函数： torch_recurrent_gated_delta_rule
# If enabled, normalize query and key vectors (in fp32 to match the FLA library)
if use_qk_l2norm_in_kernel:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
# And always normalize queries by the head dimension
scaling = query.shape[-1] ** -0.5
query = query * scaling
```

在开始之前会分配一块空间给要存进cache的`last_recurrent_state`

大小是：`[B, 32, 128, 128]`

没传`initial_state`就新建全零的，传了就用它，这个`.to(value)`主要是为了对齐dtype和device
```py
# 推理分支函数： torch_recurrent_gated_delta_rule
# Create the storage for the last recurrent state, which will be updated in place. If a previous state is provided,
# it is the starting point, otherwise start with a zeroed buffer.
if initial_state is None:
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
else:
    last_recurrent_state = initial_state.to(value)
core_attn_out = torch.zeros_like(value)
```

主体更新部分：

$S_t = S_(t - 1)(alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T$

这里用的是另一个形式

$S_t = alpha_t S_(t - 1) + beta_t (v_t - alpha_t S_(t - 1) k_t) k_t^T$




先计算$alpha_t S_(t - 1)$ `last_recurrent_state = last_recurrent_state * decay_t`

然后计算 `kv_mem =  (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)` 含义是用k去查记忆里的内容，维度变化是，公式是$sum_k S[k, v] k_t [k]$ 得到的结果和value相同形状，是`[batch_size, 32, 128]` (seq_len维度被循环消掉了) （这个等价于`S^T @ k_t`）

然后delta就是目标查出来的`v_t`和`kv_mem`的差值，然后再乘上学习率，`delta = (v_t - kv_mem) * beta_t`

最后按照上面的形式合并起来：`last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)`

这里` k_t.unsqueeze(-1)` 是`[B, 32, 128, 1]`，`delta.unsqueeze(-2)`是`[B, 32, 1, 128]` 是一个外积


输出是线性注意力

```sh
S_t = Σ_{s≤t} k_s ⊗ v_s        # [k_dim, v_dim]
o_t = S_tᵀ q_t = Σ_s (q_t·k_s) · v_s

```

当前 q 和每个历史 k 算相似度 q_t·k_s，以它为权重对所有历史 v 加权求和


最后的输出也是先广播再求和，也等价于`S_t^T q_t`
`core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)`


输出会转换为`[B, 32, seq_len, 128]` -> `[B, seq_len, 32, 128]`
```py
# 推理分支函数： torch_recurrent_gated_delta_rule
# Loop over each token and update the recurrent state
for i in range(sequence_length):
    q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
    # Decay the recurrent state
    decay_t = decay[:, :, i].exp()[..., None, None]
    last_recurrent_state = last_recurrent_state * decay_t
    # Update the recurrent state with the current token
    beta_t = beta[:, :, i].unsqueeze(-1)
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    # And use it to compute the attention output for the current token
    core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

# Discard the final state if not requested
last_recurrent_state = None if not output_final_state else last_recurrent_state
# Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
return core_attn_out, last_recurrent_state
```

然后再看一下*训练分支*

```py
# 训练用的函数，把一段很长的prefill序列按照chunk_size进行切分和并行
@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes the gated delta rule, by chunking along the sequence dimension.
    Args:
        query: Query tensor of shape [batch_size, sequence_length, num_k_heads, k_head_dim]
        key: Key tensor of shape [batch_size, sequence_length, num_k_heads, k_head_dim]
        value: Value tensor of shape [batch_size, sequence_length, num_v_heads, v_head_dim]. num_v_heads can be equal
            to num_k_heads, same for v_head_dim and k_head_dim.
        g: Decay (in log space) tensor of shape [batch_size, sequence_length, num_v_heads]: the recurrent state is
            multiplied by exp(g) at each step, so entries must be <= 0.
        beta: Beta tensor of shape [batch_size, sequence_length, num_v_heads]
        chunk_size: Size of the chunks along the sequence dimension.
        initial_state: The recurrent state, an optional tensor of shape [batch_size, num_v_heads, k_head_dim, v_head_dim]
        output_final_state: Whether to output the new recurrent state along with the output.
        use_qk_l2norm_in_kernel: If this flag is set to True, query and key vectors are L2-normalized.
    Returns:
        - The output tensor of shape [batch_size, sequence_length, num_v_heads, v_head_dim]
        - Either None or the new recurrent state tensor of shape [batch_size, num_v_heads, k_head_dim, v_head_dim]
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)  # -1 is the padded sequence length
    decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # If enabled, normalize query and key vectors (in fp32 to match the FLA library)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # And always normalize queries by the head dimension
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    # Pad sequence length to be a multiple of chunk_size. Padding is described as (left_pad, right_pad) for each dim.
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    # Apply beta to K and V, which is the "learning rate" of the recurrent state for a given token, ie. how much the new
    # state influence the old state. Beta is often normalized to (0, 1) where 0 = no update; 1 = overwrite old state.
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape all tensors to chunk the sequence dimension (adds a new dimension of size chunk_size)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    # Create a chunk-sized strictly upper triangular mask, ie. the mask of what a causal chunk may not attend to
    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)

    # Cumulative decay within each chunk (dim 3 is the position inside the chunk). Since decay is in log space,
    # cum_decay[..., t] is the log of a product of decays between the start of the chunk and position t
    cum_decay = decay.cumsum(dim=3)

    # First phase: compute intra-chunk quantities.
    # The pairwise decays: pairwise_decay[..., i, j] = exp(cum_decay_i - cum_decay_j) is the decay accumulated between
    # positions j and i of a chunk. Positive values are masked to -inf before exp to avoid overflow
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    pairwise_decay = pairwise_decay.exp()  # with the exp, we exit log space, so we can apply this decay to the states

    # Compute auxiliary tensors: the Upper Triangular (ut) transform system and the intra-chunk attn (QK dot product)
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # Gated delta attention uses a UT transform to condense several delta rule updates into a few matmuls. After the UT
    # system is solved, we can then compute the new_values (called "u" in the DeltaNet paper) and the decayed keys
    # reading the old state (k_cumdecay). In the update, the part of new_values that the old state already predicts is
    # subtracted out, so that only the correction is written to the recurrent state: this is the delta rule.

    # Not all export targets support the fast triangular solver, so we build the inverse by forward substitution then
    if not is_torchdynamo_exporting():
        new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
        k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
    else:
        ut_system = -ut_system.tril(-1)  # ut_system is masked to only keep the strictly lower triangle
        for i in range(1, chunk_size):
            row = ut_system[..., i, :i].clone()
            sub = ut_system[..., :i, :i].clone()
            ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        ut_system = ut_system + torch.eye(chunk_size, dtype=ut_system.dtype, device=ut_system.device)
        new_values, k_cumdecay = ut_system @ v_beta, ut_system @ decayed_k_beta

    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)

    # Apply decay once rather than in each chunk
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # Second phase: the sequential scan over chunks
    for i in range(num_chunks):
        # Compute attention output for the current chunk: add the read of the previous recurrent state
        # (inter_chunk_attn) with the within-chunk attention (intra_chunk_attn)
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        # Update the recurrent state: new recurrent state (S_t+1) = decayed old state (S_t * (I-βkk^T)) + update (βvk^T)
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Reshape the output to the orignal shape: flatten the chunk dimension, then drop padding
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    # Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state
```

*第一步*是和推理类似的预处理


第一步和推理完全一样， 转换一下query, key, value, beta, decay的形状，然后进行一下qk_l2norm

`query`, `key`, `value`是`[B, seq_len, 32, 128]`

然后`beta`, `decay`是`[B, seq_len, 32]`

经过transpose是把这些形状变成了

`query`, `key`, `value`是`[B, 32, seq_len, 128]`

然后`beta`, `decay`是`[B, 32, seq_len]`

之后q和k的头会进行l2norm，q还会乘上一个$1/sqrt("head_dim")$

```py
# 训练函数分支：torch_chunk_gated_delta_rule
initial_dtype = query.dtype
batch_size, sequence_length, _, k_head_dim = key.shape
num_v_heads, v_head_dim = value.shape[-2:]
recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)  # -1 is the padded sequence length
decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

# Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
query, key, value, beta, decay = [
    x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
    for x in (query, key, value, beta, decay)
]
# If enabled, normalize query and key vectors (in fp32 to match the FLA library)
if use_qk_l2norm_in_kernel:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
# And always normalize queries by the head dimension
scaling = query.shape[-1] ** -0.5
query = query * scaling

```

之后会把整个序列pad到chunk_size的倍数，填充的数字是0

考虑到`query`, `key`, `value`是`[B, 32, seq_len, 128]`，然后`beta`, `decay`是`[B, 32, seq_len]`

然后`F.pad`的语义是从最后一维开始，每两个数表示这一维的`(left_pad, right_pad)`

所以`query`, `key`, `value`是`F.pad(x, (0, 0, 0, pad_size))` 表示从倒数第二维开始右pad

然后`beta`，`decay`是`F.pad(x, (0, pad_size))`，表示从最后一维开始右pad

```py
# 训练函数分支：torch_chunk_gated_delta_rule
# Pad sequence length to be a multiple of chunk_size. Padding is described as (left_pad, right_pad) for each dim.
pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

total_sequence_length = sequence_length + pad_size
num_chunks = total_sequence_length // chunk_size
```

之后会把beta乘进key和value，做成一个副本，这样后续需要算$beta_t k_t$和$beta_t v_t$的时候不需要重复计算

然后会把 chunk_size作为一个维度加进向量里，这里`query`,`key`, `k_beta`，`v_beta`的形状变成了`[B, 32, num_chunks, 64, 128]`，这里64是chunk_size，然后128是head_dim

`decay` 是 `[B, 32, num_chunks, 64]`

```py
# 训练函数分支：torch_chunk_gated_delta_rule
# Apply beta to K and V, which is the "learning rate" of the recurrent state for a given token, ie. how much the new
# state influence the old state. Beta is often normalized to (0, 1) where 0 = no update; 1 = overwrite old state.
v_beta = value * beta.unsqueeze(-1)
k_beta = key * beta.unsqueeze(-1)

# Reshape all tensors to chunk the sequence dimension (adds a new dimension of size chunk_size)
query, key, k_beta, v_beta = [
    x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
]
decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
```


预处理的部分到这里就结束了， *第二步* 是 *块内并行* 部分

回过头继续看递推公式

$S_t = alpha_t S_(t - 1)(I - beta_t k_t k_t^T) + beta_t v_t k_t^T$

把这个公式写成

$S_t = alpha_t S_(t - 1) + beta_t (v_t - alpha_t S_(t - 1) k_t) k_t^T$

考虑到$k_t$ 是一个只有$[128, 1]$的列向量，那么每次对$S_t$的更改就相当于对$S_(t - 1)$的每一行 + 一个系数 $times k_t$，也就是会有某个列向量和$k_t$做外积，然后累加到衰减后的state矩阵上（证明这里的矩阵是$d_v times d_k$，代码里的矩阵是$d_k times d_v$，看代码的时候会提一下）

有了这个直观， 考虑证一个引理，大致意思是所有的$S_t$可以写成初始的$S_0$乘上一个系数，再变成所有序列在它之前的$k_t^T$ 和一个只对每个位置定义一次的列向量$xi_t$

这么做更直观的意义是，本来是一个需要递推的公式，现在变成了每一步多了一个和上一个状态无关的增量，也就不用强行递推了，但是由于这个形式这个系数需要解方程，所以又不能搞的很大，就变成chunk形式了

这个引理如下：

对每个$t$，存在向量$xi_1, xi_2, dots,xi_t, dots in RR^(d_v)$ (只在时刻$s$定义一次，之后不再改变)
$
  S_t = d_t S_0 + sum_(s lt.eq t ) (d_t / d_s) dot xi_(s) k_s^T
$ 

其中$d_t = alpha_1 alpha_2 dots alpha_t$ (也就是$alpha$的连乘)

证明是归纳法

对$t = 0$, $S_0 = S_0, d_0 = 1$ 成立

对别的$t$，如果已经有了$S_(t - 1) = d_(t - 1) S_0 + sum_(s lt.eq t - 1) (d_(t - 1) / d_s) dot xi_(s) k_s^T$

那么考虑原始的递推公式里的两项：

第一个是衰减项$alpha_t S_(t - 1)$

$alpha_t S_(t - 1) = d_t S_0 + sum_(s lt.eq t - 1) (d_(t) / d_s) dot xi_(s) k_s^T$ 形状和要证明的已经很类似的，只是连加符号里少了下标为$t$的系数

第二个是擦除算子要读的 $alpha_t S_(t - 1)k_t$

$alpha_t S_(t - 1) k_t =d_t (S_0 k_t) + sum_(s lt.eq t - 1) (d_(t) / d_s) dot xi_(s) k_s^T k_t $

第一部分$d_t (S_0 k_t)$ 可以直观理解为老状态在$k_t$维度上的残留，第二部分是前面的$k_s$($s lt t$) 的重叠部分$k_s^T k_t$理解为重叠度

定义$xi_t = beta_t (v_t - alpha_t S_(t - 1) k_t)$，代入第二部分得到

$xi_t = beta_t v_t - beta_t d_t (S_0 k_t) - sum_(s < t) beta_t (d_t / d_s) (k_s^T k_t) xi_s$

于是这个不依赖前一个状态的新变量就吸收了老状态上$k_t$的擦除，满足了我们引理要证明的形式

$S_t = alpha_t S_(t - 1) + beta_t (v_t - alpha_t S_(t - 1) k_t) k_t^T = d_t S_0 + sum_(s lt.eq t - 1) (d_(t) / d_s) dot xi_(s) k_s^T + xi_t k_t^T$


而代码里的ut_system方程组还要对$xi_t$再进行一步拆分，把$xi_t$ 按照$S_0$线性拆开，多了一个未知数，要求$xi_t = u_t - S_0 w_t$ (有点像多了一个维度)

设$N[t, s] = beta_t (d_t / d_s) (k_s^T k_t)$

对于这两个新未知数有

$u_t = beta_t v_t - sum_(s < t) N[t, s] u_s$ (理解为对本块写入的修正) 最后会得到代码里的`new_values`部分

$w_t = beta_t d_t k_t - sum_(s < t) N[t, s] w_s$ （对之前老状态的修正，因为也是$S_0$的系数，和旧状态有点关系） 最后会得到代码里的`k_cumdecay` 部分

我们要解的方程的形式就是（以$u_t$为例）

把含$u_t$的部分移到左边就可以得到$u_t + sum_(s < t) N[t, s] u_s = beta_t v_t$

然后变换成矩阵的形式，其中$N$是`[64, 64]`的矩阵， 就有

$

(I + N) mat(delim: "[", u_s; u_(s + 1); dots.v;u_t) = mat(delim:"[", beta_s v_s; beta_(s + 1) v_(s + 1); dots.v; beta_t v_t) 

$

我们要解的就是这样一个方程，对$w_t$同理

接下来看代码


首先是两个准备工作

其中`strictly_upper_mask` 就是一个*严格*(就是对角线上不是1，但是列号j大于行号i的部分都是1)的上三角矩阵，大小为`[chunk_size, chunk_size]` 也就是`[64, 64]` （用在盖在`pairwise_decay`上，之后会说）

然后是对`decay`进行累加，`decay`目前已经变成了一个`[B, 32, num_chunks, chunk_size]`的东西，累加其实就是前缀和，形状并不改变，所以`cum_decay`的大小还是`[B, 32, num_chunks, chunk_size]`，也就是对decay的分段求和。因为真正的衰减系数是还要做一次幂运算，$e^a times e^b = e^(a+b)$，直接指数再乘精度会下溢，所以对指数部分求和再做幂运算精度会比较好


```py
# Create a chunk-sized strictly upper triangular mask, ie. the mask of what a causal chunk may not attend to
strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)

# Cumulative decay within each chunk (dim 3 is the position inside the chunk). Since decay is in log space,
# cum_decay[..., t] is the log of a product of decays between the start of the chunk and position t
cum_decay = decay.cumsum(dim=3)
```

然后我们计算一个`pairwise_decay`

其中`cum_decay.unsqueeze(4)`的形状是`[B, 32, num_chunks, chunk_size, 1]`

`cum_decay.unsqueeze(3)`的形状是`[B, 32, num_chunks, 1, chunk_size]`

广播后会得到一个一个`[B, 32, num_chunks, chunk_size, chunk_size]`的矩阵

任意一个下标是`[b, h, c, i, j]`，值是$log(d_i / d_j)$。之后的`pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))` 又把所有$j > i$的部分填成-inf，只有$i >= j$的部分才是有意义的

之后会把`pairwise_decay`进行幂运算，也就是`pairwise_decay = pairwise_decay.exp()`来退出log space，而且上一步对于不合法的位置填的是-inf，这样$e^("-inf") = 0$
```py
# First Phase 块内可并行部分，在代码里被称作first phase，之后的补充注释会沿用这个叫法
# First phase: compute intra-chunk quantities.
# The pairwise decays: pairwise_decay[..., i, j] = exp(cum_decay_i - cum_decay_j) is the decay accumulated between
# positions j and i of a chunk. Positive values are masked to -inf before exp to avoid overflow
pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
pairwise_decay = pairwise_decay.exp()  # with the exp, we exit log space, so we can apply this decay to the states
```


接下来开始准备方程组系数

这个`ut_system`就是我们说的N，形状是`[B, num_head, num_chunks, chunk_size, chunk_size]` （两个乘数，`k_beta`的形状是`[B, num_head, num_chunks, chunk_size, head_dim] `，`key.transpose(-1, -2)`的形状是`[B, num_head, num_chunks, head_dim, chunk_size] `)

对于一段区间`[s, s + chunk_size)`，每个`ut_system[t, s]`位置的值就是$beta_t (d_t / d_s) k_t^T k_s$ ，这个对所有$t > s$的位置都有意义

对于$t < s$的部分， `pairwise_decay`负责掩盖掉，对于$t = s$的部分，下面解方程的部分也会给改造成1

对于`intra_chunk_attn`，是对`key`的查询，最后负责输出，形状同样也是`[B, num_head, num_chunks, chunk_size, chunk_size]`

条目`intra_chunk_attn[t, s] = (q_t^T k_s) (d_t / d_s)`，其中`q`已经乘过$1/sqrt("head_dim")$了

和`ut_system`对比一下就是一个负责擦除方程组，一个负责读取权重

然后`decayed_k_beta` 是负责解出来$w_t$的右边系数，先进行了exp再unsqueeze，形状是`[B, num_heads, num_chunks, chunk_size, 1]`，同时`k_beta`的形状是`[B, num_heads, num_chunks,chunk_size, head_dim]`，这里的乘法是广播乘法，不是矩阵乘法，所以`decayed_k_beta`的形状还是`[B, num_heads, num_chunks,chunk_size, head_dim]`

要解$u_t$的右段部分就是`v_beta`，代码之前算过了

```py
# First Phase 块内可并行部分，准备方程系数
# Compute auxiliary tensors: the Upper Triangular (ut) transform system and the intra-chunk attn (QK dot product)
ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)
```


接下来是解方程部分

if分支里的部分就是直接调库去解，看一下手写的部分

因为我们要解的是$I + N$，`ut_system.tril(-1)`相当于让我们解一个记$M = -N$，求$I - M$的逆矩阵，这里`tril(-1)`保留严格下三角(-1是相对于对角线的偏移量，0的话会保留对角线），会把对角线清除掉

这个原地求解相当于计算逆矩阵$X (I - M) = I$，也就是$X = I + M X$，但是这里相当于原地把`ut_system`改造成$(X - I) = M + M (X - I)$，最后再把这个$I$加回来

初始的时候第0行都是0，所以枚举是从1开始的，一直到`chunk_size - 1`

对于每一行`row`，还有此刻$(X - I)$的左上角`[i, i]`小矩阵，都会clone出来，clone 让读和写解耦（否则 view 别名 + 原地写会在 autograd/导出追踪里出问题）。

因为M和X都是严格下三角，所以对第i行的计算只依赖前`[0, i - 1]`行，所以可以递推

来看`ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)`里涉及到的tensor

第一个`row`就是`[B, num_heads, num_chunks, i]`

然后`row.unsqueeze(-1)`就是`[B, num_heads, num_chunks, i, 1]`

`sub`的形状是`[B, num_heads, num_chunks, i, i]`

它们之间的乘法是广播乘法，会得到`[B, num_heads, num_chunks, i, i]`的tensor，其中条目`[k, j]` 就是$M[i, k] times X[k, j]$

`.sum(-2)` 对k这个维度求和，形状变为`[B, num_heads, num_chunks, i]`，条目$j$的位置就是$M(X - I)[i, j]$

最后加上原来的$M$的第$i$行，也就是前面的`row + `

最后`ut_system`被原样改造为了逆矩阵，直接乘在右侧向量（对$u_t$来说是`v_beta`，对$w_t$来说是`decayed_k_beta`)就解出了他们的值，分别命名为`new_values`和`k_cumdecay`

```py
# First Phase 块内可并行部分，解方程
# Gated delta attention uses a UT transform to condense several delta rule updates into a few matmuls. After the UT
# system is solved, we can then compute the new_values (called "u" in the DeltaNet paper) and the decayed keys
# reading the old state (k_cumdecay). In the update, the part of new_values that the old state already predicts is
# subtracted out, so that only the correction is written to the recurrent state: this is the delta rule.

# Not all export targets support the fast triangular solver, so we build the inverse by forward substitution then
if not is_torchdynamo_exporting():
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
else:
    ut_system = -ut_system.tril(-1)  # ut_system is masked to only keep the strictly lower triangle
    for i in range(1, chunk_size):
        row = ut_system[..., i, :i].clone()
        sub = ut_system[..., :i, :i].clone()
        ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    ut_system = ut_system + torch.eye(chunk_size, dtype=ut_system.dtype, device=ut_system.device)
    new_values, k_cumdecay = ut_system @ v_beta, ut_system @ decayed_k_beta
```

接下来是第三步，块间串行的部分（代码里叫second phase，也就是说，梯度链条会有`num_chunks`那么长，但是也比原始长度要稍微好些了）

下面的代码是一些准备工作

如果没有`initial_state`就新建一个全是0的，形状和`recurrent_state_shape`一样都是`[B, num_heads, head_dim(of k), head_dim(of v)]`

`core_attn_out`的形状也是一样的

回到论文里的公式$o_t = S_t^T q_t$，考虑引理里$S_t$的形式

$
  S_t = d_t S_0 + sum_(s lt.eq t ) (d_t / d_s) dot xi_(s) k_s^T 
$ 

主要是$S_0$的前面有个$d_t$ (对于分块来说，是每个段首前一个state)，直接就乘进`q`里了，所以会算一个`query = query * cum_decay.exp().unsqueeze(-1)`

而这个`key`，也是一样的，为了一次性算到段尾，就把$d_C/d_t$一次性也乘进`key`里了，所以会算一个`key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)`

这个`chunk_decay`是为了更新状态的（q是为了更新output的），也是考虑$d_t S_0$，这个状态跨了整个段，也就是`cum_decay`每一块的最后一个值，最后补成了形状`[B, num_heads, num_chunks, 1, 1]`（因为要广播）

```py
if initial_state is None:
    last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
else:
    last_recurrent_state = initial_state.to(new_values)
core_attn_out = torch.zeros_like(new_values)

# Apply decay once rather than in each chunk
query = query * cum_decay.exp().unsqueeze(-1)
key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
chunk_decay = cum_decay[..., -1].exp()[..., None, None]
```

然后是第三步块间串行的循环部分（代码里叫second phase），比较直观

其中

`new_values`的维度是`[B, num_heads, num_chunks, chunk_size, head_dim(of v)]`

`k_cumdecay`的维度是`[B, num_heads, num_chunks, chunk_size, head_dim(of k)]` （虽然v的head_dim和k的head_dim都是128，但是这里还是标注一下）

这里计算的`v_new`就是我们之前引理里的$xi$， 计算方法我们推的是$u_t - w_t S_0$，现在这里也一样，是`new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state`（recall一下，`last_recurrent_state`的维度是`[B, num_heads, head_dim(of k), head_dim(of v)]`）

最后得到的`v_new`就是`[B, num_heads, chunk_size, head_dim(of v)]`

输出注意力的块间部分，也就是$d_t S_0^T q_t$，这里是用`query[:, :, i] @ last_recurrent_state`算的

而$sum_(s lt.eq t ) (d_t / d_s) dot xi_(s) k_s^T$和`query`的相乘，之前是用之前计算的`intra_chunk_attn`完成的

代码callback一下(这个时候的key还没有乘上cum_decay那一串）：
```py
intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
```

就这样我们得到了这个块里的`core_attn_out`的部分（论文里是$S_t^T q_t$，代码里维度转置了，就是$q_t^T S_t$，`core_attn_out`维度是`[B, num_heads, num_chunks, chunk_size, head_dim(of v)]`，每次只更新其中一个`chunk`，也就是第三个下标）

最后再一次性把中间变量`last_recurrent_state`从块前一个元素，推到这个块的末尾（也就是下一个块前的元素）`last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new`

```py
# Second phase: the sequential scan over chunks
for i in range(num_chunks):
    # Compute attention output for the current chunk: add the read of the previous recurrent state
    # (inter_chunk_attn) with the within-chunk attention (intra_chunk_attn)
    v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
    inter_chunk_attn = query[:, :, i] @ last_recurrent_state
    core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
    # Update the recurrent state: new recurrent state (S_t+1) = decayed old state (S_t * (I-βkk^T)) + update (βvk^T)
    last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
```

*第四步*就是输出上的一些处理

`core_attn_out`是`[B, num_heads, num_chunks, chunk_size, head_dim(of v)]`

其中`padded_output_shape`是`[B, num_heads(of v), -1, head_dim(of v)]` 

转换后`core_attn_out`也变成了`[B, num_heads(of v), num_chunks * chunk_size, head_dim(of v)]`

然后再把pad的部分裁掉`core_attn_out = core_attn_out[:, :, :sequence_length]`

transpose一下后，恢复到`[B, seq_len, num_heads(of v), head_dim(of v)]`的

`last_recurrent_state`形状是一样的，还是`[B, num_heads, head_dim(of k), head_dim(of v)]`

```py
# 训练分支函数torch_recurrent_gated_delta_rule第四步，形状上的一些处理
# Discard the final state if not requested
last_recurrent_state = None if not output_final_state else last_recurrent_state
# Reshape the output to the orignal shape: flatten the chunk dimension, then drop padding
core_attn_out = core_attn_out.reshape(padded_output_shape)
core_attn_out = core_attn_out[:, :, :sequence_length]
# Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
return core_attn_out, last_recurrent_state
```

回到*最初的`Qwen3NextGatedDeltaNet` forward函数*，输出还要过一个门控和一个`o_proj`，但是都比较直观


其中

```py
core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
z = z.reshape(-1, z.shape[-1])
```

先把两个向量展平成`[B * num_heads(of v) * seq_len, head_dim(of v)]`

然后过一个norm，其实也是一种门控

```py
core_attn_out = self.norm(core_attn_out, z)
```

这个norm的定义类似RMSNorm，只不过把GemmaRMSNorm里面的`hidden_state * (1 + weight)`变成了先是`hidden_state * weight`(这里weight从1开始初始化的，不知道为什么不继续用Gemma那版)

然后再额外多一个门控`hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))`，激活函数默认用`silu`

```py
# 这个RMSNormGated只在GatedDeltaNet里用了，当然Gated不是说Norm被Gated了哈，出去了之后还有个Norm要过呢
@use_kernel_forward_from_hub("RMSNormGated")
class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "silu"

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

        return hidden_states.to(input_dtype)
```


最后形状变换为`[B, seq_len, value_dim]`会经过`out_proj`的映射，回到主干里的`[B, seq_len, hidden_size]`

```py
# 主干的forward函数
# Update cache
if cache_params is not None:
    cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

z_shape_og = z.shape
# reshape input data into 2D tensor
core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
z = z.reshape(-1, z.shape[-1])
core_attn_out = self.norm(core_attn_out, z)
core_attn_out = core_attn_out.reshape(z_shape_og)
core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

output = self.out_proj(core_attn_out)
return output
```

好了，这个组件终于结束了

= 拼接


考虑到`decoder_sparse_step=1`，所以根本没有`Qwen3NextMLP`这个组件参与构建

拼接是经典的pre-norm范式

这里是`Qwen3NextRMSNorm`(就是上面那个Gemma版) -> `Qwen3NextAttention` 或者 `Qwen3NextGatedDeltaNet`(每4层先3层GDN再1层Full Attention) -> `Qwen3NextRMSNorm` -> `Qwen3NextSparseMoeBlock`

最后还会经过一个RMSNorm


```py
class Qwen3NextDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        # token mixer
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3NextAttention(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3NextMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif self.block_type == "full_attention":
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states

        return hidden_states
```


余下的`Qwen3NextForCausalLM`其实没有继承`MixtralForCausalLM`，是照着Mixtral的样板自己写的一份，和MoE有关的loss也在这里计算

顺带一提，checkpoint里还塞了一个`mtp.*`的多token预测模块：一层完整的"full attention + 512个expert的MoE"，再加`fc`和两个`pre_fc_norm`，大约1.65B参数，是照抄DeepSeek V3那套给投机解码当draft用的。transformers的modeling里没有这个组件，源码里明写了`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`，加载时直接扔掉，要用得靠vLLM专门的`qwen3_next_mtp.py`这类实现。


= 计算题

设 d = hidden_size = 2048

== 总参数量

每一层的组成：token mixer（"full_attention_interval": 4，48 层里 12 层是 full attention、36 层是 GDN，就是拼接那节说的"先 3 层 GDN 再 1 层 full"）+ router + 512 个 expert + shared expert + 两个 Norm。"decoder_sparse_step": 1，所以 48 层全是 MoE 层，没有 dense 层

*Full attention（12 层）*：q_proj 为了塞 gate 被拓宽了两倍，出来 $2 times 256 times 16 = 8192 = 4d$ 宽，q 一份满的 $4 d^2$；k、v 因为 GQA（2 个 kv head）只有 $256 times 2 = 512 = d/4$ 列，各 $d times d/4 = d^2/4$；o_proj 接的是 $256 times 16 = 4096 = 2d$ 宽的输入，$2 d^2$

$4 d^2 + 2 times d^2/4 + 2 d^2 = 6.5 d^2 = 27,262,976$

再补上从 Qwen3 继承的 q_norm + k_norm，各管一个 head_dim = 256，$2 times 256 = 512$，attention 一共 $27,263,488$。所有投影都是 `bias=False`（config 里连 "attention_bias" 这个键都没显式写，走的是 transformers 的默认值），整个 checkpoint 里一个 `.bias` 张量都没有

*GDN（36 层）*：进项是 in_proj_qkvz 一块 $[12288, d]$（$q 2048 + k 2048 + v 4096 + z 4096 = 12288 = 6d$ 行），$6 d^2$，加上 in_proj_ba 的 $64d$；出项是 out_proj $[2048, 4096]$，$2 d^2$。剩下的都是零碎：depthwise 的 conv1d 权重 $[8192, 1, 4]$ 共 $32,768$，A_log 和 dt_bias 各 32，RMSNormGated 128（按 head_v_dim 算）

$6 d^2 + 2 d^2 + 64d + 32,768 + 64 + 128 = 8 d^2 + 64d + 32,960 = 33,718,464$

GDN 反而比 full attention 厚：$33,718,464$ 对 $27,263,488$——linear attention 省的是 KV cache 和注意力的计算量，参数差不多

*MoE（每层都有）*：router 一块 $[512, d]$，$512d = 1,048,576$；512 个 expert，每个是标准 SwiGLU，intermediate 是 $512 = d/4$，$3 times d/4 times d = 3/4 d^2 = 3,145,728$，并不大，复制 512 遍就是

$384 d^2 = 1,610,612,736$

shared expert 从Qwen2.5回来了（"shared_expert_intermediate_size": 512），同样 $3/4 d^2 = 3,145,728$；再加 shared_expert_gate 一条 $d$。MoE 合计

$384.75 d^2 + 513d = 1,614,809,088$

expert 一项占了单层参数的约 $98%$（比 235B 的 97% 还要极端）：专家从"128 个 3/8 d 宽"变成"512 个 1/4 d 宽"，总宽 $512 times 512 = 262,144 = 128d$，是 hidden_size 的 128 倍（235B 是 48 倍）

*两个 Norm*：$2d = 4096$

每层合计：full attention 层 $391.25 d^2 + 515d + 512 = 1,642,076,672$，GDN 层 $392.75 d^2 + 579d + 32,960 = 1,648,531,648$

48 层一共

$12 times 1,642,076,672 + 36 times 1,648,531,648 = 79,052,059,392$

零碎部分：

最后一层 Norm $2048$

一个 token_embedding，一个 lm_head（"tie_word_embeddings": false 所以是两份）：$2048 times 151,936 times 2 = 622,329,856$

总参数 $79,052,059,392 + 622,329,856 + 2048 = 79,674,391,296$，大约 79.7B，名字里的 80B 就是它四舍五入来的。对着官方权重验证：safetensors 全部分片的 total_size 是 162,649,725,440 字节（bf16），$div 2$ 得 $81,324,862,720$，比主干多出来的 $81,324,862,720 - 79,674,391,296 = 1,650,471,424$ 恰好就是 checkpoint 里附带的那个 MTP 模块（拼接那节提过），扣掉它一分不差

== 激活参数量

每个 token 在每个 MoE 层只走 10 个 expert，专家稀疏度 $10/512 approx 2%$（235B 是 $8/128 = 1/16$，Mixtral 是 $2/8 = 25%$），稀疏得没边了；但 attention / GDN、shared expert、router、Norm 谁都要过

*每层激活*：full attention 层 = attention $6.5 d^2 + 512$ + router $512d$ + 10 个 expert $10 times 3/4 d^2 = 7.5 d^2$ + shared expert $3/4 d^2$ + gate $d$ + Norm $2d$ $= 14.75 d^2 + 515d + 512 = 62,921,216$，只占该层参数的 $approx 3.8%$

GDN 层 $= 16.25 d^2 + 579d + 32,960 = 69,376,192$，占 $approx 4.2%$

48 层一共

$12 times 62,921,216 + 36 times 69,376,192 = 3,252,597,504$

加上每个 token 必走的零碎（embedding + lm_head + 最后一层 Norm）

$3,252,597,504 + 622,331,904 = 3,874,929,408$

大约 3.9B。把 embedding 和 lm_head 摘掉再算：$3,252,597,504 approx 3.3B$，官方口径的"激活 3B"应该是砍掉 embedding 的算法。激活只有总量的 $3.9/79.7 approx 4.9%$。和 Mixtral、235B 类似，虽然每个 token 只算 3.9B 的账，512 个 expert 谁都可能被选中，全得常驻显存，实际要装的是 80B

== KV cache

这才是这个模型真正的卖点，bf16（2 字节一个数）算账：

*full attention 层（12 层）*：K、V 各 $256 times 2 = 512$ 列，一层一个 token $2 times 512 times 2 = 2048$ 字节，12 层一个 token $24 "KB"$，随上下文线性涨

*GDN 层（36 层）*：不存 KV，存固定大小的 recurrent_state，一层 $[32, 128, 128] = 2^19$ 个数，恰好整 $1 "MB" = 2^20$ 字节，36 层 $36 "MB"$，和上下文多长完全无关（另有 conv state 每层 $8192 times 3$ 个数即 $49,152$ 字节，36 层不到 2 "MB"，先忽略）

对比：如果 48 层全是 full attention，一个 token $96 "KB"$，262,144 的上下文要 $approx 25.8 "GB"$；现在是 $36 "MB" + 24 "KB" times 262,144 approx 6.5 "GB"$，正好是四分之一——intro 说的"把 3/4 的 attention 换成 delta net"落到实处就是这个 1/4

回本点：GDN 的固定开销 $36 "MB"$，每 token 省下 $96 - 24 = 72 "KB"$，$36 times 2^20 div (72 times 2^10) = 512$，恰好 512 个 token 回本，之后都是节省空间