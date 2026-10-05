#import "../template/template.typ": *

#show: docstyle 

= Mistral-7B-v0.1

#link("https://huggingface.co/mistralai/Mistral-7B-v0.1")[huggingface: mistralai/Mistral-7B-v0.1]

= 关键参数 

==  token_id 

序列开始"bos_token_id": 1

序列结束"eos_token_id": 2

目测pad_token_id应该和llama一样是0，但是似乎推理的时候用不到，它就没在公开的模型config.json里填写


== "hidden_act": "silu",
  
激活函数是silu

$
  "SiLU"(x) = x dot sigma(x) = x / (1 + e^(-x))
$
#image("/assets/silu.png")
== "hidden_size": 4096

每个token的向量维度
  
== "initializer_range": 0.02,

权重初始化标准差范围，也就是每一个参数在训练时初始化是从$N(0, 0.02)$里采样得到的

== "intermediate_size": 14336,

没有解释为什么是 3.5 x "hidden_size"，比llama给出的2.67 倍(8/3) 要宽，他们声称更宽的 FFN 在他们的消融实验里表现更好

但没有说为什么是这个数而不是别的

== "max_position_embeddings": 32768

最长的上下文长度， 这里比llama的4096要长
  
== attention 头数
"num_attention_heads": 32
  
"num_key_value_heads": 8

这种叫  GQA（Grouped-Query Attention，分组查询注意力）

head_dim 是 4096 / 32 = 128

q_proj的形状是 `[4096, 128 x 32] = [4096, 4096]`
v_proj和 k_proj的形状都是 `[4096, 8 x 32] = [4096, 1024]`

对应方式

```
Q 头:   0  1  2  3 | 4  5  6  7 | 8 ...11 | ... | 28...31
KV 头:  0  0  0  0 | 1  1  1  1 | 2  2  2  2 | ... | 7  7  7  7
```
== "num_hidden_layers": 32

模型深度32层
  
== "rms_norm_eps": 1e-05,

防止rms为0要在除数上加的极小值

== "rope_theta": 10000.0,

rope的基频，$theta = 1/10000$
  
== "sliding_window": 4096

sliding_window 是 SWA（Sliding Window Attention，滑动窗口注意力） 的窗口大小

表现为在Attention Mask里填一个条状注意力mask

```
SWA 条状注意力演示（窗口大小 = 5）

横轴 = key 位置 j，纵轴 = query 位置 i
█ = 允许注意（在窗口内）
· = 被掩蔽（窗口外）

        j=0  1  2  3  4  5  6  7  8  9
i=0  [  █    █  █  ·  ·  ·  ·  ·  ·  ·  ]   ← 窗口左边界被截断
i=1  [  █    █  █  █  ·  ·  ·  ·  ·  ·  ]
i=2  [  █    █  █  █  █  ·  ·  ·  ·  ·  ]
i=3  [  ·    █  █  █  █  █  ·  ·  ·  ·  ]
i=4  [  ·    ·  █  █  █  █  █  ·  ·  ·  ]
i=5  [  ·    ·  ·  █  █  █  █  █  ·  ·  ]
i=6  [  ·    ·  ·  ·  █  █  █  █  █  ·  ]
i=7  [  ·    ·  ·  ·  ·  █  █  █  █  █  ]
i=8  [  ·    ·  ·  ·  ·  ·  █  █  █  █  ]
i=9  [  ·    ·  ·  ·  ·  ·  ·  █  █  █  ]   ← 窗口右边界被截断

窗口范围：j ∈ [i - 2,  i + 2]
```
  
== "tie_word_embeddings": false,

编码ids到token的embeddings和把token转换为概率分布的lm_heads不是同一块权重

== "use_cache": true,

推理时会用kv cache 

== "vocab_size": 32000

词表长度，和llama一样

= 组件

RMSNorm， RoPE都和LLaMa一模一样

== MistralMLP

和LLaMa的SwiGLU一模一样，手动设置成了bias=false

== MistralAttention

第一是加了分组注意力

定义是在文件`LlamaAttention` （万模之源） 多算了一个参数

`self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads` 虽然对LLaMa来说这个值是1，但是对Mistral来说这个值就是4了

会对k,v各进行一次repeat

```py
key_states = repeat_kv(key, module.num_key_value_groups)     # (b, 8, s, 128) → (b, 32, s, 128)
value_states = repeat_kv(value, module.num_key_value_groups)
```

repeat_kv函数是 

```py
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

先expand再reshape， 对应方式如下 

```
Q 头:   0  1  2  3 | 4  5  6  7 | 8 ...11 | ... | 28...31
KV 头:  0  0  0  0 | 1  1  1  1 | 2  2  2  2 | ... | 7  7  7  7
```

第二个是SWA，滑动窗口注意力

attention的实现没变，主要变的是生成mask的逻辑，从因果注意力换成条状的因果注意力了

实现的方法是实现一个高阶的and谓词函数套在常规causal_mask_function上，和条状的注意力and一下

```py
def sliding_window_overlay(sliding_window: int) -> Callable:
    """
    This is an overlay depicting a sliding window pattern. Add it on top of a causal mask for a proper sliding
    window mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx > q_idx - sliding_window

    return inner_mask

def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a sliding window mask.
    """
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)
```

同时kv cache也可以做到一边推理一边删除旧的

= 拼接

和llama的拼接方式一样

每一层：

R1=hidden_state RMSNorm -> Casual Attention -> +R1 residual =  R2 -> RMSNorm -> FFN -> +R2 residual

输入输出和计算loss的方法都一样


= 计算题

每一层的参数量是 $10.5 d^2 + 2d^2 + 2d^2 / 4 + 2d = 13d^2 + 2d$ （比llama多了$d^2$）

MLP的参数量 $3 times 4096 times 14336 = 176,160,768$

Attention的参数量 $2 times 4096 times 4096("q + o") + 2 times 4096 times 4096 / 4 ("v + k") = 41,943,040$

两个Norm $4096 times 2 = 8192$

加起来是 $218,112,000$，大约是218 MB 

32层一共有$6,979,584,000$

零碎权重（和llama一样）

最后一层Norm $4096$

一个token_embedding，一个lm_heads：$4096 times 32000 times 2 = 262,144,000$ 大约是262 MB

加和后$6,979,584,000 + 262,144,000 + 4096 = 7,241,732,096$

约$7.24B$

