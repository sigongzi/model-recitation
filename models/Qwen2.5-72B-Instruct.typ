#import "../template/template.typ": *

#show: docstyle 
= Qwen/Qwen2.5-72B-Instruct

#link("https://huggingface.co/Qwen/Qwen2.5-72B-Instruct")[huggingface: Qwen/Qwen2.5-72B-Instruct]

提到了推理要用Yarn

add to config.json
```
{
  ...,
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
  }
}
```

然后对于小模型的1M上下文版本提到了DCA参数
```
"dual_chunk_attention_config": {
  "chunk_size": 262144,
  "local_size": 8192,
  "original_max_position_embeddings": 262144
}
```
#link("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M/blob/main/config.json")[Qwen/Qwen2.5-7B-Instruct-1M]

这里面"sliding_window"没有72B的大，但是max_window_layers>=num_hidden_layers本身就没有SWA，所以这个参数不会被用到

两边是并列的两种方案，各提一个正好：72B靠注入yarn扩到128K；1M版不靠yarn（config里没有也不需要rope_scaling，rope_theta直接提到1e7），靠DCA达到1M

下面主要说72B的参数
= 参数

== token_id

序列开始："bos_token_id": 151643,
  
序列结束 "eos_token_id": 151645,

pad没写


== "hidden_act": "silu",
  
SwiGLU经典激活函数

== "hidden_size": 8192,
  
每个token的隐藏维度

== "initializer_range": 0.02

权重初始化标准差范围，也就是每一个参数在训练时初始化是从$N(0, 0.02)$里采样得到的

  
== "intermediate_size": 29568

MLP的中间层维度，和hidden size的比值是3.609375，整数比231:64

== 上下文

设置里写的是"max_position_embeddings": 32768

但是推理可以把rope改成yarn调大


== attention heads


"num_attention_heads": 64,
"num_key_value_heads": 8,

mistral 同款GQA，但是是kv:q是1:8

head_dim仍然是`8192 // 64 = 128`

== 混合窗口

"num_hidden_layers": 80,
"max_window_layers": 70,

如果use_sliding_window为true的情况下，那么70层到79层（从0标号）都会是SWA，但是use_sliding_window是false

== SWA

"sliding_window": 131072,
"use_sliding_window": false,

设置了但是没用上

== "rms_norm_eps": 1e-06,

为了防止rms是0在分母上加的极小值
  
== rope相关

"rope_theta": 1000000.0, 有6个0
  
== "tie_word_embeddings": false, 

lm_head和token_embedding不是一个  

== "use_cache": true,

推理时用

== "vocab_size": 152064

词表大小，加了一些中文短语进去


= 组件

== 混合 滑动窗口注意力+全注意力(实际没用上)

多了一个参数是 "max_window_layers"，但是post_init里面设置的是

```
if self.layer_types is None:
  self.layer_types = [
      "sliding_attention"
      if self.sliding_window is not None and i >= self.max_window_layers
      else "full_attention"
      for i in range(self.num_hidden_layers)
  ]
```

对于#link("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json")[Qwen/Qwen2.5-7B-Instruct]来说，这两组参数是

都是full attention
```
"max_window_layers": 28
num_hidden_layers": 28
```

对于#link("https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/config.json")[Qwen/Qwen2.5-14B-Instruct]来说，这两组参数是

也都是full attention
```
"max_window_layers": 70,
"num_hidden_layers": 48,
```

对于#link("https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/config.json")[Qwen2.5-32B-Instruct]来说，这两个参数是

都是full_attention
```
"max_window_layers": 70,
"num_hidden_layers": 64,
```

一直到最大的#link("https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json")[Qwen/Qwen2.5-72B-Instruct]，才终于max_windows_layers比num_hidden_layers要小，用了大概10层的SWA

```
"max_window_layers": 70
"num_hidden_layers": 80,
```

但是与此同时"use_sliding_window": false，说明设置了也没用，都是full Attention

可能是某个实验里的结果，发现不好用就给去了，总之都是full_attention

不过马上就要流行SSM和transformer混合了，SWA不用就不用了吧

== Qwen2MLP

经典的SwiGLU，没什么好说的，说一下全尺寸的hidden_size和intermediate_size比值

对于qwen来说，2.67这个启发式的值已经不奏效了，全尺寸的比值也都不一样

#link("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json")[Qwen/Qwen2.5-0.5B-Instruct]

比值是：38/7 ≈ 5.4286，整数比值是：38:7
```
"hidden_size": 896
"intermediate_size": 4864
```

#link("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json")[Qwen/Qwen2.5-1.5B-Instruct]

比值是：35/6 ≈ 5.8333，整数比值是：35:6
```
"hidden_size": 1536
"intermediate_size": 8960
```

#link("https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json")[Qwen/Qwen2.5-3B-Instruct]

比值是：43/8 = 5.375，整数比值是：43:8
```
"hidden_size": 2048
"intermediate_size": 11008,
```

#link("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json")[Qwen/Qwen2.5-7B-Instruct]
比值是：37/7 ≈ 5.2857，整数比值是：37:7
```
"hidden_size": 3584,
"intermediate_size": 18944,
```

#link("https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/config.json")[Qwen/Qwen2.5-14B-Instruct]
比值是：27/10 = 2.7，整数比值是：27:10
```
"hidden_size": 5120
"intermediate_size": 13824,
```

#link("https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/config.json")[Qwen/Qwen2.5-32B-Instruct]

比值是：27/5 = 5.4，整数比值是：27:5
```
"hidden_size": 5120
"intermediate_size": 27648
```

#link("https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json")[Qwen/Qwen2.5-72B-Instruct]

比值是：231/64 = 3.609375，整数比值是：231:64
```
"hidden_size": 8192
"intermediate_size": 29568
```

hidden_size全系都是128的整数倍：

```
0.5B   896 = 128 × 7
1.5B  1536 = 128 × 12
3B   2048 = 128 × 16
7B   3584 = 128 × 28
14B  5120 = 128 × 40
32B  5120 = 128 × 40
72B  8192 = 128 × 64
```

除了0.5B以外，这个倍数正好就是num_attention_heads，因为head_dim全系固定是128。0.5B是14个head，`896 // 14 = 64`，head_dim减半，所以倍数是head数的一半

intermediate_size其实也全系都是128的倍数，所以上面的整数比才能约得这么干净

== Yarn RoPE 

上面的"max_position_embeddings"参数是 32768，然后要把上下文扩展到128K，需要注入参数：

```
"rope_scaling": {
  "factor": 4.0,
  "original_max_position_embeddings": 32768,
  "type": "yarn"
}
```

注入的type会改变

```py
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)
```

注入的type会改变`rope_init_fn`，换成了`_compute_yarn_parameters`


对于Qwen2.5-72B，计算的参数是

```py
base = 1_000_000                # rope_theta
partial_rotary_factor = 1.0     # Qwen2 不做 partial rotary，.get 缺省
head_dim = 128                  # getattr(config,"head_dim", 8192//64)
dim = int(128 * 1.0) = 128      # → 64 个频率维度

factor = 4.0                    # 注入的
attention_factor = None
mscale = None
mscale_all_dim = None
original_max_position_embeddings = 32,768
```

beta数值，这两个都是默认数值，在 32K 内至少转 32 圈的维度算高频，转不满 1 圈的算低频
```py
beta_fast = 32
beta_slow = 1
```

计算两条曲线
```py
pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
inv_freq_extrapolation = 1.0 / pos_freqs # 纯外推（原频率）： 1e6^(-2i/128) 长度 64
inv_freq_interpolation = 1.0 / (factor * pos_freqs) # 纯插值： inv_freq_extrapolation / 4
```

两个端点的物理量：extrap[0]=1.0，波长 6.28 token (波长是$2pi$ 除频率)，32K 内转 5215 圈（高频，管局部）；extrap[63]=1.24e-6，32K 内只转 0.0065 圈（低频，管远距粗定位）。


```py
truncate = config.rope_parameters.get("truncate", True)
low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))
def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
      """Find dimension range bounds based on rotations"""
      low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
      high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
      if truncate:
          low = math.floor(low)
          high = math.ceil(high)
      return max(low, 0), min(high, dim - 1)
```

这里面的函数`find_correction_dim`是在求解一个反问题，找到转1圈和转32圈的dim维度各是多少

这个方程是

（求解的是$i$，注意$2i/dim$等价于$i/(dim/2)$，也就是说是在64个维度中求解的)
$
  "max_position_embeddings" dot "base"^(-(2i)/dim) &= "rot" dot 2 pi \
  -(2i)/"dim" ln("base") + ln("max_position_embeddings") & = ln("rot" dot 2 pi) \
  "dim" times (ln("max_position_embeddings") - ln("rot" dot 2 pi))/(2 times ln("base")) &= i
$

代入一下并且truncate后`[floor(23.60), ceil(39.65)]` = `[23, 40]`

```py
# Get n-dimensional rotational scaling corrected for extrapolation
inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
inv_freq = (
    inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
    + inv_freq_extrapolation * inv_freq_extrapolation_factor
)
# 其中linear ramp factor函数如下
def linear_ramp_factor(min, max, dim):
  if min == max:
      max += 0.001  # Prevent singularity

  linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
  ramp_func = torch.clamp(linear_func, 0, 1)
  return ramp_func
return inv_freq, attention_factor
```

可以发现linear_ramp_factor在23及之前是0，在40及之后是1，根据混合的代码，计算的factor又用1.0 - 将其对0.5对称了一下

所以混合后的`inv_freq`，在[0,23]是原样，在[40, 63]是除4，在`[24, 39]`根据你factor混合

#table(
  columns: 6,
  table.header([dim], [gamma], [extrap], [interp], [最终inv_freq],[处理]),
  [0],[1.000],[1], [0.25],[1],[不变],
  [23],[1.000],[6.98e-3],[1.74e-3],[6.98e-3],[原样],
  [32],[0.471],[1e-3],[2.5e-4],[6.03e-4],[混合],
  [40],[0.000],[1.78e-4],[4.45e-5],[4.45e-5],[$div$ 4],
  [63],[0.000],[1.24e-6],[3.10e-7],[3.10e-7],[$div$ 4]
)


完整代码如下：


```py
def _compute_yarn_parameters(
    config: "PreTrainedConfig",
    device: Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://huggingface.co/papers/2309.00071)

    Args:
        config ([`~transformers."PreTrainedConfig"`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`, *optional*): The base wavelength from which the inverse frequencies will be derived. Defaults to `config.default_theta` if omitted.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.
            *   max_position_embeddings (`int`): The maximum length of the positional embeddings.
            *   rope_parameters (`dict[str, float | int]`): The standard RoPE scaling parameters, from which the following
                keys will be accessed:
                *   `attention_factor` (`float`, *optional*): The scaling factor to be applied to the computed cos/sin.
                    If None, the value is inferred from `factor`, `mscale`, and `mscale_all_dim` as available.
                *   `beta_fast` (`float`, *optional*, defaults to 32): Parameter to set the boundary for extrapolation
                    (only) in the linear ramp function.
                *   `beta_slow` (`float`, *optional*, defaults to 1): Parameter to set the boundary for interpolation
                    (only) in the linear ramp function.
                *   `factor` (`float`, *optional*): The scaling factor applied when interpolating the position IDs to
                    extend the possible context length. Additionally, if `attention_factor` is None, the log of this
                    value is used to compute a value for `attention_factor`, possibly in conjunction with `mscale` and
                    `mscale_all_dim`, if provided.
                *   `mscale` (`float`, *optional*): If `attention_factor` is None and both `mscale` and
                    `mscale_all_dim` are provided, `mscale` acts scalar augmenting `log(factor)` when computing the
                    numerator for the inferred value of `attention_factor`. If not provided, `attention_factor` will be
                    calculated based on `factor` only.
                *   `mscale_all_dim` (`float`, *optional*): If `attention_factor` is None and both `mscale` and
                    `mscale_all_dim` are provided, `mscale_all_dim` acts scalar augmenting `log(factor)` when computing
                    the denominator for the inferred value of `attention_factor`. If not provided, `attention_factor`
                    will be calculated based on `factor` only.
                *   `original_max_position_embeddings` (`int`): The original max position embeddings used during pretraining.
                *   `truncate` (`bool`, *optional*): Whether to truncate the correction range.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*, defaults to 1.0): If less than 1.0, inverse frequencies
                will be returned for the first fraction of the head_dim.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    factor = rope_parameters_dict["factor"]
    attention_factor = rope_parameters_dict.get("attention_factor")
    mscale = rope_parameters_dict.get("mscale")
    mscale_all_dim = rope_parameters_dict.get("mscale_all_dim")
    original_max_position_embeddings = rope_parameters_dict["original_max_position_embeddings"]

    # NOTE: DeepSeek-V3 (and potentially other models) have `original_max_position_embeddings` field
    # containing the pretrained value. They use the ratio between `max_position_embeddings` and this value
    # to compute the default attention scaling factor, instead of using `factor`.
    if factor is None:
        factor = config.max_position_embeddings / original_max_position_embeddings

    def get_mscale(scale, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = rope_parameters_dict.get("beta_fast") or 32
    beta_slow = rope_parameters_dict.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        """Find dimension range bounds based on rotations"""
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
    # to expand the possible context length. In other words, interpolation = apply scaling factor.
    pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    truncate = config.rope_parameters.get("truncate", True)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor
```

*为什么混合* 

RoPE本质

RoPE 下 q·k 的每一维贡献只依赖相对距离的相位：Δθ = (m−n) × inv_freq[i]。把 64 个维度想成 64 根速度不同的钟表指针（波长 6.28 → 506 万 token，速度差 80 万倍）。模型判断“这两个 token 离多远、该不该关注”，读的是 64 根指针的联合读数

训练时，模型只在相对距离 0…32768 内校准过这组指针。扩展到 128K 时，两种极端修法各有明确的病灶：

如果factor不变的话（纯用`inv_freq_extrapolation`）

dim63 在 32K 内只转 0.0065 圈，训练时相位只见过 0 - 0.0407 rad 这一小段单调爬升。距离拉到 500K，相位到 0.62 rad——超出训练见过的范围 15 倍。这维对模型来说输出了一个从没学过的值，64 根针的联合读数变成从未见过的组合，直接外推到训练外会崩。

如果factor全部除4

把所有频率 ÷4，慢针确实被折回训练域，但快针也被压了：dim0 上相邻 token 的角度差从 1 rad 掉到 0.25 rad。“前一个 token 要高权重”这类局部模式，模型是在 1 rad 的刻度间距上学出来的，现在整个标尺被压扁 4 倍，局部注意力失焦——短文本质量受损。这就是 PI（Position Interpolation）的代价。

* YaRN 的混合：按“训练圈内转了几圈”分工*

两根针的信息内容不同

#table(
  columns: 3,
  table.header([], [快针（≥32 圈，dim 0–23）],[慢针（< 1 圈，dim 40–63）]),
  [训练内相位], [转了几千圈，所有相位都密集见过], [单调爬了 0.04 rad，是个里程表],
  [携带的信息],[局部精度（区分相邻 token）], [远距粗定位（这段隔得远不远）],
  [压缩的代价], [毁掉局部标尺，不可接受],[它本来就没有局部信号可毁，无损],
  [不处理的代价], [无（周期化，天然外推安全）],[相位冲出训练域，不可接受]
)

== DCA 

以下的参数根据#link("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M/blob/main/config.json")[Qwen/Qwen2.5-7B-Instruct-1M] 谈论，hidden_size是3584 $28 times 128$

这个hugging face里面没有，下载了qwen更改的#link("https://github.com/QwenLM/vllm")[QwenLM/vllm] 才找到源码（下面的`dev/dual-chunk-attn`分支）


DCA不是为了减轻计算设计的，而是为了能让只在262144长度上微调的模型能直接适应1M的上下文

公开的参数`chunk_size=262144, local_size=8192`， `chunk_len = 262144 − 8192 = 253952`，对于1010000的上下文
是3个整块+248144的一个零碎块

这两个参数的含义如下

#table(
  columns: 3,
  table.header([参数],[语义],[在 1M 配置中的值]),
  [chunk_size], [相对距离预算：任何注意力路径允许的最大 RoPE 距离，应等于模型训练长度（262144）],[262144],
  [local_size], [边界精确窗口宽度：块首之后多少个 token 对上一块保持“连续计数”的精确距离],[8192]
)

工程上，k只有一份，形状为`[1010000, 4, 128]`

维护三份q，分别是`q`,`q_succ`,`q_inter`，形状为`[1010000, 28, 128]`。默认路径和decode只用这3份；rope实际还会多产出两个critical变体`q_succ_critical`/`q_inter_critical`，位置不饱和（分别是`p + 253952`和`p + 262144`），只在sparse attention路径（取每个chunk末尾last_q个token做稀疏采样计算）里用到

对于某部分的q，比如最后一块，也就是`[761856, 1010000)`这块

它要做的是把attention分3路，第一部分是intra block的，也就是`q[761856:]` 和 `k[761856:]` 做attention （扩张k，再做矩阵乘法）

第二部分是succ block，也就是它的前一块，这里的职责是`q_succ[761856:]` 和`k[507904: 761856]` 相乘

第三部分是succ block之前所有，叫inter，这里的职责是`q_inter[761856:]` 和 `k[:507904]` 相乘

如果q和k上加上的位置编码都是正常的 `position x inv_freq`，那么这和普通的attention没有区别，但是由于模型没有见过1010000长的上下文，为的就是把RoPE编码的 $delta theta = (m - n) theta_i$(m是q的位置，n是k的位置) 的这个$(m - n)$限制住

设$t$为真实的位置，$p = t % C$，也就是位置对块长取模后的位置，取值是$[0, 253951]$

这里面的k加上的编码是$p theta_i$(i是0到63) 其中C就是chunk_len，这里是`253952`

对于`q`这个向量，加的编码和k一样都是$p theta_i$

然后对于`q_succ`这个向量，加的编码是$min(p + C, 262144)$，$p gt.eq 8192$后这个值就会饱和在262144

对于`q_inter`，恒定是262144

所以说，q只能感知到这个k是在块内，或者已经跨块了，这样就把$(m - n)$限制在训练看过的范围内了

这个主要是发现比较远的信息和精确的位置数值无关，只和qk的具体数值有关，所以就都设置成了离得远就采用最大值来编码

另外对于长序列还会有一个补偿 $alpha = max(0.1 dot ln("len" / 262144) + 1, 1)$，但它不是乘在k上。cache里的k从头到尾都是"裸"的，没有温度系数；这个$alpha$在prefill时乘在softmax_scale上，decode时乘在当前的q上（三个q都乘），效果等价于每步attention的logits统一乘上当前长度的$alpha$

如果是800000长度的序列，那么$alpha = 0.1 dot ln(800000 / 262144) + 1 = 1.1116$

然后decode的时候序列逐渐变长，如果当前的len是800000，那么三个q也都会乘上这个$alpha = 0.1 dot ln(800000/262144) + 1 = 1.1116$

如果q生成后逐渐变长，那么在下次生成的时候，会乘上一个新的$alpha$，假如已经长到了上限，那么就是$alpha = 0.1 dot ln(1010000 / 262144) + 1 = 1.1349$



每步的时间线（len 含新 token）：

#table(
  columns: 5,
  table.header([步], [len], [$alpha(t) = 0.1 dot ln("len"/262144) + 1$],[新加入的k], [该步q乘的]),
  [1], [800,001],[1.11156],[$1.11156 times K_"new"_1$(就只乘在新加的这1个token上，不改变之前的token)], [$times 1.11156$],
  [5000],[805,000],[1.11244],[$1.11244 times K_"new"_(5000)$],[$times 1.11244$],
  [...], [...],[缓慢变大],[...], [...],
  [210,000],[1,010,000],[1.13489],[$1.13489 times K_"new"_(210000)$],[$times 1.13489$]
)

= 拼接

没有什么好说的，和llama完全一样，虽然有sliding_windows的分支，但是Qwen2.5所有家族都不启用

= 计算题

计算qwen家族要特地查询intermediate_size，比较随机

设`intermediate_size`为 i，hidden_size还是d，再设$r = "num_attention_heads" / "num_key_value_heads"$（每个kv head对应几个q head）

那么计算公式是

$2d + d^2 times 2 + d^2 times 2/r ("o + q + k + v proj") + 3 times i d ("gate + up + down")$

*qwen家族的GQA比例不是固定的1:8*，每个尺寸都得单独查config.json，顺手全查了一遍：

#table(
  columns: 4,
  table.header([模型], [num_attention_heads], [num_key_value_heads], [kv:q]),
  [0.5B], [14], [2], [1:7],
  [1.5B], [12], [2], [1:6],
  [3B], [16], [2], [1:8],
  [7B], [28], [4], [1:7],
  [14B], [40], [8], [1:5],
  [32B], [40], [8], [1:5],
  [72B], [64], [8], [1:8],
)

72B恰好$r = 8$，k、v才各是$d^2 / 8$；14B和32B只有5:1，GQA省得比72B少（k、v是$d^2 / 5$）。不过kv head数倒是全系只有2、4、8三档，head_dim除了0.5B以外都是128，所以真正随机的只有intermediate_size和这个比例

代入72B的参数：$d = 8192$，$i = 29568$，kv head只有8个，所以k、v都是$d^2 / 8$

MLP的参数量 $3 times 8192 times 29568 = 726,663,168$

Attention的参数量 $2 times 8192 times 8192 ("q + o") + 2 times 8192 times 1024 ("k + v") = 150,994,944$

两个Norm $8192 times 2 = 16,384$

每一层加起来是 $726,663,168 + 150,994,944 + 16,384 = 877,674,496$ 大约是877 M

写成d的形式是 $2.25d^2 + 3 i d + 2d$，又因为$i / d = 231 / 64$，也就是约$13.08d^2 + 2d$（attention因为GQA比llama省了$1.75d^2$，但MLP被随机的intermediate_size撑大了，单层反而比llama的$12d^2 + 2d$还重）

80层一共有 $877,674,496 times 80 = 70,213,959,680$ 大约是70.2 B

零碎权重

最后一层Norm $8192$

一个token_embedding，一个lm_head（"tie_word_embeddings": false所以是两份）：$8192 times 152064 times 2 = 2,491,416,576$ 大约是2.49 B （因为词表和hidden size都长了，所以加起来就比较大了）

加和后 $70,213,959,680 + 2,491,416,576 + 8192 = 72,705,384,448$

约$72.7B$，和名字里的72B对上了


