#import "../template/template.typ": *

#show: docstyle 

= Mixtral-8x7B-v0.1 

#link("https://huggingface.co/mistralai/Mixtral-8x7B-v0.1")[huggingface: mistralai/Mixtral-8x7B-v0.1]

= 关键参数

==  token_id 

序列开始"bos_token_id": 1

序列结束"eos_token_id": 2

目测pad_token_id应该和llama一样是0，但没写

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

== "rope_theta": 1000000

$10^6$，这次比mistral直接多了两个0

== router结构参数

num_experts_per_tok: int = 2
num_local_experts: int = 8


主要用来定义`class MixtralTopKRouter(nn.Module)`

router是一块 `[num_local_expert, hidden_size]`的参数

最后会计算一个`[seq_len,num_experts_per_tok] `的专家idx来确定激活哪些专家

== moe训练参数





*router_jitter_noise: float = 0.0*
给 router 输入加噪声，防止路由早期“塌缩”到固定几个专家。默认 0.0 即关闭，mixtral自己是关闭。（可能训练开了也说不定）


*output_router_logits: bool = False*

决定各层 router 的 logits 是否被收集进输出（MoeModelOutputWithPast.router_logits）。看起来像调试开关，但它实际控制着 aux loss 的计算（modeling_mixtral.py:632）：为 True 时才调 load_balancing_loss_func，为True且有 labels 时把 coef $times$ aux_loss 加进总 loss（第 640 行），只开不训则单独返回 aux_loss 字段。

虽然config给关了，但是训练应该要打开。不然下一个参数就没作用了

*router_aux_loss_coef: float = 0.02*

上面那条的系数。aux loss（loss_utils.py 的 load_balancing_loss_func）度量的是“专家负载不均程度”：所有 token 都挤在同一个专家上时最大。没有它，router 会富者愈富——早期随机偏爱某个专家，那个专家被训得更好，之后被选得更多，最终塌缩成 dense 模型。系数控制这个约束和语言建模目标的权衡，调太大会伤主任务质量。

== "tie_word_embeddings": false,

编码ids到token的embeddings和把token转换为概率分布的lm_heads不是同一块权重

== "use_cache": true,

推理时会用kv cache 

== "vocab_size": 32000

词表长度，和llama一样

== "sliding_window": null

mixtral又不用swa了


= 组件

== MixtralTopKRouter

其实只有一份参数，大小是`[num_experts, hidden_dim]`，也就是`[8, 4096]`

先把`[batch, seq_len, hidden_dim]`摊平成token视角
`hidden_states = hidden_states.reshape(-1, self.hidden_dim)`

然后每个token对router进行大门，也就是`router_logits = F.linear(hidden_states, self.weight)` 大小是`[batch * seq_len, num_experts]`

然后把分数变成概率，注意输入进softmax的数据要转换成float32，得到`router_probs = torch.nn.functional.softmax(router_logits.float(), dim=-1)`

然后硬选择2个(self.top_k = config.num_experts_per_tok = 2)`router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)`

之后再选出来的2个上重新归一化，让得到的概率不受别的softmax出的概率的影响`router_top_value /= router_top_value.sum(dim=-1, keepdim=True)`

最后得到的`router_logits` 大小为`[batch * seq_len, 8]`
剩下的`router_scores`和`router_indices` 是`[batch * seq_len, 2]` 一个是`[0,1]`的概率，一个是下标


```py
class MixtralTopKRouter(nn.Module)
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = torch.nn.functional.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices
```


== MixtralSparseMoeBlock

先不看Experts，先看Router的输出怎么被利用

三个输出里，`router_logits`被丢弃了，只有`router_scores`和`router_indices` 作为

`top_k_weights`, `top_k_index` 输入到experts里面，他们的维度都是`[batch_size * seq_len, 2]`


```py
class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise
        self.gate = MixtralTopKRouter(config)
        self.experts = MixtralExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states
```


== MixtralExperts


expert是一个正常的SwiGLU，复制了8遍，也就是num_experts遍

`expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)` 的运算是把`top_k_index`的`[batch_size * seq_len, 2]`转换为`[batch_size * seq_len, 2, 9]`的`one_hot` 也就是把index （比如`[1,2]`这种）变成

```
[0, 1, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 1, 0, 0, 0, 0, 0, 0]
```

为什么是9不是8，大概是留个空位？

之后会对维度进行重排序，把`expert_mask`的维度变成`[9, 2, batch_size* seq_len]`

还会计算一个`expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()`，如果这个专家没用到这个前向就不运算了

`top_k_pos, token_idx = torch.where(expert_mask[expert_idx])`返回所有 True 元素的坐标，一维一个张量，按维度给。对形状 (2, N) 的布尔矩阵，返回一个二元组，每个元素都是一维 int64 张量

也就是说`top_k_pos`是0或者1，`token_idx`是对应的行号

然后用 行号把对应的token取出来`current_state = hidden_states[token_idx]`

进行经典的SwiGLU运算
```py
current_hidden_states = self.act_fn(gate) * up
current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
```

`current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]` 然后把hidden_states乘上对应的权重， 这里注意`top_k_weights[token_idx, top_k_pos]` 是一维向量`(n_e,)` 也就是用到这个专家的token个数，这里加一个None变成`[n_e, 1]`

这里是逐元素乘法，而非矩阵乘法，这里的广播方式就三种


1. 两个张量从最后一维向前逐维对齐；
2. 每一对维度：相等 → 逐元素配对；其中一个是 1 → 这个 1 被拉伸成对方的尺寸；缺失 → 当作 1；
3. 拉伸不复制内存，只是以步长 0 的方式“假装重复”。

如果n_e恰好是4096的话，那么会乘错，多加一维限制死了语义

然后把这些取出来的token加回去

`final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(dtype))` 这里第一个参数是dim，除了dim 0外维度必须完全一致（都是4096），语义是对每个 i 执行 final[token_idx[i]] += source[i]，把 n_e 行按行号散射累加进 N 行的大表。

```py
@use_experts_implementation
class MixtralExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
```


= 连接

主体部分和mistral一致，只不过没有SWA，连接方式仍然是

R1 -> Norm -> Attention -> +R1 R2-> Norm -> MoeMLP -> +R2 

重复32层，最后是一个Norm

不同的是多了一个辅助loss

== load_balancing_loss_func

这个函数的4个输入

`gate_logits` 就是 `MixtralTopKRouter`出的 router_logits
形状是一个tuple，每一层的都会保存，所以是一个长度为32的tuple，每一个元素是一个tensor，形状是`[batch_size * seq_len, 8]`，注意这些tensor都带着梯度

然后`num_experts=8`，`top_k=2`

还有`attention_mask`，这个标注用不到的token，形状是`[batch_size X sequence_length]`

主要算了两个东西

`f`: tokens_per_expert 统计 8 个专家各被“选中”了多少次（每个 token 贡献 2 次 top-k 选择），除以 token 数 → f_e = 专家 e 接到的分配占比。8 个数加起来 = top_k = 2。

`P`:routing_weights.sum(dim=0) / N → P_e = 全体 token 平均分给专家 e 的 softmax 概率(用route_logits过softmax求出来的概率)。8 个数加起来 = 1。

然后计算

$
  "loss" = E times sum_(i = 1)^(8) f_i times P_i
$


为什么这个乘积能压塌缩：只有当“硬选择集中在专家 e”且“软概率也堆在专家 e”同时成立时，f_e·P_e 才大。两个指标只要有一个均匀了，乘积就小。

完美均衡：f_e = 2/8 = 0.25，P_e = 1/8 → loss = 8 × 8 × 0.25 × 0.125 = 2（正好 = top_k，是下界）
完全塌缩（全挤 1 个专家）：f ≈ 1（那个专家的位置，可以认为第二个专家的概率被挤的近似0），P ≈ 1（那个专家的概率） → loss ≈ 8 × 1 × 1 = 8（上界）

实现上有一些注意的点：

1. 梯度只走 P，不走 f。 topk/bincount 是离散操作，f 对 router 权重没有梯度（被当常数）；P 来自 softmax，可导。所以梯度信号是“把概率推离那些已经很拥挤的专家”
2.  attention_mask 分支在做什么。 有 padding 时：flat_mask（N,）为 0/1，`scatter_add_` 只把真实 token 的分配计入f和P
3.  逐层累加而不是堆大张量。 注意开头的注释：32 层的中间结果如果摞成 (32, N, 8)，显存峰值是 O(L·N·E)；这里只维护三个 (8,) 累加器逐层加，峰值 O(N·E)

```py
def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    # Accumulate assignment counts and probability sums layer by layer, normalizing at the end,
    # so peak memory stays O(seq_len * num_experts) regardless of the number of layers.
    compute_device = gate_logits[0].device
    tokens_per_expert_sum = torch.zeros(num_experts, dtype=torch.float32, device=compute_device)
    router_prob_sum = torch.zeros(num_experts, dtype=torch.float32, device=compute_device)
    total_rows = 0.0

    if attention_mask is not None:
        # The same flat mask applies to every layer's [batch_size * sequence_length] rows.
        flat_mask = attention_mask.reshape(-1).to(device=compute_device, dtype=torch.float32)

    for layer_gate in gate_logits:
        routing_weights = torch.nn.functional.softmax(layer_gate.to(compute_device), dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        if attention_mask is None:
            # Count of top-k assignments per expert
            tokens_per_expert_sum = (
                tokens_per_expert_sum + torch.bincount(selected_experts.reshape(-1), minlength=num_experts).float()
            )
            # Sum of routing probabilities per expert
            router_prob_sum = router_prob_sum + routing_weights.float().sum(dim=0)
            total_rows = total_rows + routing_weights.shape[0]
        else:
            # Same reductions, weighted by the attention mask to exclude padding tokens
            tokens_per_expert_sum = tokens_per_expert_sum + torch.zeros(
                num_experts, dtype=torch.float32, device=compute_device
            ).scatter_add_(0, selected_experts.reshape(-1), flat_mask.repeat_interleave(top_k))
            router_prob_sum = router_prob_sum + (routing_weights.float() * flat_mask.unsqueeze(-1)).sum(dim=0)
            total_rows = total_rows + flat_mask.sum()

    tokens_per_expert = tokens_per_expert_sum / total_rows
    router_prob_per_expert = router_prob_sum / total_rows

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
```

*上下界的证明*

设$s_i$ 是第i个token花在它选中的那k个专家上的概率和，也就是

$s_i = sum_(e in "topk"_i) p_i (e)$

比如（E=8，k=2）

```
分布:  [0.10, 0.06, 0.50, 0.24, 0.04, 0.03, 0.02, 0.01]
                  └─e2─┘ └─e3─┘   ← 被选中的两个
s_i = 0.50 + 0.24 = 0.74
```

$s_i$的取值下界是$2/8=1/4$，上界是1

然后对于loss来说

$E times sum_(e = 1)^8 f_e P_e = E times 1/N sum_(i = 1)^N sum_(e in "topk"_i) p_i (e) = E times 1/N sum_(i = 1)^N s_i$

考虑$k/E lt.eq s_i lt.eq 1$ 可以得到

$
  E times 1/N times N times k/E & lt.eq "loss" lt.eq E times 1/N times N times 1 \
  k lt.eq "loss" lt.eq E
$


*梯度大小正比于该专家的当前负载*——专家越拥挤，把概率推向它的梯度阻力越大。这是一个经典的比例控制器（负反馈）：负载偏离均匀越多，修正力越强；负载均衡时各 $f_e$ 相等，对推力的抵抗均匀，系统稳定在均衡点。掉队专家不会被直接“拉”（没有 +f 的项），但挤在热门专家上的概率被压走，相对地就流向冷门专家。

= 计算题

设 d = hidden_size = 4096

== 总参数量

每一层的组成：attention + router + 8 个 expert + 两个 Norm

*Attention* 和 Mistral 完全一样：q、o 是满的 $d^2$，k、v 因为 GQA（8 个 kv head）只有 $d^2 / 4$

$2 d^2 ("q + o") + 2 times d^2 / 4 ("k + v") = 2.5 d^2 = 41,943,040$

*Router*：一块 $[8, d]$ 的参数，$8d = 32,768$，很小但每层都有一份（checkpoint 里的 `block_sparse_moe.gate.weight`）

*8 个 expert*：每个 expert 是一个标准 SwiGLU，$3 times 3.5d times d = 10.5 d^2 = 176,160,768$，恰好等于 Mistral 整个 FFN 的参数量；复制 8 遍就是

$84 d^2 = 1,409,286,144$

*两个 Norm*：$2d = 8192$

每层合计 $41,943,040 + 32,768 + 1,409,286,144 + 8192 = 1,451,270,144$，写成 d 的形式是 $86.5 d^2 + 10d$。MLP 一项就占了单层的 97%，这才是"8x7"的含义：被 $times 8$ 的只有 MLP，attention 和 Norm 没有跟着复制，所以总参数远到不了 56B

32层一共 $1,451,270,144 times 32 = 46,440,644,608$

零碎部分权重和 LLaMa 还有 Mistral 一样

最后一层 Norm $4096$

一个 token_embedding，一个 lm_heads（"tie_word_embeddings": false 所以是两份）：$4096 times 32000 times 2 = 262,144,000$ 大约是 262 MB

总参数 $46,440,644,608 + 262,144,000 + 4096 = 46,702,792,704$，大约 46.7B。和官方能对严：safetensors 全部分片的 total_size 是 93,405,585,408 字节，$div 2$（bf16）恰好等于这个数。router 很容易漏算——漏掉的话会算成 $46,439,596,032 + 262,144,000 + 4096 = 46,701,744,128$，差出来的 $1,048,576$ 恰好是 32 层 $times 32,768$ 的 gate

== 激活参数量

每个 token 在每个 MoE 层只走 2 个 expert，但 attention、router、Norm 谁都要过

*每层激活*：attention $2.5 d^2$ + 2 个 expert $2 times 10.5 d^2 = 21 d^2$ + Norm $2d$ + router $8d$ $= 23.5 d^2 + 10d = 394,305,536$

只占每层参数的 $23.5 / 86.5 approx 27%$：expert 内部的稀疏度是 $2/8 = 25%$，attention 全走所以整体略高

32层一共 $394,305,536 times 32 = 12,617,777,152$

加上每个 token 必走的零碎（embedding + lm_heads + 最后一层 Norm）

$12,617,777,152 + 262,144,000 + 4096 = 12,879,925,248$

大约 12.9B，和官方说的每 token 12.9B 激活参数对上了。所以 Mixtral 是 46.7B 总参、12.9B 激活：每个 token 的计算量和一个 13B dense 模型相当，但 8 个 expert 没法预测会不会被选中，都得常驻显存，实际要装的是 46.7B





