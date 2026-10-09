#import "../template/template.typ": *

#show: docstyle

= Qwen/Qwen3-235B-A22B-Thinking-2507

#link("https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507")[huggingface: Qwen/Qwen3-235B-A22B-Thinking-2507]


本来想从qwen 2.5 moe讲的，但是找了下全网找不到一份官方的config参数（论文里的链接挂了），而且看了下qwen3新东西又比较少，qwen3 moe的moe基本就是qwen2.5-moe，所以就两个一起讲了

= 参数


== "attention_bias": false

Qwen 2.5没关，所以昨天的参数计算有点错误，每个qkvo proj是两份参数


== "attention_dropout": 0.0,
  
dropout推理不开，训练填的什么也不知道

== 词表相关

"bos_token_id": 151643
"eos_token_id": 151645
"vocab_size": 151936

（Qwen2.5 是 152064，看来是删掉了一些词）


== "decoder_sparse_step": 1

决定隔"decoder_sparse_step - 1层就用一个`MoeSparseMoeBlock`， =1的话表明每层都是`MoeSparseMoeBlock`

== "hidden_size": 4096

隐藏层维度， 反而比Qwen2.5 72B的8192要小了

== "intermediate_size": 12288

和无moe版本兼容的参数，实际上没用上，被下面的moe_intermediate_size取代了

和上面的数字刚好是1:3的关系，qwen3开始不填一些奇奇怪怪的intermediate_size和hidden_size比例了

== 注意力相关

"head_dim": 128
"num_attention_heads": 64
"num_key_value_heads": 4

这里注意`128 x 64 = 8192` 和 `4096`不相等，但是llama原始的注意力保证了head_dim缺省就是`hidden_size // num_attention_heads`，不缺省就是填入的设定值

最后`q_proj`的目标维度还会是 `head_dim x num_attention_heads`

`k`和`v`的目标维度是`head_dim x num_key_value_heads`

这种还是叫GQA，Gemma那种`num_key_value_heads = 1`的那种叫MQA，算是一个特例

== "hidden_act": "silu"

SwiGLU经典silu

== 层数和滑窗

"max_window_layers": 94
"num_hidden_layers": 94
"sliding_window": null
"use_sliding_window": false
  
在Qwen 2.5 SWA就类似一种废稿了，目前还是以参数（设为没有）的形式保留了下来
  
== "initializer_range": 0.02,

权重初始化标准差范围，也就是每一个参数在训练时初始化是从$N(0, 0.02)$里采样得到的

== "max_position_embeddings": 262144

其实上下文多长都无所谓，这个可能标注了模型训练时见过的最长上下文
  

  
== moe相关

"moe_intermediate_size": 1536
"num_experts": 128

没有搞出mixtral版本的宽而少的moe，这里是用很多个小专家拼起来的，但是总和也是有196608那么宽了（是hidden_size的48倍）


"norm_topk_prob": true,
mixtral根据挑选出来的专家进行重归一化是默认的，这里是加了一个开关，不过也是和mixtral一样的行为

"num_experts_per_tok": 8,
每个token过几个专家

"output_router_logits": false
推理不开，训练应该要开，旁路收集信息

"router_aux_loss_coef": 0.001,
和mixtral一样的训练参数，但是mixtral是0.02，这里小了很多


== "rms_norm_eps": 1e-06

防止rms除0加的极小值，和qwen 2.5一样


== rope相关

"rope_scaling": null,
"rope_theta": 5000000 （6个0，前面变成5）

虽然报告里提了yarn，但是模型卡没提，个人认为可能只用了DCA

或者其实训练时最长上下文是65536，然后靠yarn先翻4倍，然后再DCA支持1M上下文
  
== "tie_word_embeddings": false

token embedding和lm_head不是一个

== "use_cache": true

推理时用kv cache


= 组件

== attention的qk norm

比一般的q k proj多了一个`q_norm`和`k_norm`，代码如下

norm就是正常的RMSNorm，再借过来用一遍，但是是对每个head_dim做的，也就是说每个norm的$gamma$ 参数量是128 而不是Transformer里的和hidden_size一样大（在这里是4096）

这么做的动机是因为attention logits可能会爆炸

attention 的 logit 是 $(q dot k) / sqrt("head_dim")$,其量级正比于 ‖q‖·‖q‖。预训练后期 hidden states 里会出现 massive activations / outlier channels,导致 q、k 的范数持续增长 → logits 变得非常大 → softmax 饱和成接近 one-hot(attention entropy collapse)→ 梯度要么消失要么尖峰 → loss spike 甚至训练发散。这个问题在大模型、大 batch、高学习率、长上下文时尤其严重。

QK-Norm 直接把每个 head 的 ‖q‖、‖k‖ 钉在 ~sqrt(head_dim)·$|gamma|$($gamma$ 是可学习的 RMSNorm weight),logit 上界由此被约束住，且 scale 由 $gamma$ 显式学出，不再受 hidden state 幅度漂移的影响。

这一路线的谱系：ViT-22B 首次发现大模型必须加 qk-layernorm 才训得动，Chameleon 沿用，OLMo 2 系统验证了 QK-Norm 能消除 loss spike、从而去掉其他补丁式稳定手段。Qwen2 没有它，Qwen3 加上就是这个动机。

```py
class Qwen3Attention(LlamaAttention):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        super().__init__(config, layer_idx)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

== MoE模块

Qwen 2.5的MoE有一个shared expert，每个token除了会根据Router去查询专家以外，还会都过一遍shared expert（模仿deepseek moe） 


注意还为shared_expert加了一个可学习的门控`self.shared_expert_gate`，并且门控的输出还用`F.sigmoid` 钳制在`(0,1)`，乘上后和多专家的输出相加
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

Qwen 3放弃shared_expert，只押注多专家（和mixtral近乎完全一致了），相当于化繁为简了

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

= 拼接

和llama一样，然后router_logits和mixtral一样靠旁路收集

= 计算题

设 d = hidden_size = 4096

== 总参数量

每一层的组成：attention（带 qk norm）+ router + 128 个 expert + 两个 Norm。"decoder_sparse_step": 1，所以 94 层全是 MoE 层，没有 dense 层

*Attention*：$q$ 出来是 $128 times 64 = 8192 = 2d$，是 hidden_size 的两倍宽，q、o 各一份满的 $2 d^2$；k、v 因为 GQA（4 个 kv head）只有 $128 times 4 = 512$ 列，各 $d^2 / 8$

$2 d^2 ("q + o") + 2 times d^2 / 8 ("k + v") = 4.25 d^2 = 71,303,168$

再补上 Qwen3 特有的 q_norm + k_norm，各管一个 head_dim，$2 times 128 = 256$，attention 一共 $71,303,424$

*Router*：一块 $[128, d]$ 的参数，$128d = 524,288$（checkpoint 里的 `gate.weight`）

*128 个 expert*：每个 expert 是一个标准 SwiGLU，intermediate 是 $1536 = 3/8 d$，$3 times 3/8 d times d = 9/8 d^2 = 18,874,368$，是个小不点；复制 128 遍就是

$144 d^2 = 2,415,919,104$

*两个 Norm*：$2d = 8192$

每层合计 $71,303,424 + 524,288 + 2,415,919,104 + 8192 = 2,487,755,008$，写成 d 的形式是 $148.25 d^2 + 130d + 256$。expert 一项占了单层的 $144 / 148.25 approx 97%$——和 Mixtral 同款结构，只是把"8 个 3.5d 宽的大专家"换成了"128 个 3/8 d 宽的小专家"

94层一共 $2,487,755,008 times 94 = 233,848,970,752$

零碎部分：

最后一层 Norm $4096$

一个 token_embedding，一个 lm_head（"tie_word_embeddings": false 所以是两份）：$4096 times 151,936 times 2 = 1,244,659,712$

总参数 $233,848,970,752 + 1,244,659,712 + 4096 = 235,093,634,560$，大约 235B。和官方权重对齐：safetensors 全部分片的 total_size 是 470,187,269,120 字节，$div 2$（bf16）恰好等于这个数，一分不差

== 激活参数量

每个 token 在每个 MoE 层只走 8 个 expert，专家内部的稀疏度是 $8/128 = 1/16$，比 Mixtral 的 $2/8 = 25%$ 更加稀疏；但 attention、router、Norm 谁都要过

*每层激活*：attention $4.25 d^2 + 256$ + router $128d$ + 8 个 expert $8 times 9/8 d^2 = 9 d^2$ + Norm $2d$ $= 13.25 d^2 + 130d + 256 = 222,830,848$

只占每层参数的 $13.25 / 148.25 approx 8.9%$

94层一共 $222,830,848 times 94 = 20,946,099,712$

加上每个 token 必走的零碎（embedding + lm_head + 最后一层 Norm）

$20,946,099,712 + 1,244,659,712 + 4096 = 22,190,763,520$

大约 22.2B，符合名字里的 "A22B"，激活只有总量的 $22.2 / 235 approx 9.4%$。一个巧合：每层被激活的 8 个 expert 共 $150,994,944 = 2.25 times 8192^2$ 个参数，恰好是 Qwen2.5-72B 一整层 attention 的权重数（Qwen2.5 没关 attention_bias，qkv bias 另加 10,240，o_proj 没有 bias）。和 Mixtral 一样，虽然每个 token 只算 22B 的账，128 个 expert 谁都可能被选中，全得常驻显存，实际要装的是 235B