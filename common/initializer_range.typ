#import "../template/template.typ":*
#show: docstyle

= initializer range

核心思路是：*让每一层前向传播后的方差不要爆炸，也不要消失。*

== 推导直觉

考虑一个线性层，忽略 bias:

 
假设：
权重 $w_(i,j)$ 独立同分布，均值为方差为$sigma^2$
输入 $x_j$ 也独立同分布，均值为方差为 $"Var"(x)$ 
权重和输入相互独立。
那么：
$"Var"(y) = n_("in") dot "Var"(w) dot "Var"(x)$

我们希望前向传播后尺度不变，也就是：$"Var"(y) approx "Var"(x)$

于是得到：
$n_"in" dot "Var"(w) approx 1$

所以：
$"Var"(w) approx 1/n_("in")$
 
标准差就是：$sigma(w) approx sqrt(1 / n_("in"))$

== 反向传播


反向传播时，梯度会经过权重矩阵的转置传播回来。

$"Var"((partial L) /(partial x)) = n_"out" dot "Var"(W) dot "Var"(delta^(y))$

我们同样希望梯度不爆炸，不消失

也就是：$"Var"((partial L) /(partial x)) approx "Var"(delta^(y))$

类似推导可以得到：$n_"out" dot "Var"(w) approx 1$

也就需要：
$"Var"(w) approx 1/n_"out"$
 
前向要求 $1/n_"in"$，反向要求 $1/n_"out"$，两者通常不能同时满足。Xavier 的做法是取折中：

$"Var"(w) = 2/(n_"in" + n_"out")$
 
对应标准差：
$sigma(w) = sqrt(2 / (n_"in" + n_"out"))$

这就是 Xavier 初始化的经典形式。

== 那 He 初始化为什么是  $sqrt(2 / n_"in")$

Xavier 推导时假设激活函数近似线性、关于 0 对称，比如 tanh。但 ReLU 会把负半轴清零，大约让方差减半。为了补偿这个损失，He 初始化把方差放大 2 倍：

$"Var"(w)= 2/n_"in"$

所以标准差是：

$sigma(w)= sqrt(2/n_"in")$ 
这就是 Kaiming/He 初始化。

== 代入算一下

大部分其实都是0.02

最原始的llama，hidden size = 4096

$sqrt(2 / 4096) = 0.022097$