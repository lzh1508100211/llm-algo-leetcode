## 前置
### BatchNorm（批量归一化）

$$
\text{BatchNorm}(x) = \gamma \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta
$$
- $\mu_B$：当前 batch 的均值（沿 batch 维度计算）
- $\sigma_B^2$：当前 batch 的方差（沿 batch 维度计算）
- $\gamma, \beta$：可学习的缩放和偏移参数
- $\epsilon$：防止除零的小常数（如 `1e-5`）
### LayerNorm（层归一化）
$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} + \beta
$$
- $\mu_L$：当前样本的均值（沿特征维度计算）
- $\sigma_L^2$：当前样本的方差（沿特征维度计算）
- $\gamma, \beta$：可学习的缩放和偏移参数
- $\epsilon$：防止除零的小常数（如 `1e-5`）

## RMSNorm（均方根归一化）
$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x) + \epsilon}
$$
其中：
$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
$$
- $\text{RMS}(x)$：均方根（沿特征维度计算）
- $\gamma$：可学习的缩放参数（**无 beta**）
- $\epsilon$：防止除零的小常数（通常 `1e-6`，比 LayerNorm 小）
## 三者对比
| 归一化 | 减均值？ | 分母 | 可学习参数 | 统计维度 |
| :--- | :--- | :--- | :--- | :--- |
| **BatchNorm** | ✅ | $\sqrt{\sigma_B^2 + \epsilon}$ | $\gamma, \beta$ | 跨样本（dim=0） |
| **LayerNorm** | ✅ | $\sqrt{\sigma_L^2 + \epsilon}$ | $\gamma, \beta$ | 样本内（dim=-1） |
| **RMSNorm** | ❌ | $\text{RMS}(x) + \epsilon$ | 仅 $\gamma$ | 样本内（dim=-1） |
## 出现原因
研究发现，大模型的中间层均值通常已接近0，所以想要舍弃均值计算，提升前向和反向传播的计算速度。但是方差的计算又依赖均值的计算。想要替换掉使用方差进行计算。所以选择了均方根。
## RMSNorm 在 Block 里的位置
```
                   输入 x (B, T, D)
                         │
                         ▼
              ┌─────────────────────┐
              │     RMSNorm 1       │  ← 归一化，稳定数值
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Multi-Head Attention│  ← 词间交流，跨 T
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   残差连接 (+)      │  ← x = x + attn_out
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │     RMSNorm 2       │  ← 归一化，再次稳定
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │        MLP          │  ← 深度思考，D → 4D → D
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   残差连接 (+)      │  ← x = x + mlp_out
              └─────────────────────┘
                         │
                         ▼
                   输出 x (B, T, D)
```
放回 LLaMA block 里看，RMSNorm 是 attention / MLP 前的稳定器：
x ─► RMSNorm ─► Attention ─► residual add
h ─► RMSNorm ─► MLP       ─► residual add
它不改变 token 数，也不混合 token 之间的信息；它只让每个 token 自己的 hidden 向量尺度更稳定。
## RMSNorm 在 Block 里归一化什么
```
RMSNorm 作用在每个 token 的 hidden dimension 上，输入输出形状不变。
输入 x (B, T, D)
│
├─ token 0: [d0, d1, d2, ..., dD] ──► 计算 RMS ──► 除以 RMS ──► 乘以 gamma
├─ token 1: [d0, d1, d2, ..., dD] ──► 计算 RMS ──► 除以 RMS ──► 乘以 gamma
├─ token 2: [d0, d1, d2, ..., dD] ──► 计算 RMS ──► 除以 RMS ──► 乘以 gamma
│          ...（每个 token 独立处理，互不干扰）
└─ token T: [d0, d1, d2, ..., dD] ──► 计算 RMS ──► 除以 RMS ──► 乘以 gamma

输出 x (B, T, D) ← 形状完全不变

```
## 代码实现与混合精度 (AMP) 陷阱
**数学公式翻译成代码并不难，真正需要小心的是混合精度训练时的数值稳定性——FP16 下平方运算极易溢出，这里给出标准处理方案。**
在代码实现时，有一个非常关键的工程细节需要处理：**数值溢出 (Numerical Overflow)**。
> **工程经验：为什么要强制转换精度？** 现代大模型训练与推理几乎都会使用混合精度 (AMP) 或半精度格式 (`FP16`) 以节省显存。但我们需要注意，`FP16` 的最大安全数值仅为 `65504`。
> 在计算 RMSNorm 时，第一步是求输入张量的平方 ($x^2$)。如果输入特征中某个值大于 256（由于$256^2$ =65536>65504 ），该位置计算后就会溢出变为 `inf`（无穷大），进而导致损失函数出现 `NaN`，引发训练崩溃。
> **标准处理方案 (Upcasting)：** 无论模型输入是什么精度格式，在执行平方和均值操作前，通常需要显式地将其转换为 `float32` 计算。待归一化计算完毕后，再将结果转换回原有精度。这是深度学习框架中处理该算子的标准做法。
## 动手实战
```
import torch  
import torch.nn as nn  
class RMSNorm(nn.Module):  
    def __init__(self, hidden_size: int, eps: float = 1e-6):  
        super().__init__()  
        self.eps = eps  
        # ==========================================  
        # TODO 1: 定义可学习参数 weight，并初始化为全 1  
        # 形状: [hidden_size]  
        # 提示: 使用 nn.Parameter 包装张量使其可学习        
        # ==========================================        self.weight = nn.Parameter(torch.ones(hidden_size))  

    def _norm(self, x: torch.Tensor) -> torch.Tensor:  
        # ==========================================  
        # TODO 2: 实现 RMSNorm 核心计算逻辑  
        # 提示:  
        # 1. 为防止 FP16 溢出，需要在高精度下计算        
        # 2. 计算输入的均方值（平方后求均值），注意保持维度以便广播        
        # 3. 使用均方根的倒数进行归一化，torch.rsqrt 比 1/sqrt 更快        
        # 4. 返回归一化后的结果（保持高精度，便于后续操作）        
        # ==========================================        
        x_fp32 = x if x.dtype == torch.float32 else x.float()  
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)  
        # 此处有个代码实现的更优解，1.公式中是先开根号再加eps，2.代码实现是先加eps再开根号，当rsqrt(0)时，如果采用1的方案，最后计算出的值会很大x/eps,如果采用方案2，则结果会很小eps/x ，更有利于计算  
        return x_fp32 * torch.rsqrt(variance + self.eps)  
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        # ==========================================  
        # TODO 3: 先归一化，再缩放并转回输入精度  
        # 提示: 调用 _norm 进行归一化后，乘以可学习的 weight，最后转回输入精度  
        # ==========================================        
        weight = self.weight.to(x.dtype)  
        return (weight * self._norm(x)).to(x.dtype)
```
