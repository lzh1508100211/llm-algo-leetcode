## 生活例子：你站在钟表盘上

想象你站在一个巨大的钟表盘中央，每个词（比如“猫”、“吃”、“老鼠”）都站在表盘的不同位置上。
-   **“猫”** 指向 **3 点钟方向**
-   **“吃”** 指向 **12 点钟方向**
-   **“老鼠”** 指向 **9 点钟方向**
**关键思想：**
-   模型不需要知道“猫在几号位置”，只需要知道“猫和吃的指针差了多少度”，就能判断它们在句子里的相对距离。
-   **角度差越大，词离得越远；角度差越小，词离得越近。**
## 一张图说清 RoPE
```
原本的词向量（一个方向）：
         ↑
         │
     ────┼────→
         │


加入位置信息后（旋转了一个角度）：
         ↗
        │
    ────┼────→
        │

位置1: 旋转 20°
位置2: 旋转 40°
位置3: 旋转 60°
位置4: 旋转 80°
...


注意力计算时：
词1（20°）和词2（40°）→ 角度差 = 20° → 离得近
词1（20°）和词4（80°）→ 角度差 = 60° → 离得远
```
## 核心思想与痛点
**为什么需要 RoPE？** 
原生的 Transformer 使用绝对位置编码（如正弦波或可学习参数），导致模型很难泛化到比训练集更长的序列。我们希望模型能在计算 Attention 时感知到 Token 之间的**相对距离**。 
**RoPE 的本质：** “借用复数的旋转”。通过将 Query 和 Key 的向量映射到复数空间并旋转特定角度，在计算内积（Dot-product）时，结果自然就带有了相对位置信息 。其中， 是 Query 的位置， 是 Key 的位置，两者之差  就是它们之间的相对距离——RoPE 通过复数旋转让 Attention 内积的结果只依赖于这个差值，从而让 Attention 内积的结果依赖于 Token 间的相对位置。
### 所处的注意力机制的位置
```
输入 x
    │
    ▼
┌─────────────────────────────────────────────┐
│  1. 生成 Q、K、V                           │
│     Q = x × W_Q                            │
│     K = x × W_K                            │
│     V = x × W_V                            │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  2. ⭐ RoPE 在这里！⭐                       │
│     旋转 Q 和 K，加入位置信息               │
│     Q_rotated = RoPE(Q, positions)         │
│     K_rotated = RoPE(K, positions)         │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  3. 计算注意力分数                         │
│     scores = Q_rotated × K_rotated^T       │
│     现在分数里已经包含了位置信息！          │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  4. Softmax + × V                          │
└─────────────────────────────────────────────┘
```
### 三种注意力机制
#### 核心公式（基准）
无论有没有位置编码，Attention 的计算核心都是：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

#### 无位置编码（原始版本）
直接使用词嵌入 $X$ 生成 Q、K、V。

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

代入核心公式得到注意力分数：

$$
\text{Score} = \frac{Q K^T}{\sqrt{d_k}}
= \frac{(X W_Q) (X W_K)^T}{\sqrt{d_k}}
$$

**特点**：完全没有位置信息，交换词的顺序后结果不变。
#### Sinusoidal 位置编码（原始 Transformer）
位置编码 $P$ 直接加到词嵌入 $X$ 上。

$$
Q = (X + P) W_Q, \quad K = (X + P) W_K, \quad V = (X + P) W_V
$$

代入核心公式：

$$
\text{Score} = \frac{Q K^T}{\sqrt{d_k}}
= \frac{((X + P) W_Q) ((X + P) W_K)^T}{\sqrt{d_k}}
$$

**特点**：模型知道每个词的绝对位置编号，但难以直接捕获词之间的相对距离。
#### RoPE 旋转位置编码（LLaMA / 现代大模型）
位置信息通过旋转矩阵 $R_\theta$ 作用于 Q 和 K，不改变 V。
位置 $m$ 的旋转矩阵为 $R_\theta(m)$，位置 $n$ 的旋转矩阵为 $R_\theta(n)$：

$$
Q_m = (X_m W_Q) \cdot R_\theta(m), \quad 
K_n = (X_n W_K) \cdot R_\theta(n), \quad 
V_n = X_n W_V
$$

代入核心公式：

$$
\text{Score}_{m,n} = \frac{Q_m K_n^T}{\sqrt{d_k}}
= \frac{(X_m W_Q) R_\theta(m) \cdot \left( (X_n W_K) R_\theta(n) \right)^T}{\sqrt{d_k}}
$$

利用旋转矩阵的性质，化简为：

$$
\boxed{\text{Score}_{m,n} = \frac{(X_m W_Q) (X_n W_K)^T \cdot \cos(m - n)}{\sqrt{d_k}}}
$$

**特点**：分数直接依赖 $(m - n)$，模型能清楚感知两个词之间的相对距离，外推能力更强。
#### 综合对比
| 方法 | 公式 | 位置信息形式 | 能否感知相对距离 |
| :--- | :--- | :--- | :--- |
| 无位置编码 | $\text{Score} \propto (X W_Q) (X W_K)^T$ | ❌ 无 | ❌ 不能 |
| Sinusoidal | $\text{Score} \propto ((X+P) W_Q) ((X+P) W_K)^T$ | ✅ 绝对位置 | ❌ 很难 |
| RoPE | $\text{Score} \propto (X W_Q) (X W_K)^T \cdot \cos(m-n)$ | ✅ 旋转角度 | ✅ 能（含 $m-n$） |
## RoPE 的张量变化
```
      输入阶段
┌─────────────────────────────────────────────────────────────────┐
│  x: [B, T, D]                                                  │
│  ↓                                                             │
│  Q = x @ W_Q  →  [B, T, D]                                    │
│  K = x @ W_K  →  [B, T, D]                                    │
│  V = x @ W_V  →  [B, T, D]                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
      多头拆分阶段
┌─────────────────────────────────────────────────────────────────┐
│  Q → reshape → [B, T, H, Dh]   (D = H × Dh)                  │
│  K → reshape → [B, T, H, Dh]                                  │
│  V → reshape → [B, T, H, Dh]                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
      RoPE 注入位置信息
┌─────────────────────────────────────────────────────────────────┐
│  Q_rot = RoPE(Q)  →  [B, T, H, Dh]   ← 只改数值，不改形状     │
│  K_rot = RoPE(K)  →  [B, T, H, Dh]                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
      注意力打分
┌─────────────────────────────────────────────────────────────────┐
│  scores = Q_rot @ K_rot.T  →  [B, H, T, T]                   │
│  (每个 head 独立计算 T×T 的注意力矩阵)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
      Softmax + 因果 Mask
┌─────────────────────────────────────────────────────────────────┐
│  weights = Softmax(scores / √Dh)  →  [B, H, T, T]            │
│  (因果模型：右上三角用 mask 置为 -inf)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
      加权求和
┌─────────────────────────────────────────────────────────────────┐
│  attn_out = weights @ V  →  [B, H, T, Dh]                    │
│  (每个 token 的新表示 = 所有 token V 的加权平均)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
      多头合并
┌─────────────────────────────────────────────────────────────────┐
│  attn_out → reshape → [B, T, D]                              │
│  attn_out = attn_out @ W_O  →  [B, T, D]                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   输出 hidden states
              ┌─────────────────────────┐
              │  x = x + attn_out       │  ← 残差连接
              │  形状: [B, T, D]        │
              └─────────────────────────┘
```
## RoPE（旋转位置编码）数学公式
### 1. 频率公式

$$\theta_i = 10000^{-2i/d}, \quad i \in [0, 1, 2, ..., d/2-1]$$

其中：
- $d$ = head_dim（每个注意力头的维度）
- $i$ = 维度对索引

特点：
- $i=0$ 时，$\theta_0 = 1$（最快旋转）
- $i=d/2-1$ 时，$\theta_{d/2-1} = 10000^{-(d-2)/d}$（最慢旋转）


### 2. 角度计算

对于位置 $m$ 和频率 $\theta_i$：

$$\text{angle}(m, i) = m \times \theta_i$$

其中：
- $m \in [0, 1, 2, ..., T-1]$（位置索引）
- $i \in [0, 1, 2, ..., d/2-1]$（频率索引）


### 3. 复数旋转因子

$$\text{freqs\_cis}(m, i) = e^{i \times \text{angle}(m, i)}$$

展开为欧拉公式：

$$\text{freqs\_cis}(m, i) = \cos(m \times \theta_i) + i \times \sin(m \times \theta_i)$$


### 4. 应用旋转到 Q 和 K

将向量拆分为实部和虚部：

$$x = [x_0, x_1, x_2, x_3, ..., x_{d-2}, x_{d-1}]$$

拆成两半：
- 实部：$[x_0, x_2, x_4, ..., x_{d-2}]$
- 虚部：$[x_1, x_3, x_5, ..., x_{d-1}]$

复数表示：

$$x_{\text{complex}} = (x_0 + i x_1), (x_2 + i x_3), ..., (x_{d-2} + i x_{d-1})$$

应用旋转：

$$x_{\text{rotated}}(m) = x_{\text{complex}}(m) \times \text{freqs\_cis}(m)$$


### 5. 注意力分数（核心）

未使用 RoPE：

$$\text{Score}(m, n) = Q(m) \cdot K(n)$$

使用 RoPE：

$$\text{Score}(m, n) = Q_{\text{rot}}(m) \cdot K_{\text{rot}}(n)$$

关键推导结果：

$$\text{Score}(m, n) = Q(m) \cdot K(n) \times \cos(m - n)$$


### 6. 形状对照表

| 符号 | 含义 | 形状 |
| :--- | :--- | :--- |
| $\theta_i$ | 频率 | `[d//2]` |
| $\text{angle}(m, i)$ | 旋转角度 | `[T, d//2]` |
| $\text{freqs\_cis}(m, i)$ | 复数旋转因子 | `[T, d//2]` |
| $Q_{\text{rot}}$ | 旋转后的查询 | `[B, T, H, d]` |
| $K_{\text{rot}}$ | 旋转后的键 | `[B, T, H, d]` |
| $\text{Score}(m, n)$ | 注意力分数 | `[B, H, T, T]` |


### 7. 核心结论

1. RoPE 不改变向量形状：应用前后都是 `[B, T, H, d]`
2. 位置信息编码在角度里：每个词被旋转 $m \times \theta_i$ 弧度
3. 模型通过角度差感知距离：$\cos(m-n)$ 直接出现在注意力分数中
4. 多频率覆盖多尺度：快频率看近处，慢频率看远处
## 代码实现
```
import torch  
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):  
    """  
    预计算复数旋转因子 freqs_cis。  
    Args:        
	    dim: head_dim，必须为偶数        
	    end: 序列长度        
	    theta: 基数，默认 10000  
    Returns:        freqs_cis: 形状为 [end, dim//2] 的复数张量    
    """    
    # ==========================================    
    # TODO 1: 用极坐标生成复数张量 (提示: torch.polar)  
    # 1. 计算逆频率向量 inv_freq = 1/(theta ** (2j/d))  
    #   torch.arange(0, dim, 2) 步长为 2，对应公式中的 2j    
    # 2. 生成位置索引 t = [0, 1, ..., end-1]    
    # 3. 计算角度矩阵 angles = outer(t, inv_freq)    
    # 4. 用 torch.polar 生成复数 e^{i * angles}    
    # ==========================================    
    assert dim % 2 == 0, f"Head dimension must be even for RoPE, got {dim}"  
    # 生成逆频率向量：对应公式中的 theta_j = 10000^{-2j/d}  
    # torch.arange(0, dim, 2) 步长为 2，对应公式中的 j 索引    
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  
    t = torch.arange(end, device=inv_freq.device, dtype=torch.float32)  
    # 位置与频率的外积：angles[m, j] = m * theta_j  
    angles  = torch.outer(t, inv_freq)  
    # 生成复数旋转因子 e^{i * angles}  
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  
    return freqs_cis  
  
# 将频率张量扩展到可广播形状，供 Step 3 的复数乘法使用  
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):  
    """  
    将 freqs_cis 变形为与 x 广播对齐的形状。  
    假设 x 的形状为 [batch, seq_len, heads, head_dim//2]（复数形式），    
    将 freqs_cis 从 [seq_len, head_dim//2] 变形为 [1, seq_len, 1, head_dim//2]。
	"""    
    ndim = x.ndim  
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]  
    return freqs_cis.view(*shape)  
  
def apply_rotary_emb(  
    xq: torch.Tensor,  
    xk: torch.Tensor,  
    freqs_cis: torch.Tensor,  
) -> tuple[torch.Tensor, torch.Tensor]:  
    """  
    将旋转位置编码应用到 Query 和 Key 上  
    Args:        
	    xq: [batch, seq_len, heads, head_dim]        
	    xk: [batch, seq_len, heads, head_dim]        
	    freqs_cis: [seq_len, head_dim//2]，预计算的旋转因子    
	Returns:        旋转后的 xq, xk，形状与输入一致    
	"""    
	# ==========================================    
	# TODO 2: 将 xq, xk 从实数张量转为复数张量  
    # 提示: 先把最后一维拆成两个一组，再转成复数  
    #   1. 提升精度到 FP32: .float()    
    #   2. 将最后一维 head_dim 拆分为 (-1, 2)，其中 2 对应实部和虚部    
    #   3. 用 torch.view_as_complex 转为复数    
    # 提示：reshape(*xq.shape[:-1], -1, 2) 保留前面所有维度，最后变为 (..., -1, 2) 
	# ==========================================    
	xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  
  
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)  
    # 确保类型一致  
    freqs_cis = freqs_cis.to(xq_complex.dtype)  
  
  
    # ==========================================  
    # TODO 3: 进行复数乘法，并转回实数张量  
    # 步骤：  
    #   1. 复数乘法: xq_complex * freqs_cis（自动广播）    
    #   2. 用 torch.view_as_real 转回实数，形状变为 (..., 2)    
    #   3. 用 .flatten(-2) 将最后两维合并回 head_dim    
    #   4. 用 .type_as(xq) 恢复为输入的数据类型    
    # ==========================================    
    # 复数乘法 (a+bi)*(c+di) 自动实现旋转矩阵的效果    
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)  
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)  
  
    # 恢复为输入的 dtype（如 BF16）  
    return xq_out.type_as(xq), xk_out.type_as(xk)
```
```
# 运行此单元格以测试你的实现
def test_rope():
    try:
        print("=" * 60)
        print("开始测试 RoPE 旋转位置编码")
        print("=" * 60)

        batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64

        # Test 1: 形状测试
        print("\n【Test 1】形状测试")
        xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
        xk = torch.randn(batch_size, seq_len, num_heads, head_dim)

        freqs_cis = precompute_freqs_cis(head_dim, seq_len)
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

        assert xq_out.shape == xq.shape, f"Query 输出形状错误: 期望 {xq.shape}, 实际 {xq_out.shape}"
        assert xk_out.shape == xk.shape, f"Key 输出形状错误: 期望 {xk.shape}, 实际 {xk_out.shape}"
        assert freqs_cis.shape == (seq_len, head_dim // 2), f"频率张量形状错误"
        
        # 核心修复：防止占位符作弊，输出绝不能等于输入
        assert not torch.allclose(xq, xq_out, atol=1e-5), "TODO 3 未完成: 输出与输入完全相同，RoPE 旋转未生效！"
        
        print("  ✅ 输出形状测试通过")
        print("  ✅ 频率张量形状测试通过")

        # Test 2: 数值范围测试
        print("\n【Test 2】数值范围测试")
        norm_before = torch.norm(xq, dim=-1)
        norm_after = torch.norm(xq_out, dim=-1)
        assert torch.allclose(norm_before, norm_after, rtol=1e-4, atol=1e-5), "RoPE 改变了向量模长！"
        print("  ✅ 向量模长保持不变（旋转不变性）")

        assert not torch.isnan(xq_out).any(), "输出包含 NaN！"
        assert not torch.isinf(xq_out).any(), "输出包含 Inf！"
        print("  ✅ 无 NaN/Inf 数值异常")

        # Test 3: 相对位置编码验证
        print("\n【Test 3】相对位置编码验证")
        pos0 = xq_out[:, 0, :, :]
        pos1 = xq_out[:, 1, :, :]
        assert not torch.allclose(pos0, pos1, rtol=1e-3), "不同位置的输出相同，位置编码失败！"
        print("  ✅ 位置编码生效（不同位置输出不同）")

        # Test 4: 精度稳定性测试
        print("\n【Test 4】精度稳定性测试")
        xq_fp16 = torch.randn(1, 8, 2, head_dim, dtype=torch.float16)
        xk_fp16 = torch.randn(1, 8, 2, head_dim, dtype=torch.float16)
        freqs_fp16 = precompute_freqs_cis(head_dim, 8)

        xq_out_fp16, xk_out_fp16 = apply_rotary_emb(xq_fp16, xk_fp16, freqs_fp16)

        assert xq_out_fp16.dtype == torch.float16, "输出类型错误！"
        assert not torch.isnan(xq_out_fp16).any(), "FP16 输入导致 NaN！"
        print("  ✅ FP16 输入处理正确")
        print("  ✅ 精度提升机制工作正常")

        print("\n" + "=" * 60)
        print(" RoPE 算子实现通过测试。")
        print("   所有测试用例均已通过")
        print("=" * 60)

    except NotImplementedError:
        print("\n❌ 测试失败: 请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError) as e:
        print(f"\n❌ 测试失败: 代码可能未完成")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 发生未知异常: {type(e).__name__}: {e}")
        raise

test_rope()

```
