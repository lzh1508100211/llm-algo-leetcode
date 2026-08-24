## 场景举例
### 场景：你是 CEO，要听汇报
你有一个问题要问，**4 个部门经理（4 个注意力头）** 需要帮你分析。
### MHA（多头注意力）：每个经理配一个助理
**每个经理都有自己的助理，负责查资料（K）和整理内容（V）。**
```
你（Q）提问 → 每个经理都有自己的想法
              │
    ┌─────────┼─────────┬─────────┐
    ▼         ▼         ▼         ▼
  销售部     市场部     研发部     财务部
  (Q1)      (Q2)      (Q3)      (Q4)
    │         │         │         │
  自己的     自己的     自己的     自己的
  助理(K1,V1) 助理(K2,V2) 助理(K3,V3) 助理(K4,V4)
```
**结果：**
-   每个经理都有自己的助理（4 个助理）
-   你听了 4 份独立的汇报
-   **效果好，但成本高（4 个助理的工资）**
**对应：** MHA = 每个头有自己的 Q、K、V
### GQA（分组查询注意力）：几个经理共用一个助理
**你决定：销售部和市场部共用一个助理，研发部和财务部共用一个助理。**
```
你（Q）提问 → 每个经理都有自己的想法
              │
    ┌─────────┼─────────┬─────────┐
    ▼         ▼         ▼         ▼
  销售部     市场部     研发部     财务部
  (Q1)      (Q2)      (Q3)      (Q4)
    │         │         │         │
    └──┬──────┘         └──┬──────┘
       ▼                    ▼
   助理A (K1,V1)       助理B (K2,V2)
   (服务销售和市场)    (服务研发和财务)
```
**结果：**
-   只有 2 个助理（省了 2 个工资）
-   销售和市场用同一份资料（K），但各自有不同的分析角度（Q）
-   **效果不错，成本中等**
**对应：** GQA = Q 独立，K、V 分组共享
### MQA（多查询注意力）：所有经理共用一个助理
**你决定：只请 1 个超级助理，服务所有 4 个经理。**
```
你（Q）提问 → 每个经理都有自己的想法
              │
    ┌─────────┼─────────┬─────────┐
    ▼         ▼         ▼         ▼
  销售部     市场部     研发部     财务部
  (Q1)      (Q2)      (Q3)      (Q4)
    │         │         │         │
    └─────────┴──┬──────┴─────────┘
                 ▼
          超级助理 (K0, V0)
          (服务所有部门)
```
**结果：**
-   只有 1 个助理（最省钱）
-   所有经理用同一份资料（K），各自有不同的分析角度（Q）
-   **速度快，成本最低，但可能不够精细**
**对应：** MQA = Q 独立，K、V 全部共享
### 对比表
| 场景 | 经理数（Q）| 助理数（K,V）| 成本 | 效果 |
|--|--|--|--|--|
|**MHA** | 4 个 | 4 个助理 | 💰💰💰 最高 | ⭐⭐⭐ 最好|
|**GQA**| 4 个 | 2 个助理 | 💰💰 中等| ⭐⭐ 不错 |
| **MQA** | 4 个 | 1 个助理 | 💰 最低 | ⭐ 可以 |

### 再看“KV Cache”（助理的笔记）
在生成下一个词时，助理需要记住之前查过的资料（K 和 V），不用每次都重新查。
| 类型 | 助理数 | 需要记的笔记量 | 速度 |
| -- | -- | -- | -- |
| **MHA** | 4 个助理 | 4 份笔记 | 慢（要查 4 份）|
| **GQA** | 2 个助理 | 2 份笔记 | 中等 |
| **MQA** | 1 个助理 | 1 份笔记 | 快（只查 1 份）|
## KV Cache
在自回归生成中，每次生成第 $N$  个 Token 时，我们需要计算它与前面 $N - 1$  个 Token 的相关性。为了避免重复计算前 $N - 1$  个 Token 的特征，我们将其投影后的 Key 和 Value 张量缓存 (Cache)在显存中，当前步直接拼接读取。
然而，读取巨量的 KV Cache 会面临严重的**显存容量瓶颈**和**内存带宽瓶颈 (Memory-bound)**，导致推理极慢。
**从 MHA 到 GQA：大模型架构的进化**
-   **MHA (Multi-Head Attention)**: 标准的多头注意力。每个 Query 头都有自己专属的 Key 和 Value 头。即  $n$个 Q 头对应 $n$ 个 KV 头。KV Cache 占用最大（与 Q 头数成正比），推理时显存压力最大，但表达能力最强。
-   **MQA (Multi-Query Attention)**: 所有的 Query 头共享**同一个** Key 和 Value 头。即 $n$ 个 Q 头对应 1 个 KV 头。KV Cache 占用大幅减少（单层仅为 MHA 的 $\frac{1}{n}$），但由于 KV 表达能力锐减，模型效果往往打折扣。
-   **GQA (Grouped-Query Attention)**: LLaMA-2/3 采用的折中方案。将 Query 头分组，每组共享一个 Key 和 Value 头。即 $n$ 个 Q 头对应 $g$  个 KV 头（$1 < g < n$， $g$为组数）。KV Cache 占用介于 MHA 和 MQA 之间（单层为 MHA 的 ），在模型效果和显存占用之间取得了良好的工程平衡。
**MLA：DeepSeek 的极致 KV 压缩方案**
-   MLA (Multi-Head Latent Attention) 是 DeepSeek-V2/V3 采用的注意力机制，核心思路是不再缓存完整的 K/V 张量，而是缓存压缩后的潜在向量，计算时再实时解压。
**不同架构的 KV Cache 对比（单 Token 单层）**
| 架构 | KV Cache 大小 | 代表模型 |
|--|--|--|--|
| MHA | ~32 KB | 原始 Transformer |
| GQA | ~4 KB | LLaMA-2/3 |
| MLA | ~1.13 KB | DeepSeek-V2/V3 |
相比 MHA，MLA 的 KV Cache 缩减约 28 倍，这也是 DeepSeek 能支持 1M 上下文的工程基础之一。
**注意**：GQA 的压缩来自"共享头数"，MLA 的压缩来自"压缩维度"，两条路线可以结合使用。如果你对 MLA 的数学原理和代码实现感兴趣，可以查阅 DeepSeek-V2 技术报告。
## 核心公式与张量维度
**经过线性投影后，注意力计算公式：**

$$ 
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V 
$$

**张量变化**
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
      RoPE 注入位置信息(从03 RoPE章节拷贝来的，下方代码实例没有此步骤)
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
## 代码实例
```
import torch  
import torch.nn as nn  
import math  
import torch.nn.functional as F  
from debugpy.launcher import output  
  
  
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:  
    """  
    将 KV 头复制 n_rep 次，以匹配 Query 头的数量 (GQA/MQA 需要)    当 n_rep == 1 时（即 MHA），直接返回原张量。    
    """    
    batch, num_kv_heads, slen, head_dim = hidden_states.shape  
    if n_rep == 1:  
        return hidden_states  
    #hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)  
    hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)  
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)  
class GroupedQueryAttention(nn.Module):  
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int = None):  
        super().__init__()  
        self.hidden_dim = hidden_dim  
        self.num_heads = num_heads  
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads  
  
        # 确保 num_heads 能被 num_kv_heads 整除  
        assert num_heads % self.num_kv_heads == 0, \  
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"  
  
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads  
        self.head_dim = hidden_dim // num_heads  
  
        # 定义投影矩阵  
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)  
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)  
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)  
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)  
  
    def forward(  
        self,  
        x: torch.Tensor,  
        attention_mask: torch.Tensor = None,  
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None  
    ):  
        """  
        前向传播。  
        Args:            
	        x: 输入张量，形状 [batch, seq_len, hidden_dim]            
	        attention_mask: 注意力掩码，形状应为 [batch, 1, 1, seq_len]（因果掩码）                        或 [batch, 1, seq_len, seq_len]，会广播到 scores。            
	        kv_cache: 缓存的 (K, V) 张量，用于自回归生成。  
        Returns:            
	        输出张量 [batch, seq_len, hidden_dim]，更新后的 KV Cache        
		"""        
		batch_size, seq_len, _ = x.shape  
  
        # 1. 线性投影  
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)  
  
        # ==========================================  
        # TODO 1: Reshape xq, xk, xv 以适配多头注意力计算  
        # 提示: 先把最后一维拆成 [num_heads, head_dim] / [num_kv_heads, head_dim]（使用 reshape 或 view），  
        # 再将seq_len和num_heads换位，得到 [B, num_heads, S, head_dim] / [B, num_kv_heads, S, head_dim]        # ==========================================        xq = xq.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  
        xk = xk.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  
        xv = xv.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  
        # ==========================================  
        # TODO 2: 处理 KV Cache  
        # 提示: 如果有 cache，将历史 KV 拼接在当前 KV 的 seq_len 维度前  
        # 注意: 拼接维度是 dim=2 (seq_len)        
        # ==========================================        
        if kv_cache is not None:  
            k_cache, v_cache = kv_cache  
            xk = torch.cat((k_cache, xk), dim=2)  
            xv = torch.cat((v_cache, xv), dim=2)  
  
        new_kv_cache = (xk, xv)  
  
        # 通过 repeat_kv 把 GQA 的 KV 头数扩充到和 Query 数量一致  
        xk = repeat_kv(xk, self.num_queries_per_kv)  
        xv = repeat_kv(xv, self.num_queries_per_kv)  
  
        # ==========================================  
        # TODO 3: 计算注意力分数 (Scaled Dot-Product)  
        # 公式: scores = Q @ K^T / sqrt(head_dim)  
        # 提示: 使用 torch.matmul，并对 K 转置最后两维        
        # 注意: attention_mask 形状为 [batch, 1, 1, seq_len]（因果掩码），        
        #       会广播到 scores 的 [batch, num_heads, seq_len, seq_len]        
        # ==========================================             
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)  
  
  
        if attention_mask is not None:  
            scores = scores + attention_mask  
  
         probs = F.softmax(scores, dim=-1)  
        output = torch.matmul(probs, xv)  
  
  
        # ==========================================  
        # TODO 4: 恢复形状并输出  
        # [B, H, S, D] -> [B, S, H*D]  
        # 提示: transpose + contiguous + view        
        # ==========================================        
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)  
        return self.o_proj(output), new_kv_cache
```
```
# 运行此单元格以测试你的实现  
def test_mha_mqa_gqa():  
    try:  
        batch_size, seq_len, hidden_dim, num_heads = 2, 16, 128, 4  
  
        # 1. 测试 MHA  
        print("Testing MHA (Multi-Head Attention)...")  
        mha = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=num_heads)  
        x = torch.randn(batch_size, seq_len, hidden_dim)  
        out, _ = mha(x)  
        assert out.shape == (batch_size, seq_len, hidden_dim), "MHA 输出形状错误!"  
  
        # 2. 测试 GQA  
        print("Testing GQA (Grouped-Query Attention)...")  
        gqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=2)  
        out, _ = gqa(x)  
        assert out.shape == (batch_size, seq_len, hidden_dim), "GQA 输出形状错误!"  
  
        # 3. 测试 KV Cache  
        print("Testing KV Cache Autoregressive Decoding...")  
        prefill_len = 5  
        x_prefill = torch.randn(batch_size, prefill_len, hidden_dim)  
        _, kv_cache = mha(x_prefill)  
  
        x_decode = torch.randn(batch_size, 1, hidden_dim)  
        out_decode, new_kv_cache = mha(x_decode, kv_cache=kv_cache)  
        assert new_kv_cache[0].shape == (batch_size, num_heads, prefill_len + 1, hidden_dim // num_heads), "KV Cache 更新错误!"  
  
        print("\n✅ All Tests Passed! Attention 算子实现通过测试。")  
    except NotImplementedError:  
        print("请先完成 TODO 部分的代码！")  
        raise  
    except (AttributeError, NameError, TypeError, ValueError) as e:  
        if isinstance(e, AttributeError):  
            print("代码未完成，无法找到必要的属性")  
        elif isinstance(e, NameError):  
            print("代码可能未完成，导致了变量未定义")  
        elif isinstance(e, TypeError):  
            print("代码可能未完成，导致了类型错误")  
        else:  
            print("代码可能未完成，导致了张量维度错误")  
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e  
    except Exception as e:  
        print(f"\n❌ 测试失败，请检查张量维度: {e}")  
        raise  
test_mha_mqa_gqa()
```
