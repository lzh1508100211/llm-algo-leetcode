## Transform架构
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           原始 Transformer 架构                           │
│                          （Vaswani et al., 2017）                         │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────┐
                              │    Input Tokens     │
                              │   ["我", "爱", "你"] │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │   Token Embedding   │
                              │   [B, T] → [B,T,D]  │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  + Positional Enc   │
                              │  （正弦/余弦编码）   │
                              └──────────┬──────────┘
                                         │
                                         ▼
                         ╔═══════════════════════════════╗
                         ║      编码器（Encoder）× N      ║
                         ║                               ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  Multi-Head Attention   │  ║
                         ║  │  （每个头独立 Q,K,V）   │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  Add & LayerNorm        │  ║
                         ║  │  （Post-Norm）          │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  Feed-Forward Network   │  ║
                         ║  │  （ReLU, 4D → D）      │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  Add & LayerNorm        │  ║
                         ║  │  （Post-Norm）          │  ║
                         ║  └─────────────────────────┘  ║
                         ╚═══════════════════════════════╝
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  编码器输出 (K, V)   │
                              │  传给解码器          │
                              └──────────┬──────────┘
                                         │
                                         ▼
                         ╔═══════════════════════════════╗
                         ║      解码器（Decoder）× N      ║
                         ║                               ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  Masked MHA             │  ║
                         ║  │  （只看到之前的位置）   │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  Add & LayerNorm        │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  Cross-Attention        │  ║
                         ║  │  Q: 来自解码器         │  ║
                         ║  │  K,V: 来自编码器       │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  Add & LayerNorm        │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  Feed-Forward           │  ║
                         ║  │  （ReLU, 4D → D）      │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  Add & LayerNorm        │  ║
                         ║  └─────────────────────────┘  ║
                         ╚═══════════════════════════════╝
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │      LM Head        │
                              │  [B,T,D]→[B,T,Vocab]│
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │      Softmax        │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │   Output Tokens     │
                              └─────────────────────┘
```
## LLAMA架构
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLaMA 架构（仅解码器）                             │
│                     （Meta, 2023 - 开源大模型标杆）                        │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────┐
                              │    Input Tokens     │
                              │   ["我", "爱", "你"] │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │   Token Embedding   │
                              │   [B, T] → [B,T,D]  │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  RoPE 位置编码       │
                              │  （旋转 Q 和 K）    │
                              └──────────┬──────────┘
                                         │
                                         ▼
                         ╔═══════════════════════════════╗
                         ║       LLaMA Block × N        ║
                         ║        （如 32 层）          ║
                         ║                               ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  RMSNorm                │  ║
                         ║  │  （Pre-Norm）          │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  Multi-Head Attention   │  ║
                         ║  │  （MHA / GQA）         │  ║
                         ║  │  + RoPE 旋转           │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  残差连接 (+)            │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  RMSNorm                │  ║
                         ║  │  （Pre-Norm）          │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  SwiGLU（MLP）          │  ║
                         ║  │  D → 8/3D → D          │  ║
                         ║  │  （SiLU 激活）          │  ║
                         ║  └───────────┬─────────────┘  ║
                         ║              │                 ║
                         ║              ▼                 ║
                         ║  ┌─────────────────────────┐  ║
                         ║  │  残差连接 (+)            │  ║
                         ║  └─────────────────────────┘  ║
                         ╚═══════════════════════════════╝
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │    Final RMSNorm    │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │      LM Head        │
                              │  [B,T,D]→[B,T,Vocab]│
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │      Softmax        │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │   Output Tokens     │
                              └─────────────────────┘
```
## LLaMA vs 原始 Transformer 架构对比表

| 对比维度 | 原始 Transformer（2017） | LLaMA（Meta，2023） | 改进说明 |
| :--- | :--- | :--- | :--- |
| **模型结构** | 编码器-解码器（Encoder-Decoder） | **仅解码器（Decoder-Only）** | 文本生成任务不需要编码器，结构更简洁 |
| **归一化位置** | 后置归一化（Post-Norm） | **前置归一化（Pre-Norm）** | Pre-Norm 训练更稳定，不需要复杂预热 |
| **归一化方法** | LayerNorm（减均值 + 除方差） | **RMSNorm（只除均方根）** | 省掉均值计算，更快，分布式通信更少 |
| **位置编码** | 正弦/余弦固定编码（Sinusoidal） | **RoPE（旋转位置编码）** | 外推能力强，能处理比训练时更长的文本 |
| **激活函数** | ReLU | **SwiGLU（门控线性单元）** | 门控机制，表达能力更强，负值区保留梯度 |
| **MLP 升维比例** | 4 倍（D → 4D → D） | **8/3 倍（D → 8/3D → D）** | 配合 SwiGLU 两路结构，保持参数量相当 |
| **注意力机制** | MHA（多头注意力） | **MHA / GQA（分组查询注意力）** | GQA 减少 KV Cache，推理速度更快 |
| **Cross-Attention** | ✅ 有（编码器 → 解码器） | ❌ **无** | 仅解码器结构不需要 |
| **线性层偏置** | 有（bias=True） | **无（bias=False）** | 省参数，残差连接已提供偏移能力 |
| **参数量级** | 65M ~ 200M | **7B ~ 70B+** | 大模型 + 大数据驱动 |
| **训练数据量** | 几 GB | **几 TB（1.4T~2T tokens）** | 数据决定模型能力上限 |
| **典型代表** | BERT、GPT-2 | **LLaMA-1/2/3、Mistral** | LLaMA 已成为现代开源大模型标准 |
## 核心公式与架构
**1. SwiGLU MLP:**
$$ \text{SwiGLU}(x) = (\text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}} $$
    其中 $\\text{Swish}(z) = z \\cdot \\sigma(z)$ (在 PyTorch 中对应 `F.silu`)，其中 `⊙` 表示逐元素乘（Hadamard product）。注意，为了保持参数量与传统 MLP 一致，LLaMA 中的隐藏层维度通常设置为 $\\frac{8}{3} d$ 并向上取整（约 2.67 倍），其中 d 为 hidden_dim。
    **2. Decoder Layer 残差连接 (Residual Connections):**
$$ h = x + \text{Attention}(\text{RMSNorm}(x)) $$
    $$ \text{out} = h + \text{MLP}(\text{RMSNorm}(h)) $$
    *注意：这里的 Attention 内部包含了 RoPE 旋转位置编码（作用于 Q 和 K 投影后）*
## 代码实战
```
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
# ---------------------------------------------------------  
# 以下是我们之前实现的组件 (此处用极简占位符代替，以保持代码整洁)  
# ---------------------------------------------------------  
class DummyRMSNorm(nn.Module):  
    """占位 RMSNorm，仅用于验证结构，不做真实归一化。"""  
    def __init__(self, dim): super().__init__(); self.w = nn.Parameter(torch.ones(dim))  
    def forward(self, x): return x * self.w  
  
class DummyAttention(nn.Module):  
    def __init__(self, dim): super().__init__(); self.proj = nn.Linear(dim, dim)  
    def forward(self, x): return self.proj(x) # 假装它做了 RoPE 和 GQA  
# ---------------------------------------------------------  
  
class LlamaMLP(nn.Module):  
    def __init__(self, hidden_size: int, intermediate_size: int):  
        super().__init__()  
        # ==========================================  
        # TODO 1: 定义 SwiGLU 所需的三个线性层 (无 bias)  
        # 提示: gate_proj / up_proj 将 hidden_size 映射到 intermediate_size  
        #      down_proj 将 intermediate_size 映射回 hidden_size        
        # ==========================================        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)  
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)  
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)  
        raise NotImplementedError("请先完成 TODO 部分的代码！")  
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        # ==========================================  
        # TODO 2: 实现 SwiGLU 的前向传播  
        # 提示: gate 分支先过 F.silu，再和 up 分支逐元素相乘，最后过 down_proj  
        # ==========================================        
        hidden_states = self.gate_proj(x)  
        hidden_states = F.silu(hidden_states)  
        hidden_states = hidden_states * self.up_proj(x)  
        output = self.down_proj(hidden_states)  
        return output  
  
class LlamaDecoderLayer(nn.Module):  
    def __init__(self, hidden_size: int, intermediate_size: int):  
        super().__init__()  
        self.hidden_size = hidden_size  
  
        # 1. 注意力模块与它的前置 LayerNorm  
        self.input_layernorm = DummyRMSNorm(hidden_size)  
        self.self_attn = DummyAttention(hidden_size)  
  
        # 2. MLP 模块与它的前置 LayerNorm  
        self.post_attention_layernorm = DummyRMSNorm(hidden_size)  
        self.mlp = LlamaMLP(hidden_size, intermediate_size)  
  
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  
        """  
        Args:            hidden_states: [batch, seq_len, hidden_size]        Returns:            output: [batch, seq_len, hidden_size]        
        """        
        # ==========================================        
        # TODO 3: 实现 LLaMA 的 Pre-Norm 残差连接  
        # 提示: 先做 Attention residual，再做 MLP residual  
        # ==========================================        
        # --- Attention Block ---        
        residual = hidden_states  
        hidden_states = self.input_layernorm(hidden_states)  
        hidden_states = self.self_attn(hidden_states)  
        hidden_states = residual + hidden_states  
  
        # --- MLP Block ---  
        hidden_states = self.post_attention_layernorm(hidden_states)  
        hidden_states = self.mlp(hidden_states)  
        hidden_states = residual + hidden_states  
        return hidden_states
```
```
# 运行此单元格以测试你的实现  
def test_llama_block():  
    try:  
        batch_size, seq_len, hidden_size = 2, 16, 512  
        # LLaMA 通常设置 intermediate_size 为 8/3 * hidden_size，并向 multiple_of 取整  
        intermediate_size = 1376  
  
        layer = LlamaDecoderLayer(hidden_size, intermediate_size)  
        x = torch.randn(batch_size, seq_len, hidden_size)  
  
        out = layer(x)  
  
        assert out.shape == (batch_size, seq_len, hidden_size), "输出形状错误！"  
  
        # 简单验证一下计算图是否连通 (是否包含所有的参数)  
        out.sum().backward()  
        for name, param in layer.named_parameters():  
            assert param.grad is not None, f"参数 {name} 没有接收到梯度，请检查前向传播连接！"  
        print("\n✅ All Tests Passed! LLaMA-3 Transformer Block 组装完成，所有测试通过。")  
  
    except NotImplementedError:  
        print("请先完成 TODO 部分的代码！")  
        raise  
    except (AttributeError, NameError, TypeError) as e:  
        print(f"代码可能未完成: {e}")  
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e  
    except AssertionError as e:  
        print(f"❌ 测试失败: {e}")  
        raise  
    except Exception as e:  
        print(f"\n❌ 测试失败: {e}")  
        raise  
  
test_llama_block()
```
