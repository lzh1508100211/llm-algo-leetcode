搭完 LLaMA 风格的 Decoder Layer 以后，你会发现不同开源模型的主干很像，但并不完全一样。很多差异不是重新设计整套架构，而是在参数共享、归一化缩放或初始化方式上做小改动；这些小改动会影响参数量、梯度流向和训练早期稳定性。
本节选两个典型 architecture tricks 做最小实现：Qwen / GPT 系常见的 Tie Word Embeddings，以及 Gemma 风格的 `1 + w` RMSNorm 缩放。完成后，你应该能看懂这些“看起来很小”的结构差异为什么值得单独讨论，并能在阅读不同模型代码时快速定位它们改变了哪条参数或梯度路径。
## 核心差异与机制
本节对比 Qwen 和 Gemma 在架构设计上的两项关键改动及其设计动机。
**Trick 1: Tie Word Embeddings (权重绑定) - Qwen 系列 / GPT-2**
-   **做法**：在绝大多数模型（如 LLaMA）中，最开始的 `Token Embedding` 矩阵（把 ID 变向量）和最后的 `LM Head` 矩阵（把向量变概率）是两个独立的权重矩阵。但在 Qwen 中，**这两个矩阵共享同一份物理内存的参数！**
-   **意义**：极大减少了参数量（词表动辄 15 万，非常占参数），并且在训练时能让 Embedding 获得更直接的梯度更新。
**Trick 2: RMSNorm 的 "+1 缩放" - Gemma 系列**
-   **做法**：标准的 RMSNorm 公式是 $y = \frac{x}{\mathrm{RMS}(x)} \cdot w$，其中 $\mathrm{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$。而 Google 的 Gemma 把它改成了 。
-   **意义**：在 PyTorch 中，权重的默认初始化通常是 0（或者很小的值）。Gemma 加上 1，使得在训练的极早期（缩放参数  时），RMSNorm 直接等价于一个不做任何缩放的纯归一化层，**这带来了非常平滑的梯度和非常稳定的早期训练！**
**快速对照：**

| 模型 | Embedding / LM Head | Norm | 备注 |
| --- | --- | --- | --- |
| GPT-2 | 共享 `embed_tokens` 和 `lm_head` 的权重 | 标准 LayerNorm / RMSNorm 变体 | 经典的权重绑定 baseline，便于对照 |
| LLaMA3 | 通常不绑定 | RMSNorm | 现代主流参考，结构较简洁 |
| Qwen | 共享 `embed_tokens` 和 `lm_head` 的权重 | RMSNorm | 减少参数量，输入输出监督更一致 |
| Gemma | 通常不绑定 | `1 + w` 的 RMSNorm | 初始阶段更平滑，训练更稳 |

一句话总结：Qwen 通过共享输入/输出权重压缩参数，Gemma 通过 `1 + w` 缩放提升早期训练稳定性。
## Weight Tying 与偏置项设计
Weight Tying 让 Embedding 层和输出层（LM Head）共享同一份参数；LM Head 通常不加 bias，是这类实现的常见配置。
-   **收益**：共享权重可以直接减少参数量，输入端和输出端也更容易保持表示一致。
-   **取舍**：绑定权重会牺牲一部分独立表达自由度，但通常能换来更好的参数效率。
-   **工程动机**：在大模型里，这类改动不是“省参数”这么简单，而是围绕训练稳定性、表达能力和内存布局做的取舍。
设共享权重矩阵为 $W \in \mathbb{R}^{V \times d}$，其中 $V$ 为词表大小， $d$为隐藏维度。Embedding 查表和 LM Head 投影都围绕同一份词表参数展开。
```
input_ids [B, T]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Token Embedding                                               │
│  [B, T] → [B, T, D]                                           │
│  weight: [Vocab, D]                                           │
│                                                                 │
│  ╔═══════════════════════════════════════════════════════════╗ │
│  ║  ★ Trick 1: 权重绑定 (Weight Tying)                     ║ │
│  ║  位置: Embedding 和 LM Head 共享同一份权重               ║ │
│  ║  代表: Qwen, GPT-2, T5                                  ║ │
│  ║  效果: 省 50% 词表参数（约 6 亿参数）                   ║ │
│  ╚═══════════════════════════════════════════════════════════╝ │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Transformer Blocks × N                                        │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Block 1                                                │ │
│  │                                                         │ │
│  │  ╔═════════════════════════════════════════════════════╗ │ │
│  │  ║  ★ Trick 2: RMSNorm 变体（+1 缩放）               ║ │ │
│  │  ║  位置: 每个 Block 的 Pre-Norm 位置                 ║ │ │
│  │  ║  标准: RMSNorm = x / RMS × gamma                   ║ │ │
│  │  ║  Gemma: RMSNorm = x / RMS × (gamma + 1)           ║ │ │
│  │  ║  效果: 训练初期 gamma=0 时，归一化层“不做缩放”    ║ │ │
│  │  ║        梯度更平滑，训练更稳定                       ║ │ │
│  │  ╚═════════════════════════════════════════════════════╝ │ │
│  │                                                         │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  RMSNorm (Pre-Norm)                               │ │ │
│  │  │  ── 如果是 Gemma: x / RMS × (gamma + 1)          │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  │                         │                                │ │
│  │                         ▼                                │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  Multi-Head Attention (MHA / GQA)                 │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  │                         │                                │ │
│  │                         ▼                                │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  残差连接 (+)                                      │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  │                         │                                │ │
│  │                         ▼                                │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  RMSNorm (Pre-Norm)                               │ │ │
│  │  │  ── 如果是 Gemma: x / RMS × (gamma + 1)          │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  │                         │                                │ │
│  │                         ▼                                │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  SwiGLU (MLP)                                     │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  │                         │                                │ │
│  │                         ▼                                │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  残差连接 (+)                                      │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│  其他结构技巧（位置参考）                                      │
│                                                                 │
│  ★ RoPE: 在 Attention 的 Q 和 K 上做旋转                     │
│     位置: 生成 Q、K 之后，算 Q×K^T 之前                     │
│                                                                 │
│  ★ GQA: 在 Attention 的 K 和 V 投影时减少组数               │
│     位置: k_proj 和 v_proj 的输出维度 = num_kv_heads × Dh   │
│                                                                 │
│  ★ SwiGLU: 在 MLP 层                                        │
│     位置: FFN 的激活函数，替换 ReLU                         │
│                                                                 │
│  ★ Pre-Norm: 归一化在子层之前                               │
│     位置: 每个 Attention 和 MLP 前面                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Final RMSNorm                                                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  LM Head                                                       │
│  [B, T, D] → [B, T, Vocab]                                   │
│  weight: [Vocab, D]                                           │
│                                                                 │
│  ╔═══════════════════════════════════════════════════════════╗ │
│  ║  ★ Trick 1: 权重绑定 (Weight Tying)                     ║ │
│  ║  位置: LM Head 直接复用 Embedding 的权重                 ║ │
│  ║  代码: logits = x @ embedding.weight.T                  ║ │
│  ╚═══════════════════════════════════════════════════════════╝ │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
logits → Softmax → 下一个 Token
```
## 代码实现
```
  
import torch  
import torch.nn as nn  
# --- Trick 1: Gemma 风格的 RMSNorm ---  
class GemmaRMSNorm(nn.Module):  
    def __init__(self, hidden_size: int, eps: float = 1e-6):  
        super().__init__()  
        self.eps = eps  
        # weight 初始化为全 0  
        self.weight = nn.Parameter(torch.zeros(hidden_size))  
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        """  
        Gemma 风格的 RMSNorm        公式: output = x / RMS(x) * (1 + weight)        其中 weight 初始为 0，实现恒等映射        """        # 计算均方根（使用 FP32 保证数值稳定性）        x_f32 = x.float()  
        variance = x_f32.pow(2).mean(-1, keepdim=True)  
        x_norm = x_f32 * torch.rsqrt(variance + self.eps)  
  
        # ==========================================  
        # TODO 1: 实现 Gemma 的 +1 缩放  
        # 注意类型转换回 x.dtype  
        # Gemma 公式: output = normalized * (1 + weight)        # 注意: weight 初始为 0，确保类型转换回 x.dtype        # ==========================================        # output = ？？        output = x_norm * (1 + self.weight)  
        return output.type_as(x)  
  
  
  
  
# --- Trick 2: Qwen 风格的权重绑定 ---  
class QwenTieEmbeddings(nn.Module):  
    def __init__(self, vocab_size: int, hidden_size: int):  
        super().__init__()  
        # 1. 定义标准的 Embedding 层  
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)  
  
        # 2. 定义最后的 LM Head 预测层，注意不要 bias  
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)  
  
        # ==========================================  
        # TODO 2: 将 lm_head 的权重在内存级别绑定到 embed_tokens 上  
        # 提示: 使用赋值让 lm_head.weight 指向 embed_tokens.weight  
        # 验证: 可用 .data_ptr() 检查内存地址是否一致        
        # ==========================================        
        self.lm_head.weight = self.embed_tokens.weight  
  
    def forward_embed(self, input_ids):  
        return self.embed_tokens(input_ids)  
  
    def forward_lm_head(self, hidden_states):  
        return self.lm_head(hidden_states)  
# 测试你的实现  
def test_tricks():  
    try:  
        hidden_size = 64  
        vocab_size = 1000  
  
        # =====1. 测试 Gemma RMSNorm=====  
        print("\n[1/2] 测试 Gemma RMSNorm...")  
        gemma_norm = GemmaRMSNorm(hidden_size)  
        x = torch.randn(2, 10, hidden_size)  
        out = gemma_norm(x)  
  
        # 验证初始化时 (weight=0)，输出等价于无缩放的 norm  
        variance = x.float().pow(2).mean(-1, keepdim=True)  
        expected = (x.float() * torch.rsqrt(variance + gemma_norm.eps)).to(x.dtype)  
  
        # assert torch.allclose(out, expected, atol=1e-4, rtol=1e-3), "Gemma 的 1+w 缩放机制实现错误！"  
        # print("✅ Gemma RMSNorm (+1 trick) 测试通过！")        
        assert torch.allclose(out, expected, atol=1e-4, rtol=1e-3), \  
            "Gemma RMSNorm: weight=0 时输出与预期不符"  
  
        # 补充测试: weight 非零时缩放生效  
        with torch.no_grad():  
            gemma_norm.weight.data = torch.randn(hidden_size) * 0.1  
        out2 = gemma_norm(x)  
        assert not torch.allclose(out, out2, atol=1e-4), \  
            "Gemma RMSNorm: weight 非零时应产生不同输出"  
  
        # 补充测试: FP16 类型转换  
        x_fp16 = x.half()  
        out_fp16 = gemma_norm(x_fp16)  
        assert out_fp16.dtype == torch.float16, \  
            "Gemma RMSNorm: FP16 输入应保持 FP16 输出"  
  
        print("✅ Gemma RMSNorm 测试通过！")  
  
        # =====2. 测试 Qwen 权重绑定=====  
        print("\n[2/2] 测试 Qwen 权重绑定...")  
        qwen_model = QwenTieEmbeddings(vocab_size, hidden_size)  
  
        # 检查物理内存地址是否相同  
        ptr_embed = qwen_model.embed_tokens.weight.data_ptr()  
        ptr_head = qwen_model.lm_head.weight.data_ptr()  
        assert ptr_embed == ptr_head, "权重未在物理内存级别绑定！"  
  
        # 模拟训练更新一次 Embedding  
        qwen_model.embed_tokens.weight.data += 1.0  
  
        # 验证 LM Head 的权重也跟着变了 (因为它们是同一个指针)  
        assert torch.allclose(  
            qwen_model.lm_head.weight.data[0, 0],  
            qwen_model.embed_tokens.weight.data[0, 0]  
        ), "权重更新未同步！"  
  
        # 补充测试: 梯度共享  
        qwen_model.embed_tokens.weight.requires_grad_(True)  
        dummy_ids = torch.randint(0, vocab_size, (2, 5))  
        hidden = qwen_model.forward_embed(dummy_ids)  
        loss = qwen_model.forward_lm_head(hidden).sum()  
        loss.backward()  
  
        assert qwen_model.embed_tokens.weight.grad is not None, \  
            "Embedding 应有梯度"  
        assert qwen_model.lm_head.weight.grad is not None, \  
            "LM Head 应有梯度"  
  
        grad_embed_ptr = qwen_model.embed_tokens.weight.grad.data_ptr()  
        grad_head_ptr = qwen_model.lm_head.weight.grad.data_ptr()  
        assert grad_embed_ptr == grad_head_ptr, \  
            "梯度应共享同一内存"  
  
  
        print("✅ Qwen Tie Word Embeddings 权重绑定测试通过！")  
        print("\n架构变体技巧测试通过。")  
  
    except NotImplementedError:  
        print("请先完成 TODO 代码！")  
        raise  
    except AttributeError as e:  
        print(f"❌ 属性错误: {e}")  
        raise  
    except TypeError as e:  
        print(f"❌ 类型错误: {e}")  
        raise  
    except AssertionError as e:  
        print(f"❌ 测试失败: {e}")  
        raise  
    except NameError as e:  
        print("代码可能未完成，导致变量未定义")  
        raise NotImplementedError("请先完成 TODO 代码！") from e  
    except Exception as e:  
        print(f"❌ 发生未知异常: {e}")  
        raise  
  
test_tricks()
```
