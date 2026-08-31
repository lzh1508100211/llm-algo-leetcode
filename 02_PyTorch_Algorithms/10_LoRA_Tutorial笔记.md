## 生活例子：在打印稿上修改

你有一份打印好的**长文档**（原始模型），现在要修改它：
| 方法 | 比喻 | 成本 |
| -- | --| -- |
| **全量微调** | 重新打印整份文档，修改所有内容 | 高（耗纸、耗墨）|
| **LoRA** | 在文档上贴**便利贴**，只修改便利贴上的内容 | 低（只改几处）|
**LoRA = 在不重写整本书的前提下，用便利贴修改内容。**
## 数学原理
### 全量微调（Full Fine-tuning）
假设一个线性层：
$h=W_0·x$
-   $W_0$​：原始预训练权重（冻结，不更新）
-   全量微调需要更新整个 $W_0$
### LoRA 的核心思想
将权重更新拆解为**低秩分解**：
$h=(W_0 + \Delta(W))·x$

其中  $\Delta(W)$  被分解为两个小矩阵：
$\Delta(W)=B \cdot A$
$h=W_0 \cdot x+B \cdot A \cdot x$
| 矩阵 | 形状 | 参数量 | 是否可训练 | 备注 |
| -- | -- | -- | --| -- |
| $W_0$ | [D,D] | $D^2$ | ❌ 冻结 |  |
| A | [r,D] | $r \times D$ | ✅ 可训练 | A 通常用 Kaiming 均匀分布或高斯分布初始化 |
| B | [D, r] | $D \times r$ | ✅ 可训练 | B 严格初始化为零，以保证训练开始时 $\Delta W = BA \approx 0$，模型输出基本等于冻结基座的输出 |
| r | 秩（rank） | 超参数，通常 4/8/16 | - | - |

**参数量对比：**
| 方法 | 参数量 | 显存占用 |
| -- | -- | -- |
| **全量微调** | $D^2$ | 高 | 
| **LoRA** | $2 \times r \times D$ | **低（只有全量的 2r/D2r/D）** |
当 D=4096,r=8D=4096,r=8 时，LoRA 参数量只有全量的 16/4096≈0.39%16/4096≈0.39%！
## 调整位置
在工程实践中，**LoRA 应该加在 Transformer 的哪些层上**，没有一个绝对的标准答案，但它有一条非常清晰的**优先级路径**：从最核心的注意力层开始，逐步扩展到全连接层，最终根据显存和效果做取舍。
这个选择过程，本质上是在“微调效果”和“训练成本”之间寻找平衡。以下是工程化上最主流的做法和考量：
### 主流默认方案：覆盖所有关键线性层
对于绝大多数工程化场景，尤其是你刚开始微调一个模型时，最稳妥、最推荐的方案，就是把 LoRA 适配器加在**Transformer 块内几乎所有的关键线性投影层**上。
这具体指的是 ：
1.  **注意力机制（Attention）中的四个投影层**：
    -   `q_proj`：查询（Query）投影
    -   `k_proj`：键（Key）投影
    -   `v_proj`：值（Value）投影
    -   `o_proj`：注意力输出投影
2.  **前馈网络（MLP/FFN）中的三个线性层**：
    -   `gate_proj`：门控投影（在 SwiGLU 结构中）
    -   `up_proj`：升维投影
    -   `down_proj`：降维投影
这样做的理由是，经验研究和实验证明，**同时微调注意力和 MLP 层，性能最接近全量微调** [](https://unsloth.ai/docs/jp/meru/fine-tuning-llms-guide/lora-hyperparameters-guide#avoiding-overfitting-and-underfitting)。这个方案也成为了 Hugging Face PEFT 库和 Unsloth 等工具的官方推荐实践。
### 进阶策略：按需选择与针对性实验
当你希望进一步优化训练效率或针对特定任务调优时，就可以开始有策略地选择目标层了。
-   **仅微调注意力层（Attention-Only）**：如果你**显存非常紧张**，或者只想进行**轻量级的风格适配**，可以只针对 `q_proj` 和 `v_proj`（有时也包含 `k_proj`）进行微调。这能在资源有限的环境中快速达成目的。
-   **`o_proj` 层的特殊性**：有研究发现，`o_proj` 层（负责混合多头注意力输出的投影层）是一个“性价比”极高的单一目标。单独对它进行微调，能在几乎不增加延迟的情况下，获得不错的性能提升 。
-   **根据任务类型选择**：
    -   对于**分类、轻量生成**等任务，微调 `q_proj` 和 `v_proj` 可能就足够了。
    -   对于**机器翻译、摘要**等更复杂的生成任务，将 LoRA 扩展到 `o_proj` 甚至部分 MLP 层（如 `gate_proj`, `up_proj`），效果会更好。
### 工程决策思路
你可以用下面的决策流程来指导你的选择：
1.  **起点**：如果你**对模型架构不熟悉**或**想快速获得一个好结果**，直接使用 `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` 这个组合作为起点。
2.  **优化**：如果训练完后发现**显存占用过高**，可以尝试移除 MLP 层（如 `gate_proj`, `up_proj`, `down_proj`），只保留注意力层的四个投影。
3.  **精调**：如果只微调注意力层效果不达预期，可以**逐步加回 MLP 层**，观察验证集上的性能变化。
4.  **特殊考量**：如果任务**非常依赖长文本或复杂推理**，保留 MLP 层的 LoRA 通常更为关键，因为它负责存储和处理大量的知识。
> **注意**：不同模型（如 LLaMA、Qwen、Mistral）的层命名可能略有差异，在配置 `target_modules` 时，务必查阅对应模型的文档，确保名称准确无误。
## 代码实现
```
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import math  
def count_trainable_parameters(module: nn.Module) -> int:  
    return sum(p.numel() for p in module.parameters() if p.requires_grad)  
  
  
class LoRALinear(nn.Module):  
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):  
        super().__init__()  
        self.r = r  
        self.lora_alpha = lora_alpha  
        self.scaling = self.lora_alpha / self.r  
        self.lora_dropout = nn.Dropout(lora_dropout)  
  
        # ==========================================  
        # 主权重冻结，只让低秩旁路参与训练。        
        # TODO 1: 初始化主权重和 LoRA 矩阵  
        # ==========================================  
        self.linear = nn.Linear(in_features, out_features, bias=False)  
        self.linear.weight.requires_grad = False  
        self.lora_A = nn.Parameter(torch.empty(r, in_features))  
        self.lora_B = nn.Parameter(torch.empty(out_features, r))  
        self.reset_parameters()  
  
    def reset_parameters(self):  
        # ==========================================  
        # 主权重和 LoRA 旁路分别按各自规则初始化。        
        # TODO 2: 初始化权重  
        # ==========================================  
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))  
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  
        nn.init.zeros_(self.lora_B)  
  
        pass  
  
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        # ==========================================  
        # 先走主分支，再叠加低秩旁路的增量。        
        # TODO 3: 实现前向传播  
        # 1. 计算主权重的输出  
        # 2. 对 LoRA 分支输入应用 dropout        
        # 3. 计算 LoRA 分支的输出（先降维再升维，最后乘以缩放因子）        
        # 4. 将两者相加        
        # ==========================================        
        result = self.linear(x)  
        dropped = self.lora_dropout(x)  
        lora_out = (dropped @ self.lora_A.T) @ self.lora_B.T * self.scaling  
        result += lora_out  
        return result  
  
    def merge_weights(self):  
        # ==========================================  
        # TODO 4: 合并权重（零延迟推理）  
        # 提示: 将 LoRA 的低秩更新合并到主权重中  
        # ==========================================
		self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling  
        pass  
    # 运行此单元格以测试你的实现  
def test_lora():  
    try:  
        in_dim, out_dim = 128, 256  
        batch_size, seq_len = 32, 10  
        layer = LoRALinear(in_dim, out_dim, r=8, lora_alpha=16, lora_dropout=0.0)  
  
        x = torch.randn(batch_size, seq_len, in_dim)  
  
        # 1. 验证初始化导致 B 全零，所以初始输出等于冻结权重的输出  
        with torch.no_grad():  
            out_lora = layer(x)  
            out_base = layer.linear(x)  
            assert torch.allclose(out_lora, out_base), "初始化错误: lora_B 未被初始化为 0"  
  
        # 2. 验证只训练 LoRA 参数  
        expected_trainable = 8 * (in_dim + out_dim)  
        assert not layer.linear.weight.requires_grad, "主权重应该被冻结"  
        assert count_trainable_parameters(layer) == expected_trainable, "LoRA 可训练参数量统计错误"  
  
        # 3. 模拟训练一步，改变 B 的值  
        layer.lora_B.data.normal_(0, 0.02)  
  
        out_trained = layer(x)  
        assert not torch.allclose(out_trained, out_base), "前向传播错误: 旁路未能注入梯度值"  
  
        # 4. 验证合并权重的正确性  
        layer.eval()  
        out_trained = layer(x)  
        layer.merge_weights()  
        out_merged = layer.linear(x)  
        assert torch.allclose(out_trained, out_merged, atol=1e-5), "权重合并错误: 合并后的输出与分离时的输出不一致！"  
  
        print("\n✅ All Tests Passed! LoRA 核心算子、参数统计和 merge 逻辑实现正确。")  
  
    except NotImplementedError:  
        print("请先完成 TODO 部分的代码！")  
        raise  
    except (AttributeError, NameError, TypeError, ValueError) as e:  
        print("代码可能未完成，导致变量未定义" if isinstance(e, NameError) else "代码可能未完成，导致了类型错误")  
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e  
    except AssertionError as e:  
        print(f"❌ 测试失败: {e}")  
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e  
    except Exception as e:  
        print(f"❌ 发生异常: {e}")  
        raise  
  
test_lora()
```
