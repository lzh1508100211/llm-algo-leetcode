## 预训练和微调的区别

| 对比维度 | 预训练（Pre-training） | SFT（监督微调） |
| :--- | :--- | :--- |
| **学习方式** | 无监督学习 | 有监督学习 |
| **训练目标** | 学习语言规律 + 世界知识 | 学习问答格式 + 人类偏好 |
| **有无标准答案** | ❌ 无（预测下一个词） | ✅ 有（人工标注的 response） |
| **数据格式** | 纯文本 | `(prompt, response)` 问答对 |
| **数据量** | TB 级（海量） | GB 级（十万~百万条） |
| **人工标注** | ❌ 不需要 | ✅ 需要 |
| **input_ids** | `tokenizer(raw_text)` | `tokenizer(prompt + response)` |
| **labels** | `labels = input_ids`（所有 token 都学） | `labels = [-100] * len(prompt) + response_ids`（只学 response） |
| **Loss 计算** | 对所有 token 计算 | **只对 response 部分计算**（prompt 被 -100 忽略） |
| **模型初始化** | 随机初始化（从头训练） | **加载预训练权重**（继续训练） |
| **计算量** | 极大（数万 GPU 天） | 较小（数十 GPU 天） |
| **典型输出** | Base Model（基础模型） | Chat Model（对话模型） |
| **代表模型** | LLaMA-base, GPT-3-base | LLaMA-Chat, Qwen-Chat |

### 一句话总结

> **预训练 = 海量阅读，没有老师（无监督）**
> **SFT = 对答案刷题，有老师教（有监督）**
>
> **代码核心差异：SFT 比预训练多了一句 `labels = [-100] * len(prompt) + response_ids`**
> 

> 真实微调里通常还会多一层 chat template：先把多轮 `messages` 渲染成模型约定的 prompt/response 文本，再 tokenizer 成 token id。无论模板长什么样，最后进入训练循环时都要落成三件套：
> -   `input_ids`：prompt、response、可选 EOS 和 padding 后的完整 token 序列。
> -   `attention_mask`：真实 token 为 `1`，padding 为 `0`，告诉模型哪些位置是有效上下文。
> -   `labels`：prompt 和 padding 为 `-100`，response / EOS 保留原 token id，告诉 loss 哪些位置要学习
## 关键环节
### 🎯 一句话先抓住本质
> **Attention Mask = “哪些位置能互相看到”（控制注意力范围）**  
> **Loss Mask = “哪些位置参与 Loss 计算”（-100 忽略）**  
> **Shift Logits = “把输入右移一位，让模型预测下一个词”**
### Attention Mask（注意力遮罩）
#### 是什么？
告诉模型 **“哪些 token 是有效的，哪些是 padding”**。
#### 为什么需要？
因为一个 batch 里的句子长度不一样，短的句子需要补 `[PAD]` 到相同长度。模型**不应该关注这些 padding 位置**。
#### 长什么样？
```
原始句子："今天天气很好"  → token: [今, 天, 天, 气, 很, 好]
batch 里另一句："你好"  → token: [你, 好, PAD, PAD, PAD, PAD]

Attention Mask:
[1, 1, 1, 1, 1, 1]   ← 第一句全是有效 token
[1, 1, 0, 0, 0, 0]   ← 第二句只有前两个是有效的
```
#### 在代码里
```
# Hugging Face 自动处理
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,  # 传入后模型会自动忽略 padding
    labels=labels
)
```
#### 因果关系（Causal Mask）
在自回归模型（如 GPT）里，还需要 **因果遮罩**，确保每个 token 只能看到**前面的 token**，不能看到未来的 token。
```
因果 Mask（下三角矩阵）：
[1, 0, 0, 0]   ← token0 只能看自己
[1, 1, 0, 0]   ← token1 只能看 token0 和 token1
[1, 1, 1, 0]   ← token2 能看到前三个
[1, 1, 1, 1]   ← token3 能看到所有
```
**代码中自动生成：**
```
causal_mask = torch.tril(torch.ones(T, T))  # 下三角矩阵
```
### Loss Mask（损失遮罩）
#### 是什么？
告诉损失函数 **“哪些位置要计算 Loss，哪些位置忽略”**。
#### 为什么需要？
在 SFT 训练中，我们**只希望模型学习 response 部分**，prompt 部分不应该参与 Loss 计算。
#### 长什么样？
```
完整序列：[prompt, response]
prompt = "今天天气怎么样？"  → 8 个 token
response = "今天天气晴"      → 5 个 token

labels = [-100, -100, ..., -100, 今, 天, 天, 气, 晴]
         ↑____prompt 全部忽略____↑  ↑__response 保留__↑
```
#### 在代码里
```
def build_sft_sample(prompt_ids, response_ids):
    labels = [-100] * len(prompt_ids) + response_ids  # ← Loss Mask
    return labels
```
#### 为什么是 `-100`？
`CrossEntropyLoss` 默认 `ignore_index = -100`，遇到 `-100` 时自动跳过该位置。
```
loss = F.cross_entropy(logits, labels, ignore_index=-100)
# logits 和 labels 中，-100 的位置会被自动忽略
```
### Shift Logits（移位逻辑）
#### 是什么？
在因果语言模型中，模型的任务是 **“预测下一个词”**。
-   **输入：** `[A, B, C, D]`
-   **输出预测：** `[B, C, D, E]`
所以需要把输入**右移一位**，让每个位置的输出对应“下一个位置”的标签。
#### 为什么要 Shift？
```
输入 x = [A, B, C, D]
模型输出 logits = [P(B), P(C), P(D), P(E)]
             ↑      ↑      ↑      ↑
           预测B    预测C   预测D   预测E

标签 labels = [A, B, C, D]
             ↓
实际想要 = [B, C, D, E]  ← 需要右移一位！
```
#### 还是没理解为什么移位！！！！
##### 用代码看 Shift
##### 模型输出
```
logits = model(input_ids)  # [B, T, vocab]
# logits 的每个位置都对应一个词表概率分布

# logits[0] 表示：根据第一个词，预测下一个词的概率
# logits[1] 表示：根据前两个词，预测下一个词的概率
# ...
```
##### 正确答案
```
labels = input_ids  # [今, 天, 天, 气, 很, 好]
```
**问题来了：**
-   `logits[0]` 对应“根据‘今’预测‘天’”
-   `labels[0]` 却是“今”本身 → **错位了！**
##### 解决方案：Shift
```
# 去掉 logits 的最后一个位置（没有下一个词可以预测了）
shift_logits = logits[..., :-1, :]  # [今, 天, 天, 气, 很] 的预测

# 去掉 labels 的第一个位置（不用预测“今”，它只是开头）
shift_labels = labels[..., 1:]       # [天, 天, 气, 很, 好]

# 现在对齐了：
shift_logits[0] → 预测 [天]      ← 对应 shift_labels[0] = 天 ✅
shift_logits[1] → 预测 [天]      ← 对应 shift_labels[1] = 天 ✅
shift_logits[2] → 预测 [气]      ← 对应 shift_labels[2] = 气 ✅
shift_logits[3] → 预测 [很]      ← 对应 shift_labels[3] = 很 ✅
shift_logits[4] → 预测 [好]      ← 对应 shift_labels[4] = 好 ✅
```
#### 原始 Transformer 写法
```
# 手动 shift
logits = model(input_ids)  # [B, T, vocab]
shift_logits = logits[..., :-1, :]  # 去掉最后一个位置
shift_labels = labels[..., 1:]      # 去掉第一个位置

loss = cross_entropy(shift_logits, shift_labels)
```
#### Hugging Face 自动处理
Hugging Face 的模型**内部自动做了 shift**：
```
outputs = model(input_ids, labels=labels)
loss = outputs.loss  # 模型内部已经 shift 好了
```
## 审计清单
| 检查项 | 你要确认什么 | 常见问题 | 后果（脏数据毒副作用） | 解决方案 |
| :--- | :--- | :--- | :--- | :--- |
| **Prompt 是否可见** | prompt 只负责提供上下文，**不参与 loss** | prompt 被误当成监督目标（labels 没设为 -100） | 模型会学着“模仿用户提问”，而不是“回答用户问题” | 确保 `labels = [-100] * len(prompt_ids) + response_ids` |
| **Response 是否存在** | 至少有一段可学习的回答 | 空 response、模板残缺（如只有 `<eos>`） | 该样本不产生任何有效梯度，浪费算力，模型学到“沉默” | 过滤掉 `len(response_ids) == 0` 的样本 |
| **EOS 是否保留** | 让模型学会**结束回答** | 只学会续写，不学会停止（EOS 被截断或遗忘） | 推理时模型永远不输出 `<eos>`，一直生成到 `max_new_tokens` 上限 | 在 response 末尾强制加入 `<eos>`，且确保其参与 loss |
| **截断后是否仍有监督 token** | `max_len` 不能把 response 全截没 | 截断后 `labels` 全是 `-100` | 该 batch 的 loss 为 0，模型完全没学到东西 | 截断时优先保留 response 尾部；或设置 `max_len` 至少大于 `len(prompt) + 2` |
| **Padding 是否只在尾部** | padding 只能补在**真实 token 之后** | 中间 padding 破坏 causal 结构（如左侧填充） | 模型在生成时“看到”了不该看到的位置，破坏自回归假设 | 统一使用**右侧 padding**，`attention_mask` 必须正确标记 |

工程上最常见的坏例子有三类：
-   **格式坏样本**：chat template 没渲染完整，`prompt / response` 边界不清楚。
-   **监督坏样本**：`labels` 不是 `-100` 就是错位 token，loss 看似下降但学不到回答。
-   **长度坏样本**：`max_len` 太短把 response 截没了，样本等于无监督数据。
真实目标不是“把 token 拼起来”，而是先确认：**哪些 token 是上下文，哪些 token 是监督，哪些样本应该直接过滤。**
## 实战代码
```

```
