import torch
import torch.nn as nn

# PyTorch 的交叉熵自带了 Softmax，所以模型直接输出原始得分（Logits）就行
# 假设预测 3 个词：猫、狗、鱼
# 真实标签是“狗”（索引为 1）
input_logits = torch.tensor([[3.0, 1.0, 0.0]]) # 模型觉得最可能是猫
target = torch.tensor([1]) # 但实际上是狗（索引1）

# 1. 定义交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()

# 2. 算 Loss
loss = loss_fn(input_logits, target)
print("模型猜错时的交叉熵 Loss:", loss.item())

# 3. 如果模型觉得狗的可能性最大呢？
correct_logits = torch.tensor([[0.0, 3.0, 1.0]]) 
loss_correct = loss_fn(correct_logits, target)
print("模型猜对时的交叉熵 Loss:", loss_correct.item())

# 你会发现，猜错时 Loss 巨大，猜对时 Loss 很小。这就是模型被“惩罚”的依据。