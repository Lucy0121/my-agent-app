import torch
import torch.nn as nn

# 1. 定义参数
d_model = 4   # 词向量的维度（之前我们手写用的是 2 维，这里用 4 维）
nhead = 2     # 注意力头的数量（把注意力分成 2 份，各看各的）

# 2. 核心魔法：直接调用 PyTorch 写好的 Transformer 层
# 这一行代码，等价于你手写的注意力层 + FFN + 残差连接！
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)

# 3. 假装我们有 1 句话，3 个词，每个词 4 维
# (形状: [batch_size=1, 句子长度=3, 特征维度=4])
x = torch.randn(1, 3, d_model)
print("输入的词向量矩阵形状:", x.shape)

# 4. 把句子喂给 Transformer 层
output = encoder_layer(x)

print("\n经过 Transformer 处理后的输出形状:", output.shape)
print("输出结果:\n", output)