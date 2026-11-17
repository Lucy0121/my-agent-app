import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead):
        super().__init__()
        # 1. 词嵌入层（Embedding）：把输入的词 ID 变成高维向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 2. Transformer 层（大脑）
        self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        # 3. 输出层：把 4 维向量压缩成 1 个得分（情感预测）
        self.fc = nn.Linear(d_model, 1)
        # 4. Sigmoid 激活
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 第一步：词变向量
        x = self.embedding(x) 
        # 第二步：交给 Transformer 思考
        x = self.transformer(x) 
        # 第三步：求平均（把 3 个词的输出压缩成 1 个）
        x = x.mean(dim=1) 
        # 第四步：分类打分
        x = self.fc(x)
        return self.sigmoid(x)

# --- 测试运行 ---
# 假设词表有 100 个词，每个词用 4 维向量表示，2 个注意力头
model = TinyGPT(vocab_size=100, d_model=4, nhead=2)

# 假装输入一句话，包含 3 个词（用词表中的索引代表，比如 [5, 23, 8]）
input_ids = torch.tensor([[5, 23, 8]])

# 预测！
prediction = model(input_ids)
print("这句话是正面情感的预测概率:", prediction.item())