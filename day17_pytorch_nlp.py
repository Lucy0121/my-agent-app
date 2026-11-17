import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. 定模型（造个空壳）
# ==========================================
# 真实的大模型继承自 nn.Module，必须写 __init__ 和 forward

class SimpleNLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(2, 1) 的意思：输入维度是 2（因为词向量是2维），输出维度是 1（输出一个情感得分）
        # 这一行代码，等价于我们之前手写的 w1, w2 和 b！它自动帮我们初始化了参数。
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        # 前向传播 把输入 x 扔进 linear 层，然后用 sigmoid 激活
        return torch.sigmoid(self.linear(x))

model = SimpleNLPModel()

# ==========================================
# 2. 定损失（准备教鞭）
# ==========================================
# 我们之前手写的是 (y_hat - y_true)**2，PyTorch 已经封装好了，叫 MSELoss
loss_fn = nn.MSELoss()

# ==========================================
# 3. 定优化器（准备方向盘）
# ==========================================
# SGD（随机梯度下降），学习率设为 0.1。
# model.parameters() 会自动把刚才 nn.Linear 里面的 w 和 b 都抓过来。
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ==========================================
# 4. 准备数据
# ==========================================
# 假设词向量矩阵：第一句是“好开心” [0.75, 0.875]，第二句是“差讨厌” [-0.75, -0.875]
X = torch.tensor([[0.75, 0.875], 
                  [-0.75, -0.875]])

#真实标签：第一句是正面（1），第二句是负面（0）
y_true = torch.tensor([[1.0], 
                       [0.0]])

# ==========================================
# 5. 开始训练（做题 -> 改错）
# ==========================================
for epoch in range(10):
    # 第一步：前向传播（做题）
    y_hat = model(X)
    
    # 第二步：算误差
    loss = loss_fn(y_hat, y_true)
    
    # 第三步：清空之前的梯度（非常重要！如果不清空，梯度会累加）
    optimizer.zero_grad()
    
    # 第四步：反向传播（自动算梯度）
    loss.backward()
    
    # 第五步：更新参数（根据梯度修改 w 和 b）
    optimizer.step()
    
    if (epoch + 1) % 2 == 0:
        print(f"第 {epoch+1} 轮: 预测值 = {y_hat.data.flatten().numpy()}, 误差 = {loss.item():.4f}")

# ==========================================
# 训练结束后，测试一下新数据
# ==========================================
# 测试一句“好”，向量假设是 [0.8, 0.9]
test_x = torch.tensor([[0.8, 0.9]])
print("\n测试句子 ['好'] 的预测概率:", model(test_x).item())