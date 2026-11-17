import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = SimpleNLPModel()

# 【修改点 1】把 MSELoss 换成 BCELoss（交叉熵，专门用于 0~1 的概率分类）
loss_fn = nn.BCELoss()

# 【修改点 2】把学习率加大到 1.0，步子迈大一点
optimizer = optim.SGD(model.parameters(), lr=1.0)

X = torch.tensor([[0.75, 0.875], 
                  [-0.75, -0.875]])
y_true = torch.tensor([[1.0], 
                       [0.0]])

# 【修改点 3】把循环次数从 10 次变成 50 次
for epoch in range(50):
    y_hat = model(X)
    loss = loss_fn(y_hat, y_true)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"第 {epoch+1} 轮: 预测值 = {y_hat.data.flatten().numpy()}, 误差 = {loss.item():.4f}")

test_x = torch.tensor([[0.8, 0.9]])
print("\n测试句子 ['好'] 的预测概率:", model(test_x).item())