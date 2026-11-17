import torch

# 1. 创建张量（Tensor），和 NumPy 几乎一模一样
x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([0.2, -0.1, 0.5], requires_grad=True) # 只要加上这个，它就会自动求导！

# 2. 算点积
z = w @ x
print("z 的得分:", z)

# 3. 算 Sigmoid 概率
y_hat = torch.sigmoid(z)
print("y_hat 的概率:", y_hat)

# 4. 定义真实标签和误差 Loss
y_true = torch.tensor(1.0)
loss = (y_hat - y_true) ** 2
print("误差 Loss:", loss)

# 5. 魔法时刻：自动算梯度（以前我们要手写微积分公式，现在只要一行！）
loss.backward()
print("PyTorch 自动算出来的 w 的梯度是:", w.grad)