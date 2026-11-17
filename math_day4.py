# 假设模型公式是：z = w1*x1 + w2*x2
x1 = 2.0
x2 = 3.0

# 初始权重（瞎猜的）
w1 = 1.0
w2 = 1.0

# 真实结果
y_true = 10.0
learning_rate = 0.1

for step in range(5):
    # 1. 前向计算：算出当前得分
    z = w1 * x1 + w2 * x2
    print(f"\n第 {step+1} 步: z 得分 = {z:.2f}")
    
    # 2. 算误差 (Loss = (z - y_true)^2)
    loss = (z - y_true) ** 2
    print(f"当前误差 = {loss:.2f}")
    
    # 3. 【手推微积分】计算偏导数 (梯度)
    # 对 z 求导，结果是 2 * (z - y_true)
    d_loss_dz = 2 * (z - y_true)
    
    # 偏导数：w1 的梯度 = 对z的导数 * x1 (这就是链式法则的甩锅过程！)
    grad_w1 = d_loss_dz * x1
    grad_w2 = d_loss_dz * x2
    
    print(f"w1 的梯度 = {grad_w1:.2f}, w2 的梯度 = {grad_w2:.2f}")
    
    # 4. 更新权重（分别下山）
    w1 = w1 - learning_rate * grad_w1
    w2 = w2 - learning_rate * grad_w2

print(f"\n最终参数: w1={w1:.2f}, w2={w2:.2f}")
print(f"最终得分: {w1*x1 + w2*x2:.2f} (目标是 10.0)")