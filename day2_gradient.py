import math

y_true = 1.0 # 真实情感是1（正面）
w = 1.0
x = 0.5
b = -2.0

learning_rate = 0.1 # 学习率（也就是迈步的大小）

# 让模型循环学习 10 次
for i in range(10):
    #1. 前向计算 （算出预测概率）
    z = w * x + b
    y_hat = 1 / (1+math.exp(-z))

    # 2. 计算误差 Loss
    loss = -(y_true * math.log(y_hat) + (1 - y_true) * math.log (1 - y_hat))

    # 3. 计算梯度 （关键：这就是微积分里求导的结果）
    gradient_b  = (y_hat - y_true)

    # 4. 更新b （让b往反方向走一点点）
    b = b - learning_rate * gradient_b
    
    print(f"第 {i+1} 次学习: b 变成了 {b:.3f}, 预测概率 y_hat 变成了 {y_hat:.3f}, 误差 {loss:.3f}")