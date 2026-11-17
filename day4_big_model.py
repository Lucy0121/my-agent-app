import random
import math

# 1. 假装我们有 100 个特征（比如100个词）
# 我们用列表推导式生成 100 个 1
x = [0.01] * 100 

# 2. 人类“初始化”模型：瞎猜 100 个参数 w
# random.random() 会生成 0 到 1 之间的随机数
w = [random.uniform(-0.1,0.1) for _ in range(100)]
b = 0.0 
学习率 = 0.5
y_true = 1.0


# 3. 机器开始学习 5 次
for epoch in range(5):
    # 第一步：把 100 个乘积加起来，再加上 b (这就是矩阵的点积)
    # 等价于 z = w1*x1 + w2*x2 + ... + w100*x100 + b
    z = sum(wi * xi for wi, xi in zip(w, x)) + b
    
    # 第二步：算预测概率 (Sigmoid)
    # 限制一下 z 的大小，防止数字太大报错
    z = max(min(z, 100), -100) 
    y_hat = 1 / (1 + math.exp(-z))
    
    # 第三步：算偏差
    偏差 = y_hat - y_true
    
    # 第四步：统一更新 100 个参数！
    # 你看，这里不用写 100 行代码，一个循环就搞定了
    # 这就是为什么公式一样：我们对每一个 w 都做一样的处理
    for i in range(100):
        w[i] = w[i] - 学习率 * 偏差 * x[i]
        
    b = b - 学习率 * 偏差
    
    print(f"第{epoch+1}次学习: 预测={y_hat:.4f}, 偏差={偏差:.4f}, 第1个w={w[0]:.4f}, 第100个w={w[99]:.4f}")