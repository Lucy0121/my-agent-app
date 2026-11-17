import numpy as np

# 假设正确答案是 1（正面情感），真实概率分布是 100% 押注在正面
y_true = np.array([1.0, 0.0])

# 模型 A 的预测（非常自信且正确）
y_hat_A = np.array([0.9, 0.1])

# 模型 B 的预测（稍微有点犹豫）
y_hat_B = np.array([0.6, 0.4])

# 模型 C 的预测（完全猜反了，极其自信地猜了负面）
y_hat_C = np.array([0.1, 0.9])

def cross_entropy(y_true, y_hat):
    # 交叉熵公式：- (真实概率 * log(预测概率)) 的总和
    # 别怕，代码里 np.log 就是算数学里的 ln
    return -np.sum(y_true * np.log(y_hat))

print("模型 A (猜得很准) 的交叉熵 Loss:", cross_entropy(y_true, y_hat_A))
print("模型 B (有点犹豫) 的交叉熵 Loss:", cross_entropy(y_true, y_hat_B))
print("模型 C (猜反了) 的交叉熵 Loss:", cross_entropy(y_true, y_hat_C))