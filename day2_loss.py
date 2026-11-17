import math

# 假设这句话真实是负面情感（用 0 表示)
y_true = 0

# 我们刚才算出来的预测概率
y_hat = 0.858 

# 计算误差（Loss）
loss = -(y_true*math.log(y_hat)+(1-y_true)*math.log(1-y_hat))

print("当前误差：", loss)