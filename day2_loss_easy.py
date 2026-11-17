# 真实标签是 1 （这句话是正面情感）
y_true = 1

# 假设我们模型计算出来的概率是 0.858
y_hat = 0.858

# 计算误差（就是用最简单的：预测值减去真实值，然后平方）
loss = (y_hat - y_true)**2
print("当前预测 0.858，误差是：", loss)

# 现在把 y_hat 改成0.1 试试
y_hat_bad = 0.1
loss_bad = (y_hat_bad - y_true) **2
print("当前预测 0.1， 误差是：", loss_bad)