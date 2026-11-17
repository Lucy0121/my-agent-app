import math

x=[1.0,2.0,3.0]
w=[0.2,-0.1,0.5]
b=0.3

#计算 z
z = sum(wi * xi for wi, xi in zip(w,x))+b

print("z是：", z)

#计算 y_hat （概率）
y_hat = 1/(1+math.exp(-z))
print("y_hat （概率）是：", y_hat)