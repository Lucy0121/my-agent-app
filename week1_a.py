import math

x = [1.0, 2.0, 3.0]
w = [0.2, -0.1, 0.5]
b = 0.3

z = sum(wi * xi for wi, xi in zip(w, x)) + bx
y_hat = 1 / (1 + math.exp(-z))

print("z =", z)
print("y_hat =", y_hat)