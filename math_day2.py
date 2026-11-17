import numpy as np

# 1. 定义三个向量
v1 = np.array([1, 1])       # “好”
v2 = np.array([10, 10])     # “好极了”（方向一样，但很长）
v3 = np.array([1, -1])      # “差”（方向相反）

print("--- 点积（会被长度欺骗）---")
print("v1 和 v2 的点积:", np.dot(v1, v2)) # 1*10 + 1*10 = 20
print("v1 和 v3 的点积:", np.dot(v1, v3)) # 1*1 + 1*(-1) = 0

# 我们手写一个余弦相似度函数
def cosine_similarity(a, b):
    # 1. 算点积
    dot_product = np.dot(a, b)
    # 2. 算范数（长度）。np.linalg.norm 就是算长度的
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # 3. 套公式
    return dot_product / (norm_a * norm_b)

print("\n--- 余弦相似度（只看方向，剥离长度）---")
print("v1 和 v2 的相似度:", cosine_similarity(v1, v2)) # 应该是 1.0 (完全一样)
print("v1 和 v3 的相似度:", cosine_similarity(v1, v3)) # 应该是 0.0 (完全垂直)