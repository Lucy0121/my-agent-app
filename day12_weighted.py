import numpy as np

# 1. 词向量
word_vectors = {
    "好": np.array([0.8, 0.9]),
    "差": np.array([-0.8, -0.9])
}

sentence = ["好", "差"]

# 2. 查表拿坐标
matrix = np.array([word_vectors[w] for w in sentence])

# 3. 传统求平均（一视同仁）
mean_vec = np.mean(matrix, axis=0)
print("传统求平均的结果:", mean_vec) # 结果是 [0. 0.]

# 4. 加权平均（我们认为“差”更重要，给它分配 0.8 的权重，“好”只有 0.2）
weights = np.array([0.2, 0.8])
weighted_vec = np.average(matrix, axis=0, weights=weights)
print("加权平均后的结果:", weighted_vec) # 结果就变成了负数！