import numpy as np

# 1. 词向量还是老样子
word_vectors = {
    "好": np.array([0.8, 0.9]),
    "差": np.array([-0.8, -0.9])
}

# 2. 注意力权重（这是我们人为设定的，真实大模型会自己学出来）
# 假设模型发现“差”这个词对情感的决定权更大
attention_weights = {
    "好": 0.9,
    "差": 0.1
}

sentence = ["好", "差"]
words = sentence

# 3. 查表拿坐标
vec_list = []
weight_list = []
for word in words:
    if word in word_vectors:
        vec_list.append(word_vectors[word])
        weight_list.append(attention_weights[word])

matrix = np.array(vec_list) # 变成二维表格
weights = np.array(weight_list) # 变成重要性权重列表

print("原始词向量矩阵:\n", matrix)
print("注意力权重:", weights)

# ==========================================
# 4. 核心魔法：加权平均！
# ==========================================
# 传统求平均：np.mean(matrix, axis=0)
# 加权求平均：把每个词的向量乘上它的权重，加起来，再除以总权重
# np.average 可以直接做这件事，并传入 weights 参数
final_vec = np.average(matrix, axis=0, weights=weights)

print("\n传统求平均的结果:", np.mean(matrix, axis=0))
print("加入注意力权重后的结果:", final_vec)