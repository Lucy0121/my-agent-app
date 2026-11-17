import numpy as np

# 1. 词向量
word_vectors = {
    "好": np.array([0.8, 0.9]),
    "差": np.array([-0.8, -0.9])
}

sentence = ["好", "差"]
X = np.array([word_vectors[w] for w in sentence]) # 矩阵 X：[[0.8, 0.9], [-0.8, -0.9]]
print("原始词向量矩阵 X:\n", X)

# ==========================================
# 2. 让词与词自己互打分（计算注意力分数）
# ==========================================
# X @ X.T 的意思是，用矩阵 X 乘以它的转置
# 这一步计算了每一个词，跟其他所有词的“点积”（相似度）
scores = X @ X.T
print("\n词与词之间的原始打分（矩阵）:\n", scores)
# 你会发现，对角线上的分数很高（自己跟自己比），非对角线是负数（好跟差比）

# ==========================================
# 3. 将分数转换成权重（Softmax 魔法）
# ==========================================
# 这一步是让分数变成加起来等于 1 的百分比（和我们的 Sigmoid 类似，只是用于多分类）
# 我们手写一个简单的 Softmax 逻辑
exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True)) # 防止数字太大溢出
attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
print("\n经过 Softmax 转换后的注意力权重:\n", attention_weights)
# 这里每一行的数字加起来都是 1

# ==========================================
# 4. 用算出来的权重，去加权组合向量
# ==========================================
# 用权重矩阵去乘原始的 X 矩阵
final_output = attention_weights @ X
print("\n经过注意力机制提取后的新向量:\n", final_output)