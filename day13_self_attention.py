import numpy as np

word_vectors = {
    "好": np.array([0.8, 0.9]),
    "差": np.array([-0.8, -0.9])
}
sentence = ["好", "差"]
X = np.array([word_vectors[w] for w in sentence])

# ==========================================
# 核心魔法：自己算权重！
# ==========================================
# 第一步：算点积（相似度）。X @ X.T 就是让每个词跟所有词算内积
# 算出来的矩阵，代表了词与词之间的“原始打分”
scores = X @ X.T
print("原始打分矩阵:\n", scores)

# 第二步：Softmax。把打分变成总和为 1 的百分比（权重）
# 我们手写一个 Softmax 函数（这就是大模型把原始得分转为概率的通用方法）
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # 防溢出保护
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

attention_weights = softmax(scores)
print("\n模型自己算出来的注意力权重:\n", attention_weights)

# 第三步：用算出来的权重去乘原始坐标，得到融合了上下文的新向量
final_vec = attention_weights @ X
print("\n融合了上下文的新词向量:\n", final_vec)