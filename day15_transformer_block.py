import numpy as np

# 1. 基础零件（词向量和位置编码）
word_vectors = {
    "狗": np.array([0.9, 0.8]),
    "咬": np.array([0.1, 0.2]),
    "人": np.array([0.8, 0.9])
}

def get_position(pos):
    return np.array([pos * 0.1, pos * 0.1])

# 2. Softmax 函数（你刚学过的）
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    print("x=",x - np.max(x, axis=1, keepdims=True))
    print("exp_x=",exp_x)
    # print("exp=",exp_x / np.sum(exp_x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 3. 制造输入矩阵 X（词义 + 位置）
sentence = ["狗", "咬", "人"]
X = np.array([word_vectors[w] + get_position(i+1) for i, w in enumerate(sentence)])
print("【输入矩阵 X】:\n", X)

# ==========================================
# 核心模块：Transformer Block
# ==========================================
# 第一步：Self-Attention（注意力层）
print("\n--- 进入 Attention 层 ---")
scores = X @ X.T
# print("X.T=",X.T)
print("scores=",scores)
attention_weights = softmax(scores)
print("attention_weights=",attention_weights)
attention_output = attention_weights @ X  # 根据权重，融合上下文信息
print("注意力输出:\n", attention_output)

# 第二步：残差连接（保留原本的记忆！）
# 把原始矩阵 X 和 注意力算出来的结果 相加
residual_1 = X + attention_output
print("\n【残差连接后】:\n", residual_1)

# 第三步：Feed-Forward Network (FFN 前馈网络，其实就是一层全连接)
# 假设我们有一个随机的权重矩阵 W_ffn 和 偏置 b_ffn
# 真实的大模型，这里会把 2 维变成 4096 维，再变回 2 维。这里我们保持 2 维。
W_ffn = np.array([[1.5, 0.2], [0.1, 1.5]]) 
ffn_output = residual_1 @ W_ffn # 矩阵乘法！
print("\n【FFN 层输出】:\n", ffn_output)

# 第四步：再搞一次残差连接
final_output = residual_1 + ffn_output
print("\n【最终 Transformer 输出】:\n", final_output)