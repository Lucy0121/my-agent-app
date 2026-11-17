import numpy as np

# 1. 假装我们训练好了一个超级简化的词向量表（只有3个词，每个词用2个数字表示）
# 在真实的 ChatGPT 里，每个词是用 4096 个数字表示的！
word_vectors = {
    "好":np.array([0.8, 0.9]),
    "开心":np.array([0.7,0.85]),
    "差":np.array([-0.8, -0.9])
}

# 2. 怎么算两个词有多像？
# 数学里叫“余弦相似度”，其实你现在只要把它理解为“算这两个箭头是不是指向同一个方向”
# 公式：两个向量点积 / (长度乘积)。（不用背，直接用 NumPy）
def similarity(word1, word2):
    vec1 = word_vectors[word1]
    vec2 = word_vectors[word2]

    score = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return score

# 3. 我们来测试一下
print("“好” 和 “开心” 的相似度:", similarity("好", "开心"))
print("“好” 和 “差” 的相似度:", similarity("好", "差"))