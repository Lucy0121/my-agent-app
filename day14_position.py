import numpy as np

word_vectors = {
    "狗": np.array([0.9, 0.8]),
    "咬": np.array([0.1, 0.2]),
    "人": np.array([0.8, 0.9])
}

# 用最简单的逻辑生成位置坐标：位置1加 0.1，位置2加 0.2，位置3加 0.3
def get_position(pos):
    return np.array([pos * 0.1, pos * 0.1])

def encode(sentence):
    result = []
    for i, word in enumerate(sentence):
        # 最终坐标 = 词义 + 位置
        vec = word_vectors[word] + get_position(i + 1)
        result.append(vec)
    return np.array(result)

sentence_1 = ["狗", "咬", "人"]
sentence_2 = ["人", "咬", "狗"]

matrix_1 = encode(sentence_1)
matrix_2 = encode(sentence_2)

print("【狗 咬 人】的矩阵:\n", matrix_1)
print("\n【人 咬 狗】的矩阵:\n", matrix_2)

# 你会发现，哪怕词一样，加上位置之后，矩阵完全不同了！