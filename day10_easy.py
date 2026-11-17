import numpy as np

# 1. 这就是个查字典的盒子
word_vectors = {
    "好": [0.8, 0.9],
    "开心": [0.7, 0.85],
    "差": [-0.8, -0.9],
    "环境": [0.1, 0.2] 
}

sentence = "好 开心"
words = sentence.split(" ") # 切分成 ["好", "开心"]

# 2. 把句子变成矩阵（大白话版）
vec_list = [] # 准备一个空列表
for w in words: # 挨个拿出词
    if w in word_vectors: # 如果词在字典里
        vec_list.append(word_vectors[w]) # 把坐标加进去

print("找到的词向量列表:", vec_list)

# 3. 按列求平均值（大白话版）
# 现在我们有两个词，也就是两行 [0.8, 0.9] 和 [0.7, 0.85]
# 我们手动把它们加起来，除以个数
num_words = len(vec_list) # 词的数量（2个）

col1_sum = vec_list[0][0] + vec_list[1][0] # 第一列相加：0.8 + 0.7
col2_sum = vec_list[0][1] + vec_list[1][1] # 第二列相加：0.9 + 0.85

avg_col1 = col1_sum / num_words # 算第一列平均
avg_col2 = col2_sum / num_words # 算第二列平均

final_vector = [avg_col1, avg_col2]
print("最终这句话的坐标:", final_vector)

# 看看跟高级写法的 np.mean 算出来的一样不一样
print("用 NumPy 算出来的:", np.mean(np.array(vec_list), axis=0))