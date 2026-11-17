import math
import numpy as np
import random

# ==========================================
# 1. 准备“预训练”好的词向量（昨天学的）
# ==========================================
word_vectors = {
    "好": np.array([0.8, 0.9]),
    "开心": np.array([0.7, 0.85]),
    "差": np.array([-0.8, -0.9]),
    "讨厌": np.array([-0.7, -0.85])
}

# 2. 准备训练数据
train_data = [
    (["好", "开心"], 1.0),  # 正面
    (["差", "讨厌"], 0.0)   # 负面
]

# ==========================================
# 3. 初始化模型参数
# ==========================================
# 注意！因为一句话经过求平均后，只有 2 个维度，所以 w 也只有 2 个！
w = [random.uniform(-0.1, 0.1) for _ in range(2)]
b = 0.0
学习率 = 0.5

print("--- 开始训练带有词向量的模型 ---")

# ==========================================
# 4. 开始学习
# ==========================================
for epoch in range(20):
    total_loss = 0
    
    for words, y_true in train_data:
        # [第一步] 查字典，找坐标，求平均（昨天的知识）
        vec_list = []
        for word in words:
            if word in word_vectors:
                vec_list.append(word_vectors[word])
        
        matrix = np.array(vec_list)
        x = np.mean(matrix, axis=0) # 这就是这句话的最终特征向量！比如 [0.75, 0.875]
        
        # [第二步] 前向传播算得分和概率（前面的知识）
        # 这里的 x[0] 和 x[1] 就是那两个坐标数字
        z = w[0] * x[0] + w[1] * x[1] + b
        z = max(min(z, 100), -100)
        y_hat = 1 / (1 + math.exp(-z))
        
        # [第三步] 算偏差
        偏差 = y_hat - y_true
        total_loss += 偏差 ** 2
        
        # [第四步] 反向传播，纠错更新！
        # 注意这里：更新 w[0] 的时候，用 x[0]；更新 w[1] 的时候，用 x[1]
        for i in range(2):
            w[i] = w[i] - 学习率 * 偏差 * x[i]
        b = b - 学习率 * 偏差
        
    if (epoch + 1) % 5 == 0:
        print(f"第 {epoch+1} 轮学习: 总误差 = {total_loss:.4f}")

# ==========================================
# 5. 毕业测试（预测新句子）
# ==========================================
print("\n--- 训练结束，开始测试 ---")
test_sentence = ["好", "差"]
vec_list = [word_vectors[word] for word in test_sentence]
test_x = np.mean(np.array(vec_list), axis=0)

z = w[0] * test_x[0] + w[1] * test_x[1] + b
y_hat = 1 / (1 + math.exp(-z))

print(f"测试句子: {test_sentence}")
print(f"这句话的向量坐标: {test_x}")
print(f"模型预测它是正面情感的概率: {y_hat:.4f}")

if y_hat > 0.5:
    print("最终判决：正面情感 😄")
else:
    print("最终判决：负面情感 😡")