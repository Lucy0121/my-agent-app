import random
import math

# 1. 假装我们有一个包含 5 个词的极小词表
# 词典里只有：["今天", "天气", "真", "好", "散步"]
vocab = ["今天", "天气", "真", "好", "散步"]

# 2. 真实的句子
sentence = ["今天", "天气", "真", "[遮住]", "散步"]

# 3. 假设模型要给 5 个词打分（权重 w），挑一个概率最高的填空
w = [random.uniform(-0.1, 0.1) for _ in range(5)] 
学习率 = 0.1

# 4. 真实的答案是 “好”，它在词表里的索引是 3
y_true_index = 3 

print("开始模拟 ChatGPT 的“完形填空”学习过程...\n")

for epoch in range(5):
    # 第一步：模型给词表里所有词打分 (为了简化，我们省略复杂的输入，直接看它的参数)
    z = w.copy()
    
    # 第二步：把分数变成概率 (Softmax 算法，保证 5 个词的概率加起来等于 1)
    # 这其实就是 Sigmoid 的多分类版本，也是大模型的输出层
    exp_z = [math.exp(val) for val in z]
    sum_exp = sum(exp_z)
    probabilities = [val / sum_exp for val in exp_z]
    
    # 第三步：看看模型当前认为“好”字的概率是多少
    y_hat = probabilities[y_true_index]
    
    # 第四步：算偏差（模型猜“好”的概率 - 真实情况 1.0）
    偏差 = y_hat - 1.0
    
    # 打印结果
    print(f"第 {epoch+1} 次学习:")
    print(f"  模型觉得填“好”的概率是: {y_hat:.4f}")
    
    # 第五步：更新参数！(这就是那个公式的变体)
    # 对于正确答案所在的词，惩罚它；对于其他词，奖励它们（因为目标概率是0）
    for i in range(5):
        if i == y_true_index:
            w[i] = w[i] - 学习率 * 偏差  # 惩罚：往小了调
        else:
            w[i] = w[i] - 学习率 * probabilities[i] # 奖励：往小了调（因为真实概率是0）
            
    print(f"  模型觉得填“好”的概率变成了: {probabilities[y_true_index] - 学习率 * 偏差:.4f}\n")

print("学习结束！模型现在比以前更懂怎么填空了。")