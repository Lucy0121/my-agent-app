import time

# 假设有 10 万句话，每句话有 100 个词
num_sentences = 100000
num_words = 100

# 模拟数据
sentences = [[1] * num_words for _ in range(num_sentences)]
weights = [0.5] * num_words

start_time = time.time()

# 用我们现在的老办法（for 循环加 sum）来算这 10 万句话的得分
z_list = []
for sentence in sentences:
    z = sum(w * x for w, x in zip(weights, sentence))
    z_list.append(z)

end_time = time.time()
print(f"用传统的 for 循环，算 10 万句话耗时: {end_time - start_time:.4f} 秒")