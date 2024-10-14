from collections import namedtuple
import torch
import numpy as np

# Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
# #生成一个列表，列表中是元组
# a = [(1,2,3,4),
#     (11,12,13,14),
#     (21,22,23,24),
#     (31,32,33,34)]
# b = zip(*a)

# b = list(b) # [(1, 11, 21, 31), (2, 12, 22, 32), (3, 13, 23, 33), (4, 14, 24, 34)]
# print(b)
# c = Transition(*zip(*a))
# print(c) #Transition(state=(1, 11, 21, 31), action=(2, 12, 22, 32), next_state=(3, 13, 23, 33), reward=(4, 14, 24, 34))
# c = list(c)

# 保存中间变量的函数
# a = {'state_batch':batch.state, 'next_state_batch':batch.next_state}
# torch.save(a, "./myDict.pth")

# 这玩意第一层是个元组 ，第二层是list ，最里面才是包着的tensor
# state_batch,next_state_batch = torch.load("./myDict.pth").values()

# next_state_batch = [ torch.cat((i[0],i[1]),0) for i  in zip(*next_state_batch)]
# print(next_state_batch)

# tensor1 = torch.randn(1, 10)  # 形状: [2, 3, 4]
# tensor2 = torch.randn(1, 10)  # 形状: [1, 3, 1]

# # 广播 tensor2 到 [2, 3, 4]
# tensor2_broadcasted = tensor2.expand(tensor1.shape)
# print(tensor2_broadcasted)

import matplotlib.pyplot as plt
import numpy as np

# Data: (l_uniform, l_align, score, label)
data = [
    (-3.9, 0.6, 66.28, 'BERT-whitening'),
    (-3.6, 0.6, 66.55, 'BERT-flow'),
    (-3.4, 0.3, 72.74, 'ConSERT'),
    (-2.8, 0.2, 76.25, 'SimCSE'),
    (-2.4, 0.2, 80.05, 'RankCSE_ListNet'),
    (-2.0, 0.2, 80.36, 'RankCSE_ListMLE'),
    (-1.9, 0.1, 78.11, 'PCL'),
    (-1.8, 0.2, 78.49, 'DiffCSE'),
    (-1.5, 0.2, 56.70, 'Avg. BERT'),
]

# Extract values from the data
l_uniform = [item[0] for item in data]
l_align = [item[1] for item in data]
scores = [item[2] for item in data]
labels = [item[3] for item in data]

# Create the plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(l_uniform, l_align, c=scores, cmap='hot', s=100, edgecolor='black', marker='o')

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Score', rotation=270, labelpad=15)

# Add annotations
for i, label in enumerate(labels):
    plt.annotate(f'{label}({scores[i]})', (l_uniform[i], l_align[i]), textcoords="offset points", xytext=(5,-5), ha='center')

# Set labels and title
plt.xlabel(r'$\ell_{uniform}$')
plt.ylabel(r'$\ell_{align}$')
plt.title('Comparison of Models on Uniformity and Alignment')
plt.savefig('./test2.png')
# Show the plot
plt.show()



