import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
# import transformers

from collections import namedtuple
import torch

import numpy as np
""" 测试显卡 """
print(torch.__version__)
print(torch.cuda.is_available())

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

# # 广播 tensor2 到 [2, 3, 4]
# tensor2_broadcasted = tensor2.expand(tensor1.shape)
# print(tensor2_broadcasted)



# done = 1
# done = torch.tensor(done).float().to("cuda")
# print(1- done)

# def get_rand(*arg):
#     print(arg)

# get_rand((1,2,3))  # ((1, 2, 3),)
agents = ["ag1", "ag2"]
obs = {agent_id: torch.randn((256, 1024), device="cuda") for agent_id in agents}
a = torch.randn((15)).unsqueeze(1).unsqueeze(2)
b = 0.96 * torch.randn((15,256,1))
c = a+b
print(c + b)




# c = torch.cat(c+d,1)
# print(c.shape)