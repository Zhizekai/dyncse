import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn as nn
from copy import deepcopy
import random
from tqdm import tqdm
import math

import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, num_teacher, embedding_length, device):
        super(PolicyNet, self).__init__()
        self.num_teacher = num_teacher  # 假设有两个老师模型

        # 定义权重和偏置
        self.W1 = nn.Parameter(torch.FloatTensor(embedding_length, 128).uniform_(-0.5, 0.5)) #768,128
        self.W2 = nn.Parameter(torch.FloatTensor(128, 128).uniform_(-0.5, 0.5)) #128,128
        self.W3 = nn.Parameter(torch.FloatTensor(num_teacher, 128).uniform_(-0.5, 0.5)) #2,128
        self.b = nn.Parameter(torch.FloatTensor(1, 128).uniform_(-0.5, 0.5))

        # 定义输出概率的全连接层
        self.fc = nn.Linear(128, num_teacher)

        self.epsilon = 1.0  # 初始epsilon值
        self.epsilon_min = 0.01  # epsilon的最小值
        self.epsilon_decay = 0.995  # epsilon衰减率
        self.device = device
    def forward(self, x1, x2, x3):
        # x1: (num_teacher, batch_size, embedding_length) 2,128,768
        # x2: (num_teacher, batch_size, batch_size) 2,128,128
        # x3: (num_teacher, 1) 1,2

        # 将输入与相应的权重相乘
        x1_ = torch.matmul(x1, self.W1)  #2,128,128
        x2_ = torch.matmul(x2, self.W2)  #2,128,128

        # 将第三个输入乘以其权重
        x3_ = torch.matmul(x3, self.W3)  #1,128

        # 将所有结果相加并加上偏置
        scaled_out = x1_ + x2_ + x3_ + self.b
        # 限制输出范围并保证它是1x1维度
        scaled_out = torch.clamp(scaled_out, min=1e-5, max=1 - 1e-5) #2,128,128

        # 重塑scaled_out以匹配全连接层的期望输入维度
        scaled_out_reshaped = scaled_out.view(-1, 128)  # 形状现在是 [-1, 128]
        # 添加全连接层以生成形状为 (batch_size*num_teacher, num_teacher) 的输出
        fc_out = self.fc(scaled_out_reshaped)

        # 使用sigmoid函数确保输出是概率
        probabilities = torch.sigmoid(fc_out)

        # 计算所有概率的平均值得到单个概率值
        avg_probability = probabilities.mean()

        return avg_probability

    def take_action(self, state):
        # 直接调用网络来生成一个概率
        avg_probability = self.forward(*state).to(self.device)

        if random.random() < self.epsilon:
            # 以 epsilon 的概率随机选择动作
            action = torch.tensor([random.choice([0, 1])], dtype=torch.float32).to(self.device)
        else:
            # 否则根据平均概率采样动作
            action = torch.distributions.Bernoulli(avg_probability).sample().to(self.device)

        # 更新epsilon值，但不让它低于最小值
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return action, avg_probability

    def test_policy(self, state):
        avg_probability = self.forward(*state).to(self.device) # 获取动作概率
        action = torch.distributions.Bernoulli(avg_probability).sample().to(self.device)  # 选择概率最高的动作
        return action, avg_probability


from collections import namedtuple
import random
import torch
from torch.distributions import Bernoulli

# 假设你有一个表示概率的张量
probs = torch.tensor([0.5], requires_grad=True)  # 示例概

Transition = namedtuple('Transion',
                        ('state', 'action', 'weights', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return self.memory

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = []  # 清空memory
        self.position = 0  # 重置position为0


def optimize_model(memory, policy_net, device, lr=1e-4):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # 确保内存中有数据
    if len(memory) == 0:
        return

    # 使用内存中的所有数据
    transitions = memory.sample()
    batch = Transition(*zip(*transitions))

    # 准备数据
    # Prepare data
    action_batch = torch.cat(list(map(lambda a: torch.tensor([a], device=device), batch.action)))
    reward_batch = torch.cat(list(map(lambda r: torch.tensor([r], device=device), batch.reward)))



    weights_list = []
    for state in batch.state:
        if isinstance(state, torch.Tensor):
            state = state.half().to(device)  # 转换为半精度并移至正确的设备
        elif isinstance(state, list):
            state = [s.float().to(device) for s in state]
        weight = policy_net(*state).unsqueeze(0)
        weights_list.append(weight)
    weights = torch.cat(weights_list, dim=0)

    # 计算对数概率
    m = Bernoulli(weights)
    log_probs = m.log_prob(action_batch)

    # 计算回报
    G = 0
    returns = []
    for r in reward_batch.flip(dims=(0,)):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns, device=device)

    # 计算策略梯度损失
    policy_loss = []
    for log_prob, G in zip(log_probs, returns):
        policy_loss.append(-log_prob * G)
    policy_loss = torch.cat([l.unsqueeze(0) for l in policy_loss]).sum()

    # 优化模型
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()



class actor(nn.Module):
    def __init__(self, policyNet, tau):
        super(actor, self).__init__()
        self.target_policy = policyNet
        self.active_policy = policyNet

    def get_target_logOutput(self, x1, x2, x3):
        out = self.target_policy(x1, x2, x3)
        logOut = torch.log(out)
        return logOut

    def get_target_output(self, x1, x2, x3, scope):
        if scope == "target":
            out = self.target_policy(x1, x2, x3)
        if scope == "active":
            out = self.active_policy(x1, x2, x3)
        return out

    def get_gradient(self, x1, x2, x3, reward, scope):
        if scope == "target":
            out = self.target_policy(x1, x2, x3)
            logout = torch.log(out).view(-1)
            index = reward.index(0)
            index = (index + 1) % 2
            # print(out, reward, index, logout[index].view(-1), logout)
            # print(logout[index].view(-1))
            grad = torch.autograd.grad(logout[index].view(-1),
                                       self.target_policy.parameters())  # torch.cuda.FloatTensor(reward[index])
            # print(grad[0].size(), grad[1].size(), grad[2].size())
            # print(grad[0], grad[1], grad[2])
            grad[0].data = grad[0].data * reward[index]
            grad[1].data = grad[1].data * reward[index]
            grad[2].data = grad[2].data * reward[index]
            # print(grad[0], grad[1], grad[2])
            return grad
        if scope == "active":
            out = self.active_policy(x1, x2, x3)
        return out

    def assign_active_network_gradients(self, grad1, grad2, grad3):
        params = [grad1, grad2, grad3]
        i = 0
        for name, x in self.active_policy.named_parameters():
            x.grad = deepcopy(params[i])
            i += 1

    def update_target_network(self):
        params = []
        for name, x in self.active_policy.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.target_policy.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1 - tau))
            i += 1

    def assign_active_network(self):
        params = []
        for name, x in self.target_policy.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.active_policy.named_parameters():
            x.data = deepcopy(params[i].data)
            i += 1


import random




# class critic(nn.Module):
#     def __init__(self, model=None):
#         super(critic, self).__init__()
#         self.active_pred = model
#         self.target_pred = deepcopy(model)
#
#     def forward(self, x, scope):
#         if scope == "target":
#             _, last_hidden_state = self.target_pred(x, output_hidden_states=True, return_dict=True, sent_emb=False)
#         if scope == "active":
#             _, last_hidden_state = self.active_pred(x, output_hidden_states=True, return_dict=True, sent_emb=False)
#         return last_hidden_state
#
#     def assign_target_network(self):
#         params = []
#         for name, x in self.active_pred.named_parameters():
#             params.append(x)
#         i=0
#         for name, x in self.target_pred.named_parameters():
#             x.data = deepcopy(params[i].data)
#             i+=1
#
#     def update_target_network(self):
#         params = []
#         for name, x in self.active_pred.named_parameters():
#             params.append(x)
#         i=0
#         for name, x in self.target_pred.named_parameters():
#             x.data = deepcopy(params[i].data * (tau) + x.data * (1-tau))
#             i+=1
#
#     def assign_active_network(self):
#         params = []
#         for name, x in self.target_pred.named_parameters():
#             params.append(x)
#         i=0
#         for name, x in self.active_pred.named_parameters():
#             x.data = deepcopy(params[i].data)
#             i+=1
#
#     def assign_active_network_gradients(self):
#         params = []
#         for name, x in self.target_pred.named_parameters():
#             params.append(x)
#         i=0
#         for name, x in self.active_pred.named_parameters():
#             x.grad = deepcopy(params[i].grad)
#             i+=1
#         for name, x in self.target_pred.named_parameters():
#             x.grad = None