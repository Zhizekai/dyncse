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
from collections import namedtuple
import random
import torch
from torch.distributions import Bernoulli
import os


epsilon = 1e-8


class PolicyNet(nn.Module):
    """
    策略网络
    """

    def __init__(self, num_teacher, embedding_length, device, batch_size):
        super(PolicyNet, self).__init__()
        self.num_teacher = num_teacher  # 假设有两个老师模型

        # 定义权重和偏置
        self.W1 = nn.Parameter(torch.FloatTensor(embedding_length, 128).uniform_(-0.5, 0.5))  # 768,128
        # self.W2 = nn.Parameter(torch.FloatTensor(128, 128).uniform_(-0.5, 0.5)) #128,128
        self.W2 = nn.Parameter(torch.FloatTensor(batch_size, 128).uniform_(-0.5, 0.5))  # 128,128
        self.W3 = nn.Parameter(torch.FloatTensor(num_teacher, 128).uniform_(-0.5, 0.5))  # 2,128
        self.b = nn.Parameter(torch.FloatTensor(1, 128).uniform_(-0.5, 0.5))

        self.fc_alpha = nn.Parameter(torch.FloatTensor(128, 1).uniform_(-0.5, 0.5))  # 输出 alpha
        self.fc_beta = nn.Parameter(torch.FloatTensor(128, 1).uniform_(-0.5, 0.5))  # 输出 beta

        self.epsilon = 1.0  # 初始epsilon值
        self.epsilon_min = 0.01  # epsilon的最小值
        self.epsilon_decay = 0.995  # epsilon衰减率
        self.device = device

    def forward(self, x1, x2, x3):
        # x1: (num_sent, batch_size, embedding_length) 2,128,768
        # x2: (num_teacher, batch_size, batch_size) 2,128,128
        # x3: (num_teacher, 1) 1,2

        # 将输入与相应的权重相乘
        x1_ = torch.matmul(x1, self.W1)  # 2,128,128
        x2_ = torch.matmul(x2, self.W2)  # 2,128,128  这个出问题了 x2 最后一个维度成64 了，原来是（2,256,256）现在是（2， 256， 64）

        # 将第三个输入乘以其权重
        x3_ = torch.matmul(x3, self.W3)  # 1,128
        # 将所有结果相加并加上偏置
        scaled_out = torch.relu(x1_ + x2_ + x3_ + self.b)
        # scaled_out = torch.clamp(scaled_out, min=1e-5, max=10 - 1e-5) #2,128,128
        # 重塑scaled_out以匹配全连接层的期望输入维度
        scaled_out_reshaped = scaled_out.view(-1, 128)  # 形状现在是 [-1, 128]

        # 添加全连接层以生成形状为 (batch_size*num_teacher, num_teacher) 的输出
        # 输出两个参数 alpha 和 beta
        alpha = torch.matmul(scaled_out_reshaped, self.fc_alpha)
        beta = torch.matmul(scaled_out_reshaped, self.fc_beta)

        alpha = F.softplus(alpha).mean() + 50
        # alpha = torch.relu(alpha).mean() + 50
        beta = F.softplus(beta).mean() + 50
        # beta = torch.relu(beta).mean() + 50
        weights = [alpha, beta]
        return weights

    def take_action(self, state):
        # state当中有三个变量 embeddings_tensor,soft_lable,concatenated_loss，都是get_environment_state 函数生成的

        # 直接调用网络来生成一个概率  state tensor([[510.9451, 519.7139]]
        weights = self.forward(*state)  # 这个是策略网络的前向传播输出，weights当中 是[alpha, beta]，*state 是解包成单独的参数传进去

        # 计算概率分布
        # dist = torch.distributions.Normal(weights[0].float(), weights[1].float())
        dist = torch.distributions.beta.Beta(weights[0].float(), weights[1].float())  # weights[0]是alpha，weights[1]是beta
        action = dist.sample().to(self.device)

        # 更新epsilon值，但不让它低于最小值
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        # print("policy action：",action)
        # input()
        return action, weights

    def test_policy(self, state):
        avg_probability = self.forward(*state).to(self.device)  # 获取动作概率
        action = torch.distributions.Bernoulli(avg_probability).sample().to(self.device)  # 选择概率最高的动作
        return action, avg_probability


# 假设你有一个表示概率的张量
probs = torch.tensor([0.5], requires_grad=True)
Transition = namedtuple("Transion", ("state", "next_state", "action", "weights", "reward", "value"))


# 经验回放技术，DQN里面用到的


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 增加一个空元素
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return self.memory

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = []  # 清空memory
        self.position = 0  # 重置position为0


# ddpg 的目标网络软更新
def soft_update(net, target_net):
    for param_target, param in zip(target_net.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - 0.005) + param.data * 0.005)


def compute_advantage(gamma, lmbda, td_delta, device):
    # td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0

    # td_delta[::-1] 这个就是反转列表，-1是步长，也就是反向取值，前两个:: 就是起始位置和结束位置
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float, device=device)


def optimize_model(memory, policy_net, policy_net_target, critic, critic_target, device, lr=1e-4):
    """
    这个有可能相当于update 函数
    """

    # 设置一些超参数
    CLIP_EPSILON = 0.2
    NUM_PPO_UPDATES = 1  # 选择适当的更新次数
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)  # 价值网络的优化器
    BATCH_SIZE = 10
    # 确保内存中有数据
    gamma = 0.99
    gae_lambda = 0.95
    # 这个memory 是收集之前的所有action 和状态之后一起更新网络参数
    num_batches = len(memory) // BATCH_SIZE
    all_transitions = memory.sample()

    # Transition里面放的'state', 'action', 'weights', 'reward', 'value'
    # weights 是take_action() 里面的，可能应该对应props变量
    batch = Transition(*zip(*all_transitions))
    for _ in range(NUM_PPO_UPDATES):

        # 准备数据，这里的acion 是从经验池里拿回来的
        action_batch = torch.cat(list(map(lambda a: torch.tensor([a], device=device), batch.action)))
        # print("action 动作概率：",action_batch)
        reward_batch = torch.cat(list(map(lambda r: torch.tensor([r], device=device), batch.reward)))
        episode_size = reward_batch.shape[0]  # 伪章节长度

        old_weights = torch.cat(list(map(lambda r: torch.tensor(r, device=device), batch.weights)))
        old_weights = old_weights.view(-1, 2)
        value = torch.cat([torch.tensor([v], device=device) for v in batch.value])
        # >>>>>>>>>>>
        state_batch = [torch.cat((i[0], i[1]), 0) for i in zip(*batch.state)]
        next_state_batch = [torch.cat((i[0], i[1]), 0) for i in zip(*batch.next_state)]
        # >>>>>>>>>>>

        # DDPG 算法
        # next_q_values = policy_net_target.take_action(next_state) 下面是这行代码的实现
        # >>>>>>>>>>>>>>>>>
        next_q_values = []
        for next_state in batch.next_state:
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.half().to(device)  # 转换为半精度并移至正确的设备
            elif isinstance(next_state, list):
                next_state = [s.float().to(device) for s in next_state]
            next_q_value, _ = policy_net_target.take_action(next_state)
            # weights1 = torch.stack([weight[0], weight[1]]).unsqueeze(0)
            next_q_values.append(next_q_value)
        next_q_values = torch.stack(next_q_values)

        # next_q_values = critic_target(next_states,next_q_values) # 第二个参数可能是action 下面是这个的实现
        next_q_values_2 = []
        for next_state in batch.next_state:
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.half().to(device)  # 转换为半精度并移至正确的设备
            elif isinstance(next_state, list):
                next_state = [s.float().to(device) for s in next_state]
            next_q_value_sum = 0
            for action_q in next_q_values:

                # 获取价值函数的输出
                next_q_value = critic_target(*next_state, action=action_q)
                next_q_value_sum += next_q_value
            next_q_values_2.append(next_q_value_sum)
        next_q_values_2 = torch.stack(next_q_values_2)
        # print(next_q_values_2)  # tensor([0.1948, 0.1888, 0.1929, 0.1324, 0.1510, 0.1791, 0.1885, 0.1925, 0.1838], device='cuda:0', grad_fn=<StackBackward0>)
        # input()
        # next_q_values_2 = next_q_values_2.view((episode_size,-1))
        gamma = 0.98
        q_targets = reward_batch + gamma * next_q_values_2
        """
        reward_batch([-0.3108, -0.1372, -0.0750, -0.1230, -0.0999, -0.0674, -0.0613, -0.0556,
        -0.0668], device='cuda:0', grad_fn=<AddBackward0>)
        next_q_values_2：tensor([0.1948, 0.1888, 0.1929, 0.1324, 0.1510, 0.1791, 0.1885, 0.1925, 0.1838],
            device='cuda:0', grad_fn=<StackBackward0>)
        q_targets：tensor([-0.3108, -0.1372, -0.0750, -0.1230, -0.0999, -0.0674, -0.0613, -0.0556,
        -0.0668], device='cuda:0', grad_fn=<AddBackward0>)
        """
        q_values = []
        for state in batch.state:
            if isinstance(state, torch.Tensor):
                state = state.half().to(device)  # 转换为半精度并移至正确的设备
            elif isinstance(state, list):
                state = [s.float().to(device) for s in state]
            q_value_sum = 0
            for action_q in action_batch:
                # 获取价值函数的输出
                q_value = critic(*state, action=action_q)
                q_value_sum += q_value
            q_values.append(q_value_sum)
        q_values = torch.stack(q_values)
        # q_values = critic(batch.state, action_batch) 上面是实际的代码

        critics_loss = torch.mean(F.mse_loss(q_values, q_targets))
        critic.optimizer.zero_grad()
        critics_loss.backward()
        critic.optimizer.step()  # 更新价值网络

        # actor_q_values = policy_net.take_action(batch.state)  # 当前状态的每个动作的价值[n, n_actions] 下面是这个代码的实现
        actor_q_values = []
        for state in batch.state:
            if isinstance(state, torch.Tensor):
                state = state.half().to(device)  # 转换为半精度并移至正确的设备
            elif isinstance(state, list):
                state = [s.float().to(device) for s in state]
            actor_q_value, _ = policy_net.take_action(state)
            actor_q_values.append(actor_q_value)
        actor_q_values = torch.stack(actor_q_values)  # actor_q_values提升维度

        # score = critic(batch.state, actor_q_values)  # 当前状态的动作价值
        actor_q_values_2 = []
        for state in batch.state:
            if isinstance(state, torch.Tensor):
                state = state.half().to(device)  # 转换为半精度并移至正确的设备
            elif isinstance(state, list):
                state = [s.float().to(device) for s in state]
            q_value_sum = 0
            for action_q in actor_q_values:
                # 获取价值函数的输出
                q_value = critic(*state, action=action_q)
                q_value_sum += q_value
            actor_q_values_2.append(q_value_sum)
        score = torch.stack(actor_q_values_2)
        actor_loss = -torch.mean(score)
        # print("actor_loss:", actor_loss)
        # input()

        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()  # 更新策略网络

        # 软更新策略网络
        soft_update(policy_net, policy_net_target)
        # 软更新价值网络
        soft_update(critic, critic_target)
        # print("actor_loss:", actor_loss)
        # input()

        # 计算回报 PPO
        # advantage = torch.zeros(len(reward_batch), dtype=torch.float32, device=device)
        # for t in range(len(reward_batch) - 1):  # 逆序时序差分值
        #     discount = 1
        #     a_t = 0
        #     for k in range(t, len(reward_batch) - 1):
        #         a_t += 0.99 * (reward_batch[k] + gamma * value[k + 1] * - value[k])
        #     advantage[t] = a_t

        # weights_list = []
        # for state in batch.state:
        #     if isinstance(state, torch.Tensor):
        #         state = state.half().to(device)  # 转换为半精度并移至正确的设备
        #     elif isinstance(state, list):
        #         state = [s.float().to(device) for s in state]
        #     weight = policy_net(*state)
        #     weights1 = torch.stack([weight[0], weight[1]]).unsqueeze(0)
        #     weights_list.append(weights1)
        # weights = torch.cat(weights_list, dim=0)

        # 计算对数概率和概率比率
        # # m = torch.distributions.Normal(weights[:, 0].float(), weights[:, 1].float())
        # m = torch.distributions.Beta(weights[:, 0].float(), weights[:, 1].float())
        # log_probs = m.log_prob(action_batch)
        # # beta_distribution = torch.distributions.Normal(old_weights[:, 0].float(), old_weights[:, 1].float())
        # beta_distribution = torch.distributions.Beta(old_weights[:, 0].float(), old_weights[:, 1].float())
        # old_log_probs = beta_distribution.log_prob(action_batch)
        # ratio = torch.exp(log_probs - old_log_probs)

        # 计算 clipped ratio 和 PPO 目标函数
        # clip_ratio = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)  # 截断 1.0 - CLIP_EPSILON 是最小值
        # surrogate1 = ratio * advantage
        # surrogate2 = clip_ratio * advantage
        # ppo_loss = -torch.min(surrogate1, surrogate2).mean()

        # value_list = []
        # for state in batch.state:
        #     # 为什么要把状态变成半精度
        #     if isinstance(state, torch.Tensor):
        #         state = state.half().to(device)  # 转换为半精度并移至正确的设备
        #     elif isinstance(state, list):
        #         state = [s.float().to(device) for s in state]
        #     critic_value = critic(*state, action = None).unsqueeze(0)  # 获取评价网络的输出
        #     value_list.append(critic_value)
        # values = torch.cat(value_list, dim=0)
        # returns = advantage + value

        # critic_loss = (returns - values) ** 2
        # critic_loss = critic_loss.mean()
        # total_loss = ppo_loss + 0.5 * critic_loss

        # optimizer.zero_grad()
        # critic.optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step() # 更新actor网络
        # critic.optimizer.step() # 更新critic网络


# 评价网络
class Critic(nn.Module):
    def __init__(
        self,
        input_dim,
        num_teacher,
        embedding_length,
        hidden_dim=[256, 128],
        output_dim=1,
        batch_size=128,
    ):
        super(Critic, self).__init__()

        # 定义权重和偏置
        self.W1 = nn.Parameter(torch.FloatTensor(embedding_length, 128).uniform_(-0.5, 0.5))  # 768,128
        self.W2 = nn.Parameter(torch.FloatTensor(batch_size, 128).uniform_(-0.5, 0.5))  # 128,128
        self.W3 = nn.Parameter(torch.FloatTensor(num_teacher, 128).uniform_(-0.5, 0.5))  # 2,128
        self.W4 = nn.Parameter(torch.FloatTensor(1, 128).uniform_(-0.5, 0.5))
        self.b = nn.Parameter(torch.FloatTensor(1, 128).uniform_(-0.5, 0.5))  # 1,128

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)

    def forward(self, x1, x2, x3, action):
        # x1: (num_sent, batch_size, embedding_length) 2,128,768
        # x2: (num_teacher, batch_size, batch_size) 2,128,128
        # x3: (num_teacher, 1) 1,2

        # 将输入与相应的权重相乘
        x1_ = torch.matmul(x1, self.W1)  # 2,128,128
        x2_ = torch.matmul(x2, self.W2)  # 2,128,128

        # 将第三个输入乘以其权重
        x3_ = torch.matmul(x3, self.W3)  # 1,128
        # print("action:",action)
        action_ = torch.matmul(action.view((1, 1)), self.W4)  # 1,128
        # print("action:",action)
        # input()

        # 将所有结果相加并加上偏置
        scaled_out = torch.sigmoid(x1_ + x2_ + x3_ + action_ + self.b)
        # scaled_out = torch.sigmoid(x1_ + x2_ + x3_  + self.b)
        # 限制输出范围并保证它是1x1维度
        scaled_out = torch.clamp(scaled_out, min=1e-5, max=1 - 1e-5)  # 2,128,128
        # 重塑 scaled_out 以匹配全连接层的期望输入维度
        scaled_out_reshaped = scaled_out.view(-1, 128)  # 形状现在是 [-1, 128]

        # 将重塑后的结果传递给 Critic 网络的模型部分
        critic_out = self.model(scaled_out_reshaped)
        critic_out = critic_out.mean()
        return critic_out
