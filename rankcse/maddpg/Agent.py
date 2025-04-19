from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


class Agent:
    """
    Agent that can interact with environment from pettingzoo
    obs_dim: 观测维度，actor 策略模型的输入维度
    act_dim: 动作维度, actor模型的输出维度
    global_obs_dim: 评价模型的输入维度 1024
    """

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr):
        self.actor = MLPNetwork(obs_dim, act_dim)

        # critic input all the observations and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        self.critic = MLPNetwork(global_obs_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, model_out=False):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])

        # 获取张量的维度数
        num_dims = obs.dim()
        
        # 如果张量有两个维度，则在前面添加一个维度
        if num_dims == 2:
            obs = obs.unsqueeze(0)  # 在第0维度添加一个维度
        # 如果张量有三个维度，则保持不变
        elif num_dims == 3:
            pass
        else:
            raise ValueError("输入张量的维度数必须是2或3")

        actions = []
        logits_list = []
        for obs_item in obs:
            logits = self.target_actor(obs_item) # obs_item (256, 1024)
            # action = self.gumbel_softmax(logits)
            action = F.gumbel_softmax(logits, hard=True)

            x_min, x_max = logits.min(), logits.max()
            x_normalized = (logits - x_min) / (x_max - x_min)
            action = F.softplus(x_normalized).mean()
            action = action.squeeze(0).detach()
            action = action.item()
            actions.append(action)  # action (15, 256, 1)
            logits_list.append(logits)


        action = torch.tensor(actions, device="cuda").unsqueeze(1)
        logits = torch.stack(logits_list, dim=0)  # 在第0维度上拼接

        if model_out:
            return action, logits
        return action

    def target_action(self, obs):
        # obs torch.Size([15, 256, 1024])
        actions = []
        for obs_item in obs:
            logits = self.target_actor(obs_item) # obs_item (256, 1024)
            # action = self.gumbel_softmax(logits)
            action = F.gumbel_softmax(logits, hard=True)

            x_min, x_max = logits.min(), logits.max()
            x_normalized = (logits - x_min) / (x_max - x_min)
            action = F.softplus(x_normalized).mean()
            action = action.squeeze(0).detach()
            action = action.item()
            actions.append(action)  # action (15, 256, 1)
        return torch.tensor(actions, device="cuda").unsqueeze(1)

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        # state_list[0].shape :torch.Size([15, 256, 1024]) act_list: torch.Size([15, 1])

        act_list = [x.unsqueeze(1).repeat(1, 256, 1) for x in act_list]  # torch.Size([15, 256, 1]) act_list形状不对
        x = torch.cat(state_list + act_list, 2)
        c_value = self.critic(x).squeeze(1)
        return  c_value # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        act_list = [x.unsqueeze(1).repeat(1, 256, 1) for x in act_list]
        x = torch.cat(state_list + act_list, 2)# x.shape torch.Size([15, 256, 2050])
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

        # 检查是否有 GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将模型移动到 GPU
        model = model.to(device)
        self.net = model

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
