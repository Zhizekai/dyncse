import numpy as np
import torch


class Buffer:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity
        self.obs = torch.zeros((capacity, 256, obs_dim) ,device= device)
        self.action = torch.zeros((capacity, act_dim), device=device)
        self.reward = torch.zeros(capacity, device=device)
        self.next_obs = torch.zeros((capacity, 256, obs_dim), device=device)
        self.done = torch.zeros(capacity, dtype=bool, device=device)

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        """ add an experience to the memory """
        self.obs[self._index] = obs.cpu()
        self.action[self._index] = action.cpu()
        self.reward[self._index] = reward.cpu()
        self.next_obs[self._index] = next_obs.cpu()
        self.done[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`

        
        obs = obs.to(device)  # torch.Size([batch_size, state_dim])
        action = action.to(device)  # torch.Size([batch_size, action_dim])
        reward = reward.to(device)  # just a tensor with length: batch_size
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = next_obs.to(device)  # Size([batch_size, state_dim])
        done = torch.tensor(done).float().to(device)  # just a tensor with length: batch_size

        return obs, action, reward, next_obs, done

    def __len__(self):
        return self._size
