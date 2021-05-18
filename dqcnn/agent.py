import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop

from dqcnn.model import DQCNN
from dqcnn.utils import ReplayMemory, Transition


class Agent(object):

    def __init__(self, display_state_w, display_state_h, device, replay_memory_size=20000, gamma=0.9, lr=1e-2, eps_start=0.9, eps_end=0.2, eps_decay=20000, policy_net_path=None):

        self.display_state_w = display_state_w
        self.display_state_h = display_state_h

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.device = device

        self.policy_net = DQCNN(display_state_w, display_state_h, 1, device).to(device)
        self.target_net = DQCNN(display_state_w, display_state_h, 1, device).to(device)

        # if policy_net_path is not None:
        #     state_dict = torch.load(policy_net_path)
        #     self.policy_net.load_state_dict(state_dict)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = RMSprop(self.policy_net.parameters())
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(replay_memory_size)


    def get_exploration_rate(self, episode):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * (episode / self.eps_decay))


    def select_action(self, possible_states, episode):
        # Get the exploration rate (exploration vs. exploitation --> explore more at beginning, then start exploiting the gained knowledge more)
        exploration_rate = self.get_exploration_rate(episode)

        if random.random() < exploration_rate:
            return random.choice(list(possible_states.keys())) # Exploration
            
        else:
            with torch.no_grad():
                state_action_map = {idx: key for idx, key in enumerate(possible_states.keys())}
                states = torch.tensor(np.vstack(list(possible_states.values())), device=self.device)
                prediction = self.policy_net(states)
                max_idx = prediction.argmax(dim=0)
                action = state_action_map[max_idx.item()]
                return action

    def optimize(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        # action_batch = torch.stack(batch.action).unsqueeze(0)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch)
        # state_action_values = state_action_values.gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss_cp = loss.item()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss_cp

    def add_to_memory(self, *args):
        self.memory.push(*args)

    def save_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


