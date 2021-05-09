import numpy as np
import random
import torch
from torch.optim import Adam

from dqn.model import DQN
from dqn.utils import ReplayMemory, Transition


class Agent(object):

    def __init__(self, state_size, num_actions, device, replay_memory_size=20000, gamma=0.9, lr=1e-3, eps_start=0.9, eps_end=0.2, eps_decay=20000):
        self.current_step = 0
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma

        self.policy_net = DQN(state_size, num_actions).to(device)
        self.target_net = DQN(state_size, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_memory_size)
        self.criterion = torch.nn.MSELoss()
        self.num_actions = num_actions


    def get_exploration_rate(self, step):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * (step / self.eps_decay))

    def select_action(self, state):
        # Get the exploration rate (exploration vs. exploitation --> explore more at beginning, then start exploiting the gained knowledge more)
        exploration_rate = self.get_exploration_rate(self.current_step)
        self.current_step += 1

        if random.random() < exploration_rate:
            return random.randrange(self.num_actions) # Exploration
        else:
            with torch.no_grad():
                prediction = self.policy_net(state.float()) # Exploitation
                return prediction.argmax(dim=0)

    def optimize(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Sample the experiences (state transitions)
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.vstack(batch.state)
        action_batch = torch.stack(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.vstack(batch.next_state)

        # Calculate the state-action pairs, predicted by the policy net
        state_action_values = self.policy_net(state_batch.float()).gather(0, action_batch)

        # Calculate the next-state values, used for calculating the loss
        next_state_action_values = self.target_net(next_state_batch.float()).max(1)[0].detach()

        # Calulate the expected Q-values
        expected_state_action_values = reward_batch + (self.gamma * next_state_action_values)

        # Calculate the loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss_cp = loss.item()

        # Optimize the neural network
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss_cp

    def add_to_memory(self, *args):
        self.memory.push(*args)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset(self):
        self.current_step = 0

