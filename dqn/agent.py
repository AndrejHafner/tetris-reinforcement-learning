import numpy as np
import random
import torch

class Agent(object):

    def __init__(self, num_actions, eps_start=0.9, eps_end=0.05, eps_decay=200):
        self.current_step = 0
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def get_exploration_rate(self, step):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * (step / self.eps_decay))

    def select_action(self, state, policy_net, eps):
        # Get the exploration rate (exploration vs. exploitation --> explore more at beginning, then start exploiting the gained knowledge more)
        exploration_rate = self.get_exploration_rate(self.current_step)
        self.current_step += 1

        if random.random() > exploration_rate:
            return random.randrange(self.num_actions) # Exploration
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).item() # Exploitation



