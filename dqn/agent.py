import numpy as np
import random
import torch
from torch.optim import Adam

from dqn.model import DQN
from dqn.utils import ReplayMemory, Transition


class Agent(object):

    def __init__(self, state_size, device, replay_memory_size=20000, gamma=0.9, lr=1e-2, eps_start=0.9, eps_end=0.2, eps_decay=20000, policy_net_path=None):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.device = device

        self.policy_net = DQN(state_size, 1).to(device)
        self.target_net = DQN(state_size, 1).to(device)
        if policy_net_path is not None:
            state_dict = torch.load(policy_net_path)
            self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_memory_size)
        # self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.MSELoss()


    def get_exploration_rate(self, episode):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * (episode / self.eps_decay))


    def select_action(self, state_action_pairs, episode):
        # Get the exploration rate (exploration vs. exploitation --> explore more at beginning, then start exploiting the gained knowledge more)
        exploration_rate = self.get_exploration_rate(episode)

        if random.random() < exploration_rate:
            return random.choice(list(state_action_pairs.keys())) # Exploration
        else:
            with torch.no_grad():
                state_action_map = {idx: key for idx, key in enumerate(state_action_pairs.keys())}
                print("STATE ACTIONS: ")
                print(state_action_map)

                states = torch.tensor(np.vstack(list(state_action_pairs.values())), device=self.device)

                print("STATES: ")
                print(states)
                prediction = self.policy_net(states.float()) # Exploitation

                print("PREDICTION: ")
                print(prediction)

                max_idx = prediction.argmax(dim=0)

                print("MAX_IDX: ")
                print(max_idx)
                print(max_idx.item())

                action = state_action_map[max_idx.item()]

                print("ACTION: ")
                print(action)

                return action

    def optimize(self, batch_size, iteration):
        if len(self.memory) < batch_size:
            return

        # Sample the experiences (state transitions)
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.vstack(batch.state)
        action_batch = torch.cat(batch.action)
        print(action_batch)
        action_batch = torch.stack(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.vstack(batch.next_state)

        # Calculate the state-action pairs, predicted by the policy net
        state_action_values = self.policy_net(state_batch.float())

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

        if iteration % 100 == 0:
            torch.save(self.policy_net.state_dict(), f"./dqn_checkpoint_mse.pth")

        return loss_cp

    def add_to_memory(self, *args):
        self.memory.push(*args)

    def save_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


