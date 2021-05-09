import time
from itertools import count

import numpy as np
import torch

from dqn.agent import Agent
from tetris.tetris import Tetris

if __name__ == '__main__':

    num_episodes = 1000
    state_size = 4
    num_actions = 5
    TARGET_UPDATE = 10
    batch_size = 512
    gamma = 0.95
    train_every_n = 1

    # Get the Pytorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Initialize the environment
    env = Tetris(16, 30)
    env.init_env()

    # Initialize the agent
    agent = Agent(state_size, num_actions, device, gamma=gamma)

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset_env()
        state = torch.zeros(state_size, device=device)
        # agent.reset()
        episode_losses = []

        for step in count():
            # Select and perform an action
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            agent.add_to_memory(state, torch.tensor(action, device=device), torch.tensor(next_state, device=device), reward)

            # Move to the next state
            state = torch.tensor(next_state, device=device, dtype=torch.double)

            # Perform one step of the optimization (on the policy network)

            if step % train_every_n == 0:
                loss = agent.optimize(batch_size)

                if loss != None:
                    episode_losses.append(loss)

            if done:
                print("Stopping episode, next!")
                break
        print("Average episode loss:", np.average(episode_losses))
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            agent.update_target_net()
