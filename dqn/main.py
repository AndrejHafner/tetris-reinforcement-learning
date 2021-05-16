import time
from itertools import count

import time
import numpy as np
import torch

from dqn.agent import Agent
from tetris.tetris import Tetris

if __name__ == '__main__':

    num_episodes = 3000
    state_size = 5
    num_actions = 5
    TARGET_UPDATE = 1
    batch_size = 512
    gamma = 0.95
    train_every_n = 1
    draw_every = 1
    eps_decay = 250
    eps_start = 0.99
    eps_end = 0.01
    lr = 1e-3
    train = True
    policy_net_path = None # "./dqn_checkpoint_mse_almost_perfect.pth"

    # Get the Pytorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Initialize the environment
    env = Tetris(16, 30)
    env.init_env()
    loss = -1
    best_episode_reward = 0

    # Initialize the agent
    agent = Agent(state_size, device, gamma=gamma, policy_net_path=policy_net_path, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, lr=lr)
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset_env()
        state = torch.zeros(state_size, device=device)
        # agent.reset()
        episode_losses = []
        reward_sum = 0

        draw_sim = i_episode % draw_every == 0
        episode_start = time.time()
        for step in count():
            # Select and perform an action
            state_action_pairs = env.get_next_states()

            action = agent.select_action(state_action_pairs, i_episode)
            next_state, reward, done = env.step(*action, i_episode, loss, reward_sum, draw_game=draw_sim)
            reward = torch.tensor([reward], device=device)
            reward_sum += reward.item()

            # Store the transition in memory
            agent.add_to_memory(state, torch.tensor(action, device=device), torch.tensor(next_state, device=device), reward)

            # Move to the next state
            state = torch.tensor(next_state, device=device, dtype=torch.double)

            # Perform one step of the optimization (on the policy network)



            if done:
                print("Stopping episode, next!")
                break
        if i_episode % train_every_n == 0 and train:
            loss = agent.optimize(batch_size, i_episode)

        loss = round(loss if loss != None else -1, 3)
        print("Episode loss:", loss)
        print("Episode runtime: ", time.time() - episode_start)
        print("Total episode reward: ", reward_sum)

        if reward_sum > best_episode_reward:
            best_episode_reward = reward_sum
            agent.save_model(f"./trained_models/dqn_model_best_reward_{int(best_episode_reward)}.pth")

        print("Best episode reward: ", best_episode_reward)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            agent.update_target_net()
