from agent import Agent
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym


class DQN_MiniMax_Policy(nn.Module): # Given a state, returns the best value possible
    def __init__(self, input_dim, output_dim):
        super(DQN_MiniMax_Policy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, obs):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def from_q_agent_policy(env, actions):
        return np.argmax(env.q_table, env.obs["state"])
    
class DNN_MiniMax_Agent(Agent):
    def __init__(self, name="DQN_Agent", board_size=7, from_q_agent_policy=None):
        super().__init__(name, "q_agent", from_q_agent_policy)
        self.policy = DQN_MiniMax_Policy(3, 1)
        self.env = gym.make('gymnasium_env/Blokus-v0', board_size=board_size, num_players=2, render_mode='human', render_scale=10, disable_env_checker=True, neighborhood_dir="/Users/mario/Documents/proj/cam/Blokus/gymnasium_env/envs/auxiliary/pre_neighbors", mode = "good")
        self.env = self.env.unwrapped
        self.env.order_enforce = False

    def get_action(self, env, obs) -> int:
        actions = obs["possible_actions"]
        state = env.capture_state()
        self.env.restore_state(state)
        best_value, best_action = -np.inf, None
        for action in actions:
            new_obs, reward, term, trun, info = self.env.step(action)
            action_value = self.policy(new_obs)
            if new_obs["player_turn"] != obs["player_turn"]: ## if no longer my turn
                action_value = -action_value
            if action_value + reward > best_value:
                best_value = action_value + reward
                best_action = action
        return best_action