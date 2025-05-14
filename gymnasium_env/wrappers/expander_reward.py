import gymnasium as gym
import numpy as np
from gymnasium_env.envs import BlokusAction, BlokusEnv
# from gymnasium_env.envs.single_agent_blokus_env import SingleAgentBlokusEnv

class ExpanderRewardWrapper(gym.Wrapper):
    def __init__(self, env: BlokusEnv, possible_action_diff_weight = 0.1):
        super().__init__(env)
        self.possible_action_diff_weight = possible_action_diff_weight

    def step(self: BlokusEnv, action):
        current_player = self.env.current_player
        actions = self.env.possible_actions(current_player)
        his_actions = self.env.possible_actions(1 - current_player)
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_actions = self.env.possible_actions(current_player)
        new_his_actions = self.env.possible_actions(1 - current_player)
        reward += (
            self.possible_action_diff_weight * (
                (len(new_actions) - len(actions))
                - (len(new_his_actions) - len(his_actions))
            )
        )
        score = obs["points"][current_player] - obs["points"][3 - current_player]
        if terminated:
            if score > 0:
                reward += 20
            elif score < 0:
                reward -= 20
            else:
                reward += 5
        return obs, reward, terminated, truncated, info
