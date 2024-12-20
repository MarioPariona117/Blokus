import gymnasium as gym
import random
import numpy as np
import pickle
import os
from ..utils import encode_board_string, decode_board_string

def maximize_policy(action_to_value, env, actions):
    if not actions:
        return None
    return max(actions, key=lambda action: action_to_value(action))

class Agent:
    # policy_map = {
    #     "random": random_policy,
    #     "maximize": maximize_policy,
    #     "q_agent": from_q_agent_policy
    # }
    def __init__(self, name="Agent", policy="random", simple_policy=None):
        self.name = name
        # self.policy = self.policy_map[policy]  # Function to determine the agent's actions
        # self.simple_policy = simple_policy  # Function to determine the agent's actions if the policy is "random"
    
    def get_action(self, env, obs):
        actions = obs["possible_actions"]
        next_obss = [env.get_next_obs(env, action) for action in actions]
        return self.choose_action(obs, actions, next_obss)
    
    def choose_action(self, obs, actions, next_obss):
        action = self.policy(obs, actions, next_obss)
        return action
