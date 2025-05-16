import gymnasium as gym
import numpy as np
import random
import pickle
import os

from gymnasium_env import BlokusEnv

from ..utils import encode_board_string, decode_board_string
class Agent:
    def __init__(self, name="Agent", get_action=None, seed=None):
        self.name = name
        self._get_action = get_action
        self.rng = random.Random(seed)
        print(f"Agent {self.name} initialized with seed {seed}")
        self.load_cache()

    def get_action(self, env: BlokusEnv, obs):
        return self._get_action(env, obs)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def save_cache(self):
        return
    
    def load_cache(self):
        return
    
    def close(self):
        self.save_cache()
        return