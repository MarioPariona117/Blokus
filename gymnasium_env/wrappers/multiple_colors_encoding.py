import gymnasium as gym
import numpy as np
from gymnasium_env.envs import BlokusAction, BlokusEnv, BlokusPiece, PIECE_ORDER

import torch

class MultipleColorsEncoding(gym.Wrapper):
    def __init__(self, env: BlokusEnv):
        self.env = env
        self.channels = 21
        self.total_channels = self.channels * self.env.num_players
        self.encoding = np.zeros((self.total_channels, self.env.board_size, self.env.board_size))
        
    def __getattr__(self, attr):
        # This ensures that any missing attribute will be forwarded to the original environment (self.env)
        return getattr(self.env, attr)
    
    def step(self, action_id: int):
        action = BlokusAction(board_size=self.env.board_size, action_id=action_id)

        for cells in action.piece.body:
            self.encoding[(self.env.current_player - 1) * self.channels + PIECE_ORDER[action.piece.shape_id], cells[0], cells[1]] = 1

        obs, reward, term, trunc, info = self.env.step(action_id)
        obs["encoding"] = torch.tensor(self.encoding, dtype=torch.float32)
        return obs, reward, term, trunc, info
    
    def reset(self):
        self.encoding = np.zeros((self.total_channels, self.env.board_size, self.env.board_size))
        obs, info = self.env.reset()
        obs["encoding"] = torch.tensor(self.encoding, dtype=torch.float32)
        return obs, info
    
    