import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple


class BlokusEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, board_size=14, num_agents=2):
        super(BlokusEnv, self).__init__()
        self.board_size = board_size
        self.num_agents = num_agents
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size), dtype=np.int8)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        return self.board, {}

    def step(self, action: Tuple[(int, int), ]):
        pass
        # return self.board, reward, done, {}, {}

    def render(self, mode='human'):
        for row in self.board:
            print(' '.join(str(x) for x in row))
        print()

    def close(self):
        pass

# Register the environment with Gymnasium
gym.envs.registration.register(
    id='Blokus-v0',
    entry_point='blokus_env:BlokusEnv',
)

if __name__ == "__main__":
    env = BlokusEnv()
    env.reset()
    env.render()