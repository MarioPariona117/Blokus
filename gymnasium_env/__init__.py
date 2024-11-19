from gymnasium_env.envs import blokus_env
from gymnasium_env.envs.blokus_env import BlokusEnv

from gymnasium.envs.registration import register

register(
    id="gymnasium_env/Blokus-v0",
    entry_point="gymnasium_env.envs:BlokusEnv",
)

from gymnasium_env.envs import blokus_piece