from gymnasium_env.envs import blokus_env
from gymnasium_env.envs.blokus_env import *
from gymnasium_env.envs.single_agent_blokus_env import SingleAgentBlokusEnv
from gymnasium_env.wrappers import ExpanderRewardWrapper, MultipleColorsEncoding
from gymnasium.envs.registration import register

register(
    id="gymnasium_env/Blokus-v0",
    entry_point="gymnasium_env.envs:BlokusEnv",
)

from gymnasium_env.envs import *