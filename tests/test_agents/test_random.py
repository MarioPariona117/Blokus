from src.agents import RandomAgent
from gymnasium_env import BlokusEnv

import pytest

@pytest.fixture
def blokus_env():
    env = BlokusEnv(board_size=6, num_players=2, testing_mode=False)
    yield env
    env.close()

@pytest.fixture
def random_agent():
    agent = RandomAgent()
    yield agent
    agent.close()

def test_random_agent_initialization(agent):
    assert agent is not None, "Agent should be initialized successfully"

def test_random_agent_get_action(blokus_env, random_agent):
    blokus_env.reset()
    obs = blokus_env.get_observation()
    action = random_agent.get_action(blokus_env, obs)
    assert action in blokus_env.get_possible_actions(), "Action should be valid"
    assert isinstance(action, int), "Action should be an integer representing the action ID"