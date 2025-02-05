import pytest
import gymnasium as gym
import gymnasium_env
import random
from tqdm import tqdm
import numpy as np
from deepdiff import DeepDiff

from gymnasium_env.envs.blokus_env import BlokusEnv, BlokusAction, Vector2, BlokusPieceManager

@pytest.fixture
def env():
    BOARD_SIZE = 6
    env: BlokusEnv = gym.make('gymnasium_env/Blokus-v0', board_size=BOARD_SIZE, disable_env_checker=True)
    env = env.unwrapped
    env.order_enforce = False
    env.reset()
    yield env
    env.close()

def test_invalid_piece_placement(env):
    env.reset()
    action_1 = BlokusAction(board_size=env.board_size, action_tuple=(0, 0, BlokusPieceManager.get_piece(shape_id='I3', transform=0)))
    env.step(action_1.action_id)
    action_2 = BlokusAction(board_size=env.board_size, action_tuple=(0, 2, BlokusPieceManager.get_piece(shape_id='L5', transform=3)))
    assert not env.place_piece(action_2, 2, place=False)

def test_valid_piece_placement(env):
    env.reset()
    action_1 = BlokusAction(board_size=env.board_size, action_tuple=(0, 0, BlokusPieceManager.get_piece(shape_id='I3', transform=0)))
    env.step(action_1.action_id)
    action_2 = BlokusAction(board_size=env.board_size, action_tuple=(2, 4, BlokusPieceManager.get_piece(shape_id='L5', transform=3)))
    assert env.place_piece(action_2, 2, place=False)

def test_possible_actions_efficient(env):
    player = 1
    actions = env.possible_actions_efficient(player)
    assert isinstance(actions, np.ndarray)
    assert actions.dtype == np.uint16

def test_possible_actions_inefficient(env):
    player = 1
    actions = env.possible_actions_inefficient(player)
    assert isinstance(actions, np.ndarray)
    assert actions.dtype == np.uint16

def test_possible_actions_efficient_non_empty(env):
    obs, info = env.reset()
    player = 1
    actions = env.possible_actions_efficient(player)
    assert isinstance(actions, np.ndarray)
    assert isinstance(obs["possible_actions"], np.ndarray)
    assert np.array_equal(actions, obs["possible_actions"])
    if env.board_size > 4:
        assert len(actions) == 58

def test_possible_actions_efficient_validity(env):
    env.reset()
    player = 1
    action_ids = env.possible_actions_efficient(player)
    actions = [BlokusAction(board_size=env.board_size, action_id=action_id) for action_id in action_ids]
    for action in actions:
        assert env.place_piece(action, player, place=False)

def test_actions(env, repetitions=100):
    for i in tqdm(range(repetitions)):
        obs, info = env.reset()
        while True:
            action_ids = obs["possible_actions"]
            assert len(action_ids) > 0
            action_id = random.choice(action_ids)
            action = BlokusAction(board_size=env.board_size, action_id=action_id)
            obs, reward, term, trunc, info = env.step(action.action_id)
            assert reward == action.piece.size
            assert not trunc
            if term:
                break

def test_possible_actions_inefficient_validity(env):
    env.reset()
    player = 1
    action_ids = env.possible_actions_inefficient(player)
    actions = [BlokusAction(board_size=env.board_size, action_id=action_id) for action_id in action_ids]
    for action in actions:
        assert env.place_piece(action, player, place=False)

def test_three_implementations(env, num_episodes=100):
    obs, info = env.reset()
    for _ in range(num_episodes):
        player = env.current_player
        actions_1 = np.sort(env.possible_actions_inefficient(player))
        actions_2 = np.sort(env.possible_actions_efficient(player))
        actions_3 = np.sort(env.possible_actions_precomputed(player))
        actions = np.sort(obs["possible_actions"])
        assert np.array_equal(actions_1, actions_2)
        assert np.array_equal(actions_2, actions_3)
        assert np.array_equal(actions_3, actions)
        action_id = random.choice(actions)
        obs, _, term, _, _ = env.step(action_id)
        if term:
            obs, info = env.reset()

def test_reset(env):
    observation, info = env.reset()
    assert observation["n_players"] == env.num_players
    assert observation["state"].shape == (env.board_size, env.board_size)
    assert observation["current_player"] == 1
    assert observation["steps"] == 0

def test_step(env):
    env.reset()
    action = BlokusAction(board_size=env.board_size, action_tuple=(0, 0, BlokusPieceManager.get_piece(shape_id='I3', transform=0)))
    observation, reward, terminated, truncated, info = env.step(action.action_id)
    assert reward > 0
    assert not terminated
    assert not truncated
    assert observation["current_player"] == 2

def test_capture_restore_state(env):
    env.reset()
    state = env.capture_state()
    action = BlokusAction(board_size=env.board_size, action_tuple=(0, 0, BlokusPieceManager.get_piece(shape_id='I3', transform=0)))
    env.step(action.action_id)
    env.restore_state(state)
    restored_state = env.capture_state()
    assert not DeepDiff(state, restored_state)

def test_legal_cell(env):
    env.reset()
    assert env.legal_cell((0, 0), env.current_player)
    assert not env.legal_cell((env.board_size, env.board_size), env.current_player)

def test_get_neighborhood(env):
    env.reset()
    expander = Vector2(0, 0)
    neighborhood = env.get_neighborhood(expander, env.current_player)
    assert isinstance(neighborhood, int)

def test_get_locked_squares(env):
    env.reset()
    locked_squares = env.get_locked_squares(env.current_player)
    assert isinstance(locked_squares, set)

def test_get_expander_squares(env):
    env.reset()
    expander_squares = env.get_expander_squares(env.current_player)
    assert isinstance(expander_squares, dict)

def test_render(env):
    env.reset()
    env.render()

@pytest.fixture
def env_agent_turns():
    env = gym.make('gymnasium_env/Blokus-v0', board_size=6, num_players=2, render_mode='human', render_scale=10, disable_env_checker=True)
    env = env.unwrapped
    env.order_enforce = False
    env.reset()
    yield env
    env.close()

# def test_agent_turns(env_agent_turns):
#     for i in range(1000):
#         assert env_agent_turns.current_player == i % env_agent_turns.num_players + 1
#         actions = env_agent_turns.possible_actions_efficient(env_agent_turns.current_player)
#         if not actions:
#             env_agent_turns.step(0)
#             print(i)
#             break
#         obs, reward, terminated, truncated, info = env_agent_turns.step(random.choice(actions))
#         if terminated or truncated:
#             print(i)
#             break
