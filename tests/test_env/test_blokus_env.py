import pytest
import gymnasium as gym
import gymnasium_env
import random
from tqdm import tqdm
import numpy as np
from deepdiff import DeepDiff

from gymnasium_env.envs.blokus_env import BlokusEnv, BlokusAction, Vector2, BlokusPieceManager


class TestBlokusEnv:
    @pytest.fixture(autouse=True)
    def setup_env(self):
        BOARD_SIZE = 6
        self.env: BlokusEnv = gym.make('gymnasium_env/Blokus-v0', board_size=BOARD_SIZE, disable_env_checker=True)
        self.env = self.env.unwrapped
        self.env.order_enforce = False
        self.env.reset()
        yield
        self.env.close()

    def test_invalid_piece_placement(self):
        self.env.reset()
        action_1 = BlokusAction(board_size=self.env.board_size, action_tuple=(0, 0, BlokusPieceManager.get_piece(shape_id='I3', transform=0)))
        self.env.step(action_1.action_id)
        action_2 = BlokusAction(board_size=self.env.board_size, action_tuple=(0, 2, BlokusPieceManager.get_piece(shape_id='L5', transform=3)))
        assert not self.env.place_piece(action_2, 2, place=False)

    def test_valid_piece_placement(self):
        self.env.reset()
        action_1 = BlokusAction(board_size=self.env.board_size, action_tuple=(0, 0, BlokusPieceManager.get_piece(shape_id='I3', transform=0)))
        self.env.step(action_1.action_id)
        action_2 = BlokusAction(board_size=self.env.board_size, action_tuple=(2, 4, BlokusPieceManager.get_piece(shape_id='L5', transform=3)))
        assert self.env.place_piece(action_2, 2, place=False)

    def test_possible_actions_efficient(self):
        player = 1
        actions = self.env.possible_actions_efficient(player)
        assert isinstance(actions, np.ndarray)
        assert actions.dtype == np.uint16

    def test_possible_actions_inefficient(self):
        player = 1
        actions = self.env.possible_actions_inefficient(player)
        assert isinstance(actions, np.ndarray)
        assert actions.dtype == np.uint16

    def test_possible_actions_efficient_non_empty(self):
        obs, info = self.env.reset()
        player = 1
        actions = self.env.possible_actions_efficient(player)
        assert isinstance(actions, np.ndarray)
        assert isinstance(obs["possible_actions"], np.ndarray)
        assert np.array_equal(actions, obs["possible_actions"])
        if self.env.board_size > 4:
            assert len(actions) == 58

    def test_possible_actions_efficient_validity(self):
        self.env.reset()
        player = 1
        action_ids = self.env.possible_actions_efficient(player)
        actions = [BlokusAction(board_size=self.env.board_size, action_id=action_id) for action_id in action_ids]
        for action in actions:
            assert self.env.place_piece(action, player, place=False)

    def test_actions(self, repetitions=100):
        for i in tqdm(range(repetitions)):
            obs, info = self.env.reset()
            while True:
                action_ids = obs["possible_actions"]
                assert len(action_ids) > 0
                action_id = random.choice(action_ids)
                action = BlokusAction(board_size=self.env.board_size, action_id=action_id)
                obs, reward, term, trunc, info = self.env.step(action.action_id)
                assert reward == action.piece.size
                assert not trunc
                if term:
                    break

    def test_possible_actions_inefficient_validity(self):
        self.env.reset()
        player = 1
        action_ids = self.env.possible_actions_inefficient(player)
        actions = [BlokusAction(board_size=self.env.board_size, action_id=action_id) for action_id in action_ids]
        for action in actions:
            assert self.env.place_piece(action, player, place=False)

    def test_three_implementations(self, num_episodes=100):
        obs, info = self.env.reset()
        for _ in range(num_episodes):
            player = self.env.current_player
            actions_1 = np.sort(self.env.possible_actions_inefficient(player))
            actions_2 = np.sort(self.env.possible_actions_efficient(player))
            actions_3 = np.sort(self.env.possible_actions_precomputed(player))
            actions = np.sort(obs["possible_actions"])
            assert np.array_equal(actions_1, actions_2)
            assert np.array_equal(actions_2, actions_3)
            assert np.array_equal(actions_3, actions)
            action_id = random.choice(actions)
            obs, _, term, _, _ = self.env.step(action_id)
            if term:
                obs, info = self.env.reset()

    def test_reset(self):
        observation, info = self.env.reset()
        assert observation["num_players"] == self.env.num_players
        assert observation["state"].shape == (self.env.board_size, self.env.board_size)
        assert observation["current_player"] == 1
        assert observation["steps"] == 0

    def test_step(self):
        self.env.reset()
        action = BlokusAction(board_size=self.env.board_size, action_tuple=(0, 0, BlokusPieceManager.get_piece(shape_id='I3', transform=0)))
        observation, reward, terminated, truncated, info = self.env.step(action.action_id)
        assert reward > 0
        assert not terminated
        assert not truncated
        assert observation["current_player"] == 2

    def test_capture_restore_state(self):
        self.env.reset()
        state = self.env.capture_state()
        action = BlokusAction(board_size=self.env.board_size, action_tuple=(0, 0, BlokusPieceManager.get_piece(shape_id='I3', transform=0)))
        self.env.step(action.action_id)
        self.env.restore_state(state)
        restored_state = self.env.capture_state()
        assert not DeepDiff(state, restored_state)

    def test_legal_cell(self):
        self.env.reset()
        assert self.env.legal_cell((0, 0), self.env.current_player)
        assert not self.env.legal_cell((self.env.board_size, self.env.board_size), self.env.current_player)

    def test_get_neighborhood(self):
        self.env.reset()
        expander = Vector2(0, 0)
        neighborhood = self.env.get_neighborhood(expander, self.env.current_player)
        assert isinstance(neighborhood, int)

    def test_get_locked_squares(self):
        self.env.reset()
        locked_squares = self.env.get_locked_squares(self.env.current_player)
        assert isinstance(locked_squares, set)

    def test_get_expander_squares(self):
        self.env.reset()
        expander_squares = self.env.get_expander_squares(self.env.current_player)
        assert isinstance(expander_squares, dict)

    def test_render(self):
        self.env.reset()
        self.env.render()
