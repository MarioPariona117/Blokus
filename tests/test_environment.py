import unittest
from src.environment.blokus_env import BlokusEnv
import random

class TestBlokusEnvironment(unittest.TestCase):

    def setUp(self):
        # self.env = BlokusEnv(render_mode='console')
        self.env = BlokusEnv(num_agents=1, render_mode='human', render_scale=10)

    def test_openai_gymanasium_compaitibility(self):
        
        pass
    def test_initial_state(self):
        # Test the initial state of the environment
        self.assertEqual(self.env.current_player, 1)
        self.assertEqual(len(self.env.board), 14)
        self.assertEqual(len(self.env.board[0]), 14)

    def test_place_piece(self):
        # Test placing a piece on the board
        piece_id = 'P'
        transformation = 0 # No transformation
        self.env.step((0, 0, piece_id, transformation))
        # self.assertEqual(self.env.board[0][0], 1)
        # self.assertEqual(self.env.board[0][1], 0)
        # self.assertEqual(self.env.board[1][0], 0)
        # self.assertEqual(self.env.board[1][1], 0)
        # self.env.render()
        pass

    def test_invalid_move(self):
        # Test placing a piece in an invalid position
        # piece = [[1, 1], [1, 1]]
        # position = (19, 19)
        # with self.assertRaises(ValueError):
        #     self.env.place_piece(piece, position)
        pass

    def test_locked_squares(self):
        # self.env.reset()
        # self.assertEqual(self.env.locked_squares(1), [(i, j) for i in range(20) for j in range(20)])
        pass

    def test_switch_player(self):
        # Test switching the current player
        # self.env.switch_player()
        # self.assertEqual(self.env.current_player, 1)
        # self.env.switch_player()
        # self.assertEqual(self.env.current_player, 2)
        pass

    def test_game_over(self):
        # Test the game over condition
        # self.env.board = [[1] * 20 for _ in range(20)]
        # self.assertTrue(self.env.is_game_over())
        pass

if __name__ == '__main__':
    unittest.main()