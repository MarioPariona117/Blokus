import unittest
import gymnasium as gym
import gymnasium_env
import random
from tqdm import tqdm

class TestBlokusEnvPossibleActions(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('gymnasium_env/Blokus-v0', board_size=6, num_players=2, render_mode='human', render_scale=10, disable_env_checker=True, neighborhood="/Users/mario/Documents/proj/cam/Blokus/gymnasium_env/envs/auxiliary/pre_neighbors")
        self.env = self.env.unwrapped
        self.env.order_enforce = False
        self.env.reset()

    def test_something(self):
        self.env.reset()
        self.env.step(self.env._tuple_to_action((0, 0, 'I3', 0)))
        assert not self.env.place_piece((0, 2, 'L5', 3), 2, place=False)

    def test_possible_actions_efficient(self):
        player = 1
        actions = self.env.possible_actions_efficient(player)
        self.assertIsInstance(actions, list)
        self.assertTrue(all(isinstance(action, int) for action in actions))

    def test_possible_actions_inefficient(self):
        player = 1
        actions = self.env.possible_actions_inefficient(player)
        self.assertIsInstance(actions, list)
        self.assertTrue(all(isinstance(action, int) for action in actions))

    def _test_possible_actions_efficient_non_empty(self, i):
        player = 1
        actions = self.env.possible_actions_efficient(player)
        self.assertGreater(len(actions), 0)
    
    def test_possible_actions_hey(self):
        pass

    def test_possible_actions_inefficient_non_empty(self):
        player = 1
        actions = self.env.possible_actions_inefficient(player)
        self.assertGreater(len(actions), 0)

    def test_possible_actions_efficient_validity(self):
        player = 1
        actions = self.env.possible_actions_efficient(player)
        for action in actions:
            self.assertTrue(self.env.place_piece(self.env._action_to_tuple(action), player, place=False))
    
    def test_actions(self):
        A, B, C, D = 0, 0, 0, 0
        repetitions = 100
        for i in tqdm(range(repetitions)):
            self.env.reset()
            done = [False, False]
            # print(f"Finished {i}th game")
            agents = [0, 0]
            turn = 0
            while True:
                actions = self.env.possible_actions(self.env.current_player)
                # print(actions)
                if len(actions) == 0:
                    done[turn] = True
                else:
                    action = random.choice(actions)
                    (x, y, piece_id, transformation) = self.env._action_to_tuple(action)
                    agents[turn] += len(self.env.pieces[piece_id].transformations[0].shape)
                    # self.env.render()
                    assert self.env.place_piece(action_parameters=(x, y, piece_id, transformation), player=turn + 1)
                    self.env.current_player = (self.env.current_player) % self.env.num_players + 1
                if not done[1 - turn]:
                    turn = 1 - turn
            
                elif done[turn]:
                    break
            # if i == 0:
            #     # print("asdas")
                # env.render()
            else:
                print(i)
            A += bool (agents[0] > agents[1])
            D += bool (agents[0] < agents[1])
            B += agents[0]
            C += agents[1]
        print("Winning rate", A, repetitions - A - D, D)
        print("average: ", B/repetitions, C/repetitions)
    def test_possible_actions_inefficient_validity(self):
        player = 1
        actions = self.env.possible_actions_inefficient(player)
        for action in actions:
            self.assertTrue(self.env.place_piece(self.env._action_to_tuple(action), player, place=False))

    # def _test_possible_actions_efficient_equivalence_non_empty(self, i):
    #     player = 2
    #     self.env.step(i)
    #     actions_efficient = self.env.possible_actions_efficient(player)
    #     actions_inefficient = self.env.possible_actions_inefficient(player)
    #     try:
    #         self.assertEqual(actions_efficient, actions_inefficient)
    #     except AssertionError:
    #         print(f"Assertion failed at step {i}")
    #         raise

    # def test_possible_actions_efficient_equivalence_validity(self):
    #     for i in range(1000):
    #         self._test_possible_actions_efficient_equivalence_non_empty(i)
    #         actions_efficient = self.env.possible_actions_efficient(1)
    #         actions_inefficient = self.env.possible_actions_inefficient(1)
    #         self.assertEqual(actions_efficient, actions_inefficient)

    def _test_zero_actions_implementation_equivalence_validity(self, i):
        player = 1
        self.env.step(i)
        actions_efficient = self.env.possible_actions_efficient(player)
        actions_inefficient = self.env.possible_actions_inefficient(player)
        try:
            self.assertEqual(actions_efficient, actions_inefficient)
        except AssertionError:
            print(f"Assertion failed at step {i}")
            raise

    def _test_one_actions_implementation_equivalence_validity(self, i):
        player = 2
        self.env.step(i)
        actions_efficient = self.env.possible_actions_efficient(player)
        actions_inefficient = self.env.possible_actions_inefficient(player)
        try:
            self.assertEqual(actions_efficient, actions_inefficient)
        except AssertionError:
            print(f"Assertion failed at step {i}")
            raise

        for j in actions_efficient:
            self._test_zero_actions_implementation_equivalence_validity(j)

    def test_two_actions_implementation_equivalence_validity(self):
        for i in range(0):
            self._test_one_actions_implementation_equivalence_validity(i)
            print("passed ", i)  

class TestBlokusEnvAgentTurns(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('gymnasium_env/Blokus-v0', board_size=6, num_players=2, render_mode='human', render_scale=10, disable_env_checker=True)
        self.env = self.env.unwrapped
        self.env.order_enforce = False
        self.env.reset()

    # def test_agent_turns(self):
    #     for i in range(1000):
    #         self.assertEqual(self.env.current_player, i % self.env.num_players + 1)
    #         actions = self.env.possible_actions_efficient(self.env.current_player)
    #         if not actions:
    #             self.env.step(0)
    #             print(i)
    #             break
    #         obs, reward, terminated, truncated, info = self.env.step(random.choice(actions))
    #         if terminated or truncated:
    #             print(i)
    #             break

if __name__ == '__main__':
    unittest.main()
