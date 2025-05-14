from gymnasium_env.envs.single_agent_blokus_env import BlokusEnv, SingleAgentBlokusEnv
from tqdm import tqdm
import pickle
from typing import Tuple

class BlokusGameSimulator:
    def __init__(self, board_size, num_players, wrappers=None, log_dir=None, render_mode="console"):
        self.board_size = board_size   
        self.num_players = num_players
        self.env = BlokusEnv(
            board_size = board_size,
            num_players = num_players,
            render_mode=render_mode,
        )
        self.wrapped_env = self.env
        if wrappers is not None:
            for wrapper in wrappers:
                self.wrapped_env = wrapper(self.wrapped_env)
        self.testing_env = SingleAgentBlokusEnv(
            base_env = self.wrapped_env,
            hidden_agents = [None] * (num_players),
            player_turn = 1
        )
        self.log_dir = log_dir
        self.log_dict = {}
        self.load_dict()

    def load_dict(self):
        if self.log_dir is not None:
            try:
                with open(self.log_dir, "rb") as f:
                    self.log_dict = pickle.load(f)
            except Exception as e:
                print(f"Error loading log dictionary: {e}")
                self.log_dict = {}
        else:
            self.log_dict = {}
    
    def save_dict(self):
        if self.log_dir is not None:
            try:
                with open(self.log_dir, "wb") as f:
                    pickle.dump(self.log_dict, f)
            except Exception as e:
                print(f"Error saving log dictionary: {e}")
                raise e
    
    def _do_logic(self, agent1, agent2, testing_agent_id, num_episodes=100, pbar=False):
        if testing_agent_id is not None:
            switch = 1 if testing_agent_id == 1 else -1
            win_counter, tie_counter, lose_counter = 0, 0, 0
            log_dict = {
                "win_rate": 0, "tie_rate": 0, "lose_rate": 0,
                "diff_points": []
            }
        else:
            log_dict = {
                "p1_counter": 0, "p2_counter": 0, "tie_counter": 0,
                "points": [], "steps": []
            }
        try:
            agent1.eval()
            agent2.eval()
            self.testing_env.hidden_agents[2] = agent2

            prange = range(num_episodes)

            if pbar:
                prange = tqdm(prange, desc="Episodes")

            for i in prange:
                if i and (i % 100 == 0) and pbar:
                    prange.set_description(f"E {i}: W: {win_counter / (i) * 100:.2f}%, T: {tie_counter / (i) * 100:.2f}%, L: {lose_counter / (i) * 100:.2f}%")
                obs, info = self.testing_env.reset()
                while True:
                    action = agent1.get_action(self.testing_env, obs)
                    obs, reward, terminated, truncated, info = self.testing_env.step(action)
                    if terminated:
                        break
                    if truncated:
                        raise Exception("Truncated")
                    
                diff = obs["points"][1] - obs["points"][2]
                if testing_agent_id is not None:
                    real_diff = diff * switch
                    log_dict["diff_points"].append(real_diff)
                    win_counter += real_diff > 0
                    tie_counter += real_diff == 0
                    lose_counter += real_diff < 0
                else:
                    log_dict["p1_counter"] += diff > 0
                    log_dict["tie_counter"] += diff == 0
                    log_dict["p2_counter"] += diff < 0
                    log_dict["points"].append(obs["points"])
                    log_dict["steps"].append(obs["steps"])

        except Exception as e:
            print(f"Error during testing: {e}")
            print(obs["points"])
            assert False
        finally:
            agent1.train()
            agent2.train()
            try:
                agent1.save_cache()
                agent2.save_cache()
            except Exception as e:
                print(e)
                raise e
        if testing_agent_id is not None:
            win_rate = win_counter / num_episodes * 100
            tie_rate = tie_counter / num_episodes * 100
            lose_rate = lose_counter / num_episodes * 100
            log_dict["win_rate"] = win_rate
            log_dict["tie_rate"] = tie_rate
            log_dict["lose_rate"] = lose_rate
        return log_dict
    
    def play(self, agent1, agent2, num_episodes=300, pbar=False):
        return self._do_logic(agent1, agent2, None, num_episodes, pbar)
    
    def test(self, agent1, agent2, testing_agent_id, num_episodes=300, pbar=False) -> Tuple[float, float, float]:
        """Win rate, tie rate, lose rate"""
        assert testing_agent_id in list(range(1, self.num_players + 1)), f"Testing agent id must be in the range [1 - {self.num_players}]"
        return self._do_logic(agent1, agent2, testing_agent_id, num_episodes, pbar)
