import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Any, Dict, List, SupportsInt
from gymnasium.core import ObsType, ActType
import pygame
import time

from gymnasium_env.envs.auxiliary.utils import decode, encode
from .blokus_piece import BlokusPieceTransformations, BlokusPiece, BlokusPieceManager, PIECE_SHAPE_IDS, Vector2
from .blokus_action import BlokusAction

class BlokusEnvAgentInfo:
    """Class to store information about a Blokus agent, including the player ID and available pieces."""
    def __init__(self, player_id, available_pieces: List[str]):
        self.points = 0
        self.player_id = player_id
        self.available_pieces = set(list(available_pieces))
        self.expander_squares = {}
        self.locked_squares = set()
        self.possible_actions = None
        self.started = False

class BlokusEnv(gym.Env):
    metadata = {'render_modes': ['human', 'console', 'rgb_array'], 'render_fps': 4}
    BAD_MOVE_PUNISHMENT = -100
    NEIGHBOR_POS = None
    BlokusPieceManager()
    # NEIGHBORHOOD_TO_ENCODED_ACTIONS = None
    initialization_total_time = 0
    # @classmethod
    def _load_neighborhood(neighborhood_dir):
        if BlokusEnv.NEIGHBOR_POS is None:
            BlokusEnv.NEIGHBORHOOD_TO_ENCODED_ACTIONS = np.load(
                f"{neighborhood_dir}/compute_actions.pkl", allow_pickle=True
            )
            print("Neighborhood to encoded actions loaded")
            BlokusEnv.NEIGHBOR_POS = np.load(
                f"{neighborhood_dir}/neighbor_pos.pkl", allow_pickle=True
            )
        else:
            print("Neighborhood already loaded")
            
    def __init__(
        self, 
        render_mode: str = 'console', 
        board_size: int = 14, 
        num_players: int = 2, 
        render_scale: int = 10, 
        neighborhood_dir: str = "/Users/mario/Documents/proj/cam/Blokus/gymnasium_env/envs/auxiliary/pre_neighbors", 
        testing_mode: bool = False
    ):
        """Initialize the Blokus environment with parameters for rendering, board size, number of players, and render scaling."""
        super(BlokusEnv, self).__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.freq = [0 for _ in range(130)]

        if neighborhood_dir:
            BlokusEnv._load_neighborhood(neighborhood_dir)
        
        ##################################### BOARD ######################################
        self.board_size = board_size

        ############################### AGENTS PARAMETERS ###############################
        self.num_players = num_players

        ############################## RENDERING PARAMETERS ##############################
        self.render_mode = render_mode
        self.render_scale = render_scale 

        ########################### ACTION AND OBSERVATION SPACES ######################################
        self.action_space = spaces.Discrete(self.board_size * self.board_size * BlokusPieceManager.PIECE_VARIANTS_COUNT)
        self.observation_space = spaces.Dict({
            "n_players": spaces.Discrete(5),
            "state": spaces.Box(
                low=0,
                high=self.num_players + 1,
                shape=(self.board_size, self.board_size),
                dtype=np.uint8
            ),
            "possible_actions": spaces.Sequence(self.action_space),
            "expander_squares": spaces.Tuple([
                spaces.Sequence(
                    spaces.Tuple((
                        spaces.Discrete(self.board_size), 
                        spaces.Discrete(self.board_size)
                    ))
                ) for _ in range(self.num_players)
            ]),
            "locked_squares": spaces.Tuple([
                spaces.Sequence(
                    spaces.Tuple((
                        spaces.Discrete(self.board_size), 
                        spaces.Discrete(self.board_size)
                    ))
                ) for _ in range(self.num_players)
            ]),
            "current_player": spaces.Discrete(self.num_players, start=1)
        })
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.testing_mode = testing_mode
        self.window = None
        self.clock = None
        mode = "testing" if self.testing_mode else "non-testing"
        print(f"Initialised on {mode} mode")
        self.reset()
        
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment to the initial state, preparing the board and player-specific data."""
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        # self.started = [False for _ in range(self.num_players + 1)]
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
        self.current_player = 1
        self.steps = 0
        self.agents_info = [BlokusEnvAgentInfo(player_id=i, available_pieces=PIECE_SHAPE_IDS) for i in range(0, self.num_players + 1)]

        if self.num_players == 1:
            self.agents_info[1].expander_squares = {Vector2(0, 0): Vector2(1, 1)}
        elif self.num_players == 2:
            self.agents_info[1].expander_squares = {Vector2(0, 0): Vector2(1, 1)}
            self.agents_info[2].expander_squares = {Vector2(self.board_size - 1, self.board_size - 1): Vector2(-1, -1)}
        else:
            raise ValueError("Only 1-2 player is supported for now :3")
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_obs(self):
        return {
            "n_players": self.num_players,  # Number of players in the game
            "state": np.array(self.board),  # 2D array of the current board state
            "possible_actions": np.array(self.possible_actions(self.current_player)),  # List of possible actions for the current player
            "expander_squares": [
                np.array(list(self.agents_info[i].expander_squares.keys())) for i in range(0, self.num_players + 1)
            ], # Expander squares for the current player
            "locked_squares": [
                np.array(list(self.agents_info[i].locked_squares)) for i in range(0, self.num_players + 1)
            ], # Locked squares for the current player
            "current_player": self.current_player, # ID of the current player
            "points": [self.agents_info[i].points for i in range(0, self.num_players + 1)],
            "steps": self.steps
        }
    def _get_info(self):
        return {
        }
    
    def capture_state(self):
        state = {
            "current_player": self.current_player,
            "points": [self.agents_info[i].points for i in range(0, self.num_players + 1)],
            "steps": self.steps
        }
        state["board"] = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
        for i in range(self.board_size):
            for j in range(self.board_size):
                state["board"][i, j] = self.board[i, j]

        state["agents_info"] = [{} for _ in range(self.num_players + 1)]
        for i in range(1, self.num_players + 1):
            state["agents_info"][i]["available_pieces"] = set(self.agents_info[i].available_pieces)
            state["agents_info"][i]["expander_squares"] = dict(self.agents_info[i].expander_squares)
            state["agents_info"][i]["locked_squares"] = set(self.agents_info[i].locked_squares)
            if self.agents_info[i].possible_actions is not None:
                state["agents_info"][i]["possible_actions"] = self.agents_info[i].possible_actions.copy()
            else:
                state["agents_info"][i]["possible_actions"] = None
            state["agents_info"][i]["started"] = self.agents_info[i].started
        
        return state

    def restore_state(self, state):
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.board[i, j] = state["board"][i, j]
        self.current_player = state["current_player"]
        self.steps = state["steps"]
        for i in range(1, self.num_players + 1):
            self.agents_info[i].player_id = i
            self.agents_info[i].available_pieces = set(state["agents_info"][i]["available_pieces"])
            self.agents_info[i].expander_squares = dict(state["agents_info"][i]["expander_squares"])
            self.agents_info[i].locked_squares = set(state["agents_info"][i]["locked_squares"])
            if state["agents_info"][i]["possible_actions"] is not None:
                self.agents_info[i].possible_actions = state["agents_info"][i]["possible_actions"].copy()
            else:
                self.agents_info[i].possible_actions = None
            self.agents_info[i].started = state["agents_info"][i]["started"]
            self.agents_info[i].points = state["points"][i]

    def legal_cell(self, pos, player):
        return (
            0 <= pos[0] < self.board_size and
            0 <= pos[1] < self.board_size and
            self.board[pos] == 0 and
            not pos in self.agents_info[player].locked_squares
        )
    
    @staticmethod
    def get_actions_from_neighborhood(neighborhood: int, available_pieces) -> List[Tuple[int, int, ActType]]:
        """
        Retrieve the list of possible actions from a neighborhood bitmask.
        Args:
            neighborhood (int): A bitmask representing the neighborhood of an expander square.
        Returns:
            List[int]: A list of possible actions encoded as integers.
        Notes:
            - The function uses a precomputed dictionary to map neighborhoods to encoded actions.
        """
        actions = BlokusEnv.NEIGHBORHOOD_TO_ENCODED_ACTIONS[neighborhood]
        actions = set(map(decode, actions))
        return actions

    def get_neighborhood(self, expander: Vector2, player: int) -> int:
        """
        Calculate the neighborhood bitmask for a given expander position on the board for a specific player.
        Args:
            expander (tuple): The coordinates (x, y) of the expander on the board.
            player (int): The player index for which the neighborhood is being calculated.
        Returns:
            int: A bitmask representing the neighborhood of the expander. Each bit corresponds to a neighboring position.
        Notes:
            - The function checks the legality of each neighboring cell around the expander.
            - The neighborhood bitmask is updated based on the legality of these cells.
            - The coordinates (x + dx * i, y + dy * j) represent the neighboring cell positions.
        """
        neighborhood = 0
        direction = self.agents_info[player].expander_squares[expander] # first direction
        if (not self.legal_cell((expander.x + direction.x, expander.y), player)) and (not self.legal_cell((expander.x, expander.y + direction.y), player)):
            return 0
        # # Option 1: Parallel computation
        # def check_neighbor(idx, i, j):
        #     neighbor = (x + dx * i, y + dy * j)
        #     return (1 << idx) if self.legal_cell(neighbor, player) else 0

        # neighborhood = sum(Parallel(n_jobs=-1)(delayed(check_neighbor)(idx, i, j) for idx, (i, j) in enumerate(BlokusEnv.NEIGHBOR_POS)))

        # Option 2: Sequential computation
        neighborhood = 0
        for idx, (i, j) in enumerate(BlokusEnv.NEIGHBOR_POS):
            neighbor = (expander.x + direction.x * i, expander.y + direction.y * j)
            if self.legal_cell(neighbor, player):
                neighborhood |= (1 << idx)
        return neighborhood
    
    def get_locked_squares(self, player):
        """Return locked squares for a player, where placement is restricted based on adjacent cells."""
        ret = set()
        for i in range(self.board_size):
            for j in range(self.board_size):
                add = False
                for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if min(i + x, j + y) >= 0 and max(i + x, j + y) < self.board_size and self.board[i + x, j + y] == player:
                        add = True
                if add and self.board[i, j] != player:
                    ret.add((i, j))
        return ret
    
    def get_expander_squares(self, player):
        """Return expander squares for a player, where placement is encouraged based on diagonal cells."""
        if not self.agents_info[player].started:
            if not self.legal_cell(list(self.agents_info[player].expander_squares.keys())[0], player):
                self.agents_info[player].expander_squares = {}
            else:
                if player == 1:
                    return {(0, 0):(1, 1)}
                elif player == 2:
                    return {(self.board_size - 1, self.board_size - 1):(-1, -1)}
                else:
                    raise ValueError("Only 1-2 player is supported for now :3")
        ret = {}
        for i in range(self.board_size):
            for j in range(self.board_size):
                direction, not_add = None, False
                for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    if min(i + x, j + y) >= 0 and max(i + x, j + y) < self.board_size and self.board[i + x, j + y] == player:
                        direction = (-x, -y)
                for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if min(i + x, j + y) >= 0 and max(i + x, j + y) < self.board_size and self.board[i + x, j + y] == player:
                        not_add = True
                if not not_add and direction and self.board[i, j] == 0:
                    ret[i, j] = direction
        return ret
    
    def place_piece(self, action: BlokusAction, player: int, place=True, check=True) -> bool:
        can_place = True
        if check:
            if not action.piece.shape_id in self.agents_info[player].available_pieces:
                print("Piece already used")
                return False #### PIECE ALREADY USED ####

            can_place = False
            for i, j in action.piece.body:
                board_row, board_col = i + action.x, j + action.y
                if not self.legal_cell((board_row, board_col), player):
                    print("Illegal cell")
                    print(f"Failed to place piece: {action.piece.shape_id} at ({action.x}, {action.y}) with transformation {action.piece.transform} for player {player}")
                    print(self.board)
                    assert False
                    return False
                if (board_row, board_col) in self.agents_info[player].expander_squares:
                    can_place = True #### TOUCHES A CORNER PIECE ####
        
        if not can_place:
            return False
        
        if not place:
            return True
        
        if BlokusEnv.NEIGHBOR_POS is not None:
            self.faster_update_attributes(player, action)
        else:
            self.update_attributes(player, action)
        return True

    def update_attributes(self, player, action: BlokusAction):
        for i in range(1, self.num_players + 1):
            self.agents_info[i].possible_actions = None

        for i, j in action.piece.body:
            self.board[i + action.x, j + action.y] = player
            
        self.agents_info[player].available_pieces.remove(action.piece.shape_id)
        self.agents_info[player].started = True

        for i in range(1, self.num_players + 1):
            self.agents_info[i].locked_squares = self.get_locked_squares(i)
            self.agents_info[i].expander_squares = self.get_expander_squares(i)
            
    def faster_update_attributes(self, player, action: BlokusAction):
        ### UPDATE OTHER ATTRIBUTES TO MAKE IT FASTER ###
        ### EXAMPLE: LOCKED/EXPANDER SQUARES ###
        for i in range(1, self.num_players + 1):
            self.agents_info[i].possible_actions = None

        assert 1 <= player <= self.num_players
        self.agents_info[player].available_pieces.remove(action.piece.shape_id)
        self.agents_info[player].started = True

        for i, j in action.piece.locked:
            self.agents_info[player].locked_squares.add(Vector2(action.x + i, action.y + j))
            if (action.x + i, action.y + j) in self.agents_info[player].expander_squares:
                self.agents_info[player].expander_squares.pop(Vector2(action.x + i, action.y + j))

        for (i, j), (dx, dy) in action.piece.expanders.items():
            if self.legal_cell((action.x + i, action.y + j), player): # add own expanders
                self.agents_info[player].expander_squares[Vector2(action.x + i, action.y + j)] = Vector2(dx, dy)
        
        for i, j in action.piece.body:
            self.board[i + action.x, j + action.y] = player
            if (i + action.x, j + action.y) in self.agents_info[player].expander_squares:
                self.agents_info[player].expander_squares.pop(Vector2(i + action.x, j + action.y))
            for k in range(1, self.num_players + 1):
                if k == player:
                    continue
                if (i + action.x, j + action.y) in self.agents_info[k].expander_squares:
                    self.agents_info[k].expander_squares.pop(Vector2(i + action.x, j + action.y))
        
        if self.testing_mode:
            raise ValueError("Testing mode")
            for i in range(1, self.num_players + 1):
                try:
                    assert list(sorted(self.agents_info[i].expander_squares.keys())) == list(sorted(self.get_expander_squares(i).keys()))
                    a = self.get_locked_squares(i)
                    for j in a:
                        try:
                            assert j in self.agents_info[i].locked_squares or 0 > j[0] or j[0] >= self.board_size or 0 > j[1] or j[1] >= self.board_size
                        except:
                            print("Player:", i)
                            print("locked squares:", self.agents_info[i].locked_squares)
                            print("Expected:", a)
                            assert False
                except:
                    print("Player:", i)
                    print("Expander squares:", self.agents_info[i].expander_squares)
                    print(self.board)
                    print("Expected:", self.get_expander_squares(i))
                    assert False
                # a = self.agents_info[i].expander_squares
                # self.agents_info[i].locked_squares == self.get_locked_squares(i)

    def possible_actions(self, player: int) -> np.ndarray:
        """Generate all possible moves for a player by checking each piece and transformation."""
        if self.agents_info[player].possible_actions is not None:
            return self.agents_info[player].possible_actions
        actions = self.possible_actions_precomputed(player)
        self.agents_info[player].possible_actions = actions
        return actions
        if self.testing_mode:
            efficient = self.possible_actions_efficient(player)
            self.efficient_time = end_time - start_time

            inefficient = self.possible_actions_inefficient(player)
            self.inefficient_time = end_time - start_time
            if BlokusEnv.NEIGHBORHOOD_TO_ENCODED_ACTIONS is not None:
                precomputed = self.possible_actions_precomputed(player)
                self.precomputed_time = end_time - start_time
                if not (set(efficient) == set(inefficient) == set(precomputed)):
                    print("Efficient actions:", efficient)
                    print("Inefficient actions:", inefficient)
                    print("Precomputed actions:", precomputed)
                    print("Current board state:\n", self.board)
                    print("Current player:", self.current_player)
                    print("Agents info:", self.agents_info)
                    raise ValueError("Actions are not the same")
            else:
                if not set(efficient) == set(inefficient):
                    print("Efficient actions:", efficient)
                    print("Inefficient actions:", inefficient)
                    print("Current board state:\n", self.board)
                    print("Current player:", self.current_player)
                    print("Agents info:", self.agents_info)
                    raise ValueError("Actions are not the same")
            return efficient
        else:
            if BlokusEnv.NEIGHBORHOOD_TO_ENCODED_ACTIONS is not None:
                return self.possible_actions_precomputed(player)
            else: 
                return self.possible_actions_efficient(player)
    
    def possible_actions_efficient(self, player: int) -> np.ndarray:
        """Generate all possible moves for a player by checking each piece and transformation efficiently."""
        actions = set()
        for shape_id in self.agents_info[player].available_pieces:
            for transformed_piece in BlokusPieceManager.get_transformations(shape_id):
                for x, y in transformed_piece.body:
                    for i, j in self.agents_info[player].expander_squares:
                        action = BlokusAction(board_size=self.board_size, action_tuple=(i - x, j - y, transformed_piece))
                        if self.place_piece(action, player, place=False):
                            actions.add(action.action_id)
        return np.array(list(actions))

    def possible_actions_inefficient(self, player: int) -> np.ndarray:
        """Generate all possible moves for a player by checking each piece and transformation inefficiently."""
        actions = []
        for shape_id in self.agents_info[player].available_pieces:
            for transformed_piece in BlokusPieceManager.get_transformations(shape_id):
                for i in range(self.board_size - transformed_piece.dimensions.x + 1):
                    for j in range(self.board_size - transformed_piece.dimensions.y + 1):
                        action = BlokusAction(board_size=self.board_size, action_tuple=(i, j, transformed_piece))
                        if self.place_piece(action=action, player=player, place=False):
                            actions.append(action.action_id)
        return np.array(list(actions))

    def possible_actions_precomputed(self, player: int) -> np.ndarray:
        """Generate all possible moves for a player by checking each piece and transformation using precomputed data."""
        actions = set()
        for expander in self.agents_info[player].expander_squares:
            actions.update(self.possible_actions_precomputed_expander_square(expander, player))
        return np.array(list(actions), dtype=np.int16)
    
    def possible_actions_precomputed_expander_square(self, expander: Vector2, player: int) -> ActType: # This is the most important (this should be fast)
        """Generate all possible moves for a player by checking each piece and transformation using precomputed data for a single expander square."""
        neighborhood = self.get_neighborhood(expander, player)
        if neighborhood == 0:
            self.freq[1] += 1
            if "1" in self.agents_info[player].available_pieces:
                return [BlokusAction.tuple_to_id(self.board_size, expander[0], expander[1], "1", 0)]
            else:
                return []
            
        dx, dy = self.agents_info[player].expander_squares[expander]

        actions = BlokusEnv.get_actions_from_neighborhood(neighborhood, self.agents_info[player].available_pieces)
        
        if (-2, 0, 16) in actions:
            if self.legal_cell((expander[0] - dx * 2, expander[1]), player):
                actions.add((-2, 0, 68)) # "U"
            if self.legal_cell((expander[0] - dx * 3, expander[1] + dy), player):
                actions.add((-3, 0, 40)) # "L5"

        if (0, -2, 13) in actions:
            if self.legal_cell((expander[0], expander[1] - dy * 2), player):
                actions.add((0, -2, 66)) # "U"
            if self.legal_cell((expander[0] + dx, expander[1] - dy * 3), player):
                actions.add((0, -3, 43)) # "L5"

        if (0, 0, 10) in actions and self.legal_cell((expander.x + dx * 4, expander.y), player):
            actions.add((0, 0, 37)) # "I5"
        if (0, 0, 9) in actions and self.legal_cell((expander.x, expander.y + dy * 4), player):
            actions.add((0, 0, 36)) # "I5"
        self.freq[len(actions)] += 1

        board_actions = []
        for board_row, board_col, piece_id in actions:
            shape_id, piece_transformation = BlokusPieceManager.piece_infos[piece_id]
            if not shape_id in self.agents_info[player].available_pieces:
                continue
            board_row, board_col, piece_transformation = BlokusAction.transform(board_row, board_col, dx, dy, shape_id, piece_transformation)
            board_row += expander.x
            board_col += expander.y
            board_actions.append(BlokusAction.tuple_to_id(self.board_size, int(board_row), int(board_col), shape_id, int(piece_transformation)))
        return board_actions

    def step(self, action_id: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Execute a step by making the player’s move and updating the current player."""
        assert isinstance(action_id, SupportsInt), f"Action id must be an integer, not {action_id, type(action_id)}"
        action = BlokusAction(board_size=self.board_size, action_id=action_id)
        if(not self.place_piece(action=action, player=self.current_player, check=True)):
            print("Invalid move")
            print("Piece:", action.piece.shape_id, "Transformation:", action.piece.transform, "Row:", action.x, "Col:", action.y)
            print("Current player:", self.current_player)
            print("Agents info:", self.agents_info)
            print("Board state:\n", self.board)
            assert False
            observation = self._get_obs()
            info = self._get_info()
            return observation, self.BAD_MOVE_PUNISHMENT, True, True, info
        
        self.steps += 1
        reward = action.piece.size
        self.agents_info[self.current_player].points += reward
        self.current_player = (self.current_player % self.num_players) + 1
        passed = 0
        while passed < self.num_players and len(self.possible_actions(self.current_player)) == 0:
            passed += 1
            self.current_player = (self.current_player % self.num_players) + 1
        
        if passed == self.num_players:
            terminated = True
        else:
            terminated = False
            
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

        elif self.render_mode == "console":
            self._render_console()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.board_size * self.render_scale * 4, self.board_size * self.render_scale * 4))
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        screen = pygame.Surface((self.board_size * self.render_scale * 4, self.board_size * self.render_scale * 4))
        colors = [(217, 199, 197), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        expander_color = (224, 224, 224)
        locked_color = (255, 255, 255) 
        screen.fill(colors[0])
        for i in range(self.board_size + 1):
            pygame.draw.line(screen, (128, 128, 128), (i * self.render_scale * 4, 0), (i * self.render_scale * 4, self.board_size * self.render_scale * 4))
            pygame.draw.line(screen, (128, 128, 128), (0, i * self.render_scale * 4), (self.board_size * self.render_scale * 4, i * self.render_scale * 4))
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                color = colors[self.board[i, j]]
                pygame.draw.rect(screen, color, pygame.Rect(j * self.render_scale * 4, i * self.render_scale * 4, self.render_scale * 4, self.render_scale * 4), border_radius=self.render_scale)
        
        for i, j in self.agents_info[self.current_player].locked_squares:
            if 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i, j] == 0:
                pygame.draw.rect(screen, locked_color, pygame.Rect(j * self.render_scale * 4, i * self.render_scale * 4, self.render_scale * 4, self.render_scale * 4), border_radius=self.render_scale)

        for i, j in self.agents_info[self.current_player].expander_squares:
            if self.board[i, j] == 0:
                pygame.draw.rect(screen, expander_color, pygame.Rect(j * self.render_scale * 4, i * self.render_scale * 4, self.render_scale * 4, self.render_scale * 4), border_radius=self.render_scale)
        
        if self.render_mode == "human":
            self.window.blit(screen, screen.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
            )

    def _render_console(self):
        for row in self.board:
            print(''.join(str(cell) for cell in row))
        print()

    def close(self):
        """Close the environment and perform any cleanup."""
        # TODO: Add any additional cleanup code here if needed
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            print("Stopped")

def register_blokus_env():
    if "gymnasium_env/Blokus-v0" not in gym.envs.registry.keys():
        gym.envs.registration.register(
            id='gymnasium_env/Blokus-v0',
            entry_point='gymnasium_env:BlokusEnv',
        )

if __name__ == "__main__":
    # Register environment
    register_blokus_env()
