import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Any, Dict, List
from gymnasium.core import ObsType, ActType
from gymnasium_env.envs.auxiliary.utils import decode, encode

# ActType = Tuple[int, int, int]

from . import BlokusPieceTransformations
import pygame
import time


"""
    Agents or Players
    How to model actions?
"""

class BlokusEnvAgentInfo:
    """Class to store information about a Blokus agent, including the player ID and available pieces."""
    def __init__(self, player_id, available_pieces:List[str]):
        self.player_id = player_id
        self.available_pieces = set(list(available_pieces))
        self.expander_squares = {}
        self.locked_squares = []
        self.started = False

class BlokusEnv(gym.Env):
    """Custom environment for the Blokus game using the Gymnasium framework.
    
    Attributes:
        metadata (dict): Metadata for the environment, including render modes.
        PIECE_IDS (list): List of piece identifiers used in the game.
    """
    metadata = {'render_modes': ['human', 'console']}
    PIECE_IDS = ['1', '2', 'I3', 'V3', 'I4', 'L4', 'O', 'T4', 'Z4', 'F', 'I5', 'L5', 'N', 'P', 'T5', 'U', 'V5', 'W', 'X', 'Y', 'Z5']
    BAD_MOVE_PUNISHMENT = -100
    
    def __init__(self, render_mode='console', board_size=14, num_players=2, render_scale=5, neighborhood=None, mode = 'testing'):
        """Initialize the Blokus environment with parameters for rendering, board size, number of players, and render scaling."""
        super(BlokusEnv, self).__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.total_efficient = 0
        self.total_inefficient = 0
        self.total_precomputed = 0
        self.freq = [0 for _ in range(130)]
        ##################################### PIECES #####################################
        self.pieces = {id: BlokusPieceTransformations(id=id) for id in BlokusEnv.PIECE_IDS}

        self.different_pieces = sum(map(lambda x: len(x.transformations), self.pieces.values()))
        assert self.different_pieces == 91

        self._get_piece_info = []
        
        for i in range(len(self.pieces)):
            for j in range(len(self.pieces[BlokusEnv.PIECE_IDS[i]].transformations)):
                self.pieces[BlokusEnv.PIECE_IDS[i]].transformations[j].idx = len(self._get_piece_info)
                self._get_piece_info.append((BlokusEnv.PIECE_IDS[i], j))

        assert len(self._get_piece_info) == self.different_pieces
        self.neighborhood_to_encoded_actions = None
        if neighborhood is not None:
            self.neighborhood_to_encoded_actions = np.load(f"{neighborhood}/compute_actions.pkl", allow_pickle=True)
            print("Neighborhood to encoded actions loaded")
            self.neighbor_pos = np.load(f"{neighborhood}/neighbor_pos.pkl", allow_pickle=True)
            def get_neighborhood(expander, player):
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
                x, y = expander
                dx, dy = self.agents_info[player].expander_squares[expander] # first direction
                if (not self.legal_cell((x + dx, y), player)) and (not self.legal_cell((x, y + dy), player)):
                    return 0
                for idx, (i, j) in enumerate(self.neighbor_pos):
                    neighbor = (x + dx * i, y + dy * j)
                    if self.legal_cell(neighbor, player):
                        neighborhood |= (1 << idx)
                return neighborhood
            
            self.get_neighborhood = get_neighborhood
            def get_actions_from_neighborhood(neighborhood: int) -> List[int]:
                actions = self.neighborhood_to_encoded_actions[neighborhood]
                actions = list(map(decode, actions))
                return actions
            self.get_actions_from_neighborhood = get_actions_from_neighborhood


        ## ACTIONS DECODING/ENCODING ##
        # f: (row, col, piece_id, piece_transformation) -> action [0, board_size*board_size*91]
        self._tuple_to_action = lambda action: action[0] * self.board_size * self.different_pieces + action[1] * self.different_pieces + self.pieces[action[2]].transformations[action[3]].idx
        # g: action -> (row, col, piece_id, piece_transformation) [(0, 0, '1', 0), (0, 0, '1', 1), ...]
        self._action_to_tuple = lambda action: (action // (self.different_pieces * self.board_size), (action // self.different_pieces) % self.board_size, *self._get_piece_info[action % self.different_pieces])
        
        ##################################### BOARD ######################################
        self.board_size = board_size

        ############################### AGENTS PARAMETERS ###############################
        self.num_players = num_players

        ############################## RENDERING PARAMETERS ##############################
        self.render_mode = render_mode
        self.render_scale = render_scale 

        ########################### ACTION AND OBSERVATION SPACES ######################################
        self.action_space = spaces.Discrete(self.board_size * self.board_size * self.different_pieces)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=4, shape=(self.board_size, self.board_size), dtype=np.int32),  # 20x20 board, 0-4 integer values (customize as needed)
            # "available_pieces": spaces.MultiBinary(len(self.pieces)),  # Binary vector for each piece (1 = available, 0 = used)
            "expander_squares": spaces.Dict({
                "coordinates": spaces.MultiBinary((self.board_size, self.board_size)),  # Binary 20x20 matrix indicating expander squares for the current player
                "count": spaces.Discrete(self.board_size * self.board_size)  # Count of expander squares
            }),  # Dictionary containing expander squares information
            "locked_squares": spaces.MultiBinary((self.board_size, self.board_size)),  # Binary 20x20 matrix indicating locked squares for the current player
            "current_player": spaces.Discrete(4), # Assuming a 4-player game
            "possible_actions": spaces.MultiBinary(self.board_size * self.board_size * self.different_pieces)  # Binary vector for each possible action (1 = possible, 0 = impossible)
        })
        self.reset()
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.mode = mode
        self.window = None
        self.clock = None

        print(f"Initialised on {mode} mode")
        
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
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.agents_info = [0] + [BlokusEnvAgentInfo(player_id=i, available_pieces=self.pieces.keys()) for i in range(1, self.num_players + 1)]

        if self.num_players == 1:
            self.agents_info[1].expander_squares = {(0, 0):(1, 1)}
        elif self.num_players == 2:
            self.agents_info[1].expander_squares = {(0, 0):(1, 1)}
            self.agents_info[2].expander_squares = {(self.board_size - 1, self.board_size - 1):(-1, -1)}
        else:
            raise ValueError("Only 1-2 player is supported for now :3")
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self):
        return {
            "board": np.array(self.board),  # 2D array of the current board state
            # "available_pieces": [piece_id for piece_id, piece in self.pieces.items() if not piece.used],  # List of unused pieces
            "expander_squares": np.array(self.agents_info[self.current_player].expander_squares),  # Expander squares for the current player
            "locked_squares": np.array(self.agents_info[self.current_player].locked_squares),  # Locked squares for the current player
            "current_player": self.current_player  # ID of the current player
        }

    def _get_info(self):
        return {
            # "distance": np.linalg.norm(
            #     # self._agent_location - self._target_location, ord=1
            # )
        }

    def legal_cell(self, pos, player):
        return (
            0 <= pos[0] < self.board_size and
            0 <= pos[1] < self.board_size and
            self.board[pos] == 0 and
            not pos in self.agents_info[player].locked_squares
        )
    
    def get_locked_squares(self, player):
        """Return locked squares for a player, where placement is restricted based on adjacent cells."""
        ret = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                add = False
                for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if min(i + x, j + y) >= 0 and max(i + x, j + y) < self.board_size and self.board[i + x, j + y] == player:
                        add = True
                if add and self.board[i, j] != player:
                    ret.append((i, j))
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
    
    def place_piece(self, action_parameters, player, place=True, check=True):
        """
        Place a piece on the board and update the game state.
        Args:
            action_parameters (tuple): A tuple containing the row, column, piece ID, and piece transformation.
            player (int): The ID of the player placing the piece.
            place (bool, optional): If True, the piece will be placed on the board. If False, only checks if the piece can be placed. Defaults to True.
        Returns:
            bool: True if the piece was successfully placed or can be placed, False otherwise.
        """
        row, col, piece_id, piece_transformation = action_parameters
        piece = self.pieces[piece_id].transformations[piece_transformation]
        if check:
            if not piece.id in self.agents_info[player].available_pieces:
                return False #### PIECE ALREADY USED ####

            can_place = False
            for i, j in piece.shape:
                board_row, board_col = i + row, j + col
                if not self.legal_cell((board_row, board_col), player):
                    return False
                if (board_row, board_col) in self.agents_info[player].expander_squares:
                    can_place = True #### TOUCHES A CORNER PIECE ####
        else:
            if not place:
                return True
        if not can_place:
            return False
        
        if not place:
            return True
        
        # self.update_attributes(player, row, col, piece)
        self.faster_update_attributes(player, row, col, piece)
        return True

    def update_attributes(self, player, row, col, piece):
        ### UPDATE OTHER ATTRIBUTES TO MAKE IT FASTER ###
        ### EXAMPLE: LOCKED/EXPANDER SQUARES ###
        for i, j in piece.shape:
            self.board[i + row, j + col] = player
            
        self.agents_info[player].available_pieces.remove(piece.id)
        self.agents_info[player].started = True

        for i in range(1, self.num_players + 1):
            self.agents_info[i].locked_squares = self.get_locked_squares(i)
            self.agents_info[i].expander_squares = self.get_expander_squares(i)
            
    def faster_update_attributes(self, player, row, col, piece):
        self.agents_info[player].available_pieces.remove(piece.id)
        self.agents_info[player].started = True

        for i, j in piece.locked:
            self.agents_info[player].locked_squares.append((row + i, col + j))
            if (row + i, col + j) in self.agents_info[player].expander_squares:
                self.agents_info[player].expander_squares.pop((row + i, col + j))

        for (i, j), (dx, dy) in piece.expanders.items():
            if self.legal_cell((row + i, col + j), player): # add own expanders
                self.agents_info[player].expander_squares[(row + i, col + j)] = (dx, dy)
        

        for i, j in piece.shape:
            self.board[i + row, j + col] = player
            if (i + row, j + col) in self.agents_info[player].expander_squares:
                self.agents_info[player].expander_squares.pop((i + row, j + col))
            for k in range(1, self.num_players + 1):
                if k == player:
                    continue
                if (i + row, j + col) in self.agents_info[k].expander_squares:
                    self.agents_info[k].expander_squares.pop((i + row, j + col))
        
        if self.mode == 'testing':
            for i in range(1, self.num_players + 1):
                try:
                    assert list(sorted(self.agents_info[i].expander_squares.keys())) == list(sorted(self.get_expander_squares(i).keys()))
                    a = self.get_locked_squares(i)
                    for j in range(len(a)):
                        try:
                            assert a[j] in self.agents_info[i].locked_squares or 0 > a[j][0] or a[j][0] >= self.board_size or 0 > a[j][1] or a[j][1] >= self.board_size
                        except:
                            print("Player:", i)
                            print("Expander squares:", self.agents_info[i].expander_squares)
                            print("Expected:", a)
                            assert False
                except:
                    print("Player:", i)
                    print("Expander squares:", self.agents_info[i].expander_squares)
                    print("Expected:", self.get_expander_squares(i))
                    assert False
                # a = self.agents_info[i].expander_squares
                # self.agents_info[i].locked_squares == self.get_locked_squares(i)


    def possible_actions(self, player: int) -> list[Tuple[int, int, str, int]]:
        """Generate all possible moves for a player by checking each piece and transformation."""
        # start_time = time.time()
        actions = self.possible_actions_precomputed(player)
        # end_time = time.time()
        # self.precomputed_time = end_time - start_time
        # self.log()
        return actions
        if self.mode == 'testing':
            start_time = time.time()
            efficient = self.possible_actions_efficient(player)
            end_time = time.time()
            self.efficient_time = end_time - start_time

            start_time = time.time()
            inefficient = self.possible_actions_inefficient(player)
            end_time = time.time()
            self.inefficient_time = end_time - start_time
            if self.neighborhood_to_encoded_actions is not None:
                start_time = time.time()
                precomputed = self.possible_actions_precomputed(player)
                end_time = time.time()
                self.precomputed_time = end_time - start_time
                if not (set(efficient) == set(inefficient) == set(precomputed)):
                    print("Efficient actions:", efficient)
                    print("Inefficient actions:", inefficient)
                    print("Precomputed actions:", precomputed)
                    print("Current board state:\n", self.board)
                    print("Current player:", self.current_player)
                    print("Agents info:", self.agents_info)
                    raise ValueError("Actions are not the same")
                self.log()
            else:
                if not set(efficient) == set(inefficient):
                    print("Efficient actions:", efficient)
                    print("Inefficient actions:", inefficient)
                    print("Current board state:\n", self.board)
                    print("Current player:", self.current_player)
                    print("Agents info:", self.agents_info)
                    # self.log()
                    raise ValueError("Actions are not the same")
            return efficient
        else:
            if self.neighborhood_to_encoded_actions is not None:
                return self.possible_actions_precomputed(player)
            else: 
                return self.possible_actions_efficient(player)
    
    def possible_actions_efficient(self, player: int) -> ActType:
        """Generate all possible moves for a player by checking each piece and transformation efficiently."""
        actions = set()
        for piece_id in self.agents_info[player].available_pieces:
            for piece_transformation in range(len(self.pieces[piece_id].transformations)):
                piece_shape = self.pieces[piece_id].transformations[piece_transformation].shape
                for x, y in piece_shape:
                    for i, j in self.agents_info[player].expander_squares:
                        if self.place_piece((i - x, j - y, piece_id, piece_transformation), player, place=False):
                            actions.add(self._tuple_to_action((i - x, j - y, piece_id, piece_transformation)))
        return sorted(list(actions))

    def possible_actions_inefficient(self, player: int) -> ActType:
        """Generate all possible moves for a player by checking each piece and transformation inefficiently."""
        actions = []
        for piece_id in self.agents_info[player].available_pieces:
            for piece_transformation in range(len(self.pieces[piece_id].transformations)):
                piece_shape = self.pieces[piece_id].transformations[piece_transformation].size
                
                for i in range(self.board_size - piece_shape[0] + 1):
                    for j in range(self.board_size - piece_shape[1] + 1):
                        if self.place_piece((i, j, piece_id, piece_transformation), player, place=False):
                            actions.append(self._tuple_to_action((i, j, piece_id, piece_transformation)))
        return sorted(list(actions))

    def possible_actions_precomputed(self, player: int) -> ActType:
        """Generate all possible moves for a player by checking each piece and transformation using precomputed data."""
        actions = set()
        for expander in self.agents_info[player].expander_squares:
            actions.update(self.possible_actions_precomputed_expander_square(expander, player))
        
        return sorted(list(actions))
    
    def log(self):
        self.total_efficient += self.efficient_time
        self.total_inefficient += self.inefficient_time
        self.total_precomputed += self.precomputed_time
        # print("Efficient time:", self.efficient_time)
        # print("Inefficient time:", self.inefficient_time)
        # print("Precomputed time:", self.precomputed_time)

    def possible_actions_precomputed_expander_square(self, expander, player: int) -> ActType:
        """Generate all possible moves for a player by checking each piece and transformation using precomputed data for a single expander square."""
        neighborhood = self.get_neighborhood(expander, player)
        if neighborhood == 0:
            self.freq[1] += 1
            if "1" in self.agents_info[player].available_pieces:
                return [self._tuple_to_action((expander[0], expander[1], "1", 0))]
            else:
                return []
            
        dx, dy = self.agents_info[player].expander_squares[expander]
        actions = self.get_actions_from_neighborhood(neighborhood)
        # print("neighborhood:", neighborhood, "actions:", len(actions))
        if (-2, 0, 16) in actions:
            if self.legal_cell((expander[0] - dx * 2, expander[1]), player):
                actions.append((-2, 0, 68))
            if self.legal_cell((expander[0] - dx * 3, expander[1] + dy), player):
                actions.append((-3, 0, 40))

        if (0, -2, 13) in actions:
            if self.legal_cell((expander[0], expander[1] - dy * 2), player):
                actions.append((0, -2, 66)) 
            if self.legal_cell((expander[0] + dx, expander[1] - dy * 3), player):
                actions.append((0, -3, 43))

        if self.legal_cell((expander[0] + dx * 4, expander[1]), player) and (0, 0, 10) in actions:
            actions.append((0, 0, 37))
        if self.legal_cell((expander[0], expander[1] + dy * 4), player) and (0, 0, 9) in actions:
            actions.append((0, 0, 36))
        self.freq[len(actions)] += 1

        board_actions = []
        for row, col, piece in actions:
            piece_id, piece_transformation = self._get_piece_info[piece]
            h, w = self.pieces[piece_id].transformations[piece_transformation].size
            original_piece_transform = self.pieces[piece_id].aux[piece_transformation]
            if dx == -1:
                board_row = 1 - row - h + expander[0]
                original_piece_transform ^= 1
            else: 
                board_row = row + expander[0]
            if dy == -1:
                board_col = 1 - col - w + expander[1]
                original_piece_transform ^= 2
            else:
                board_col = col + expander[1]
            board_transformation = self.pieces[piece_id].indexes[original_piece_transform]
    
            if piece_id in self.agents_info[player].available_pieces:
                # print("Piece:", piece_id, "Transformation:", board_transformation, "Row:", board_row, "Col:", board_col)
                board_actions.append(self._tuple_to_action((int(board_row), int(board_col), piece_id, int(board_transformation))))
        # print(dx, dy, actions)
        return board_actions
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Execute a step by making the player’s move and updating the current player."""
        row, col, piece_id, piece_transformation = self._action_to_tuple(action)

        if(not self.place_piece(action_parameters=(row, col, piece_id, piece_transformation), player=self.current_player)):
            self.reset()
            print("Invalid move")
            assert False
            return {}, self.BAD_MOVE_PUNISHMENT, True, True, {}
        
        self.current_player = (self.current_player % self.num_players) + 1

        terminated = False
        reward = len(self.pieces[piece_id].transformations[piece_transformation].shape)
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'console':
            self._render_console()

    def _render_human(self):
        pygame.init()
        if self.window is None:
            self.window = pygame.display.set_mode((self.board_size * self.render_scale * 4, self.board_size * self.render_scale * 4))
            self.clock = pygame.time.Clock()

        screen = self.window
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
        
        pygame.display.flip()
        
        # Add a loop to keep the window open
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.clock.tick(60)  # Limit the frame rate to 60 FPS
        # pygame.display.quit() 
        # pygame.quit()
        # print("Stopped")

    def _render_console(self):
        for row in self.board:
            print(''.join(str(cell) for cell in row))

    def close(self):
        """Close the environment and perform any cleanup."""
        # TODO: Add any additional cleanup code here if needed
        if self.render_mode == 'human' and self.window is not None:
            pygame.display.quit()
            pygame.quit()
            print("Stopped")

# Register the environment with Gymnasium
gym.envs.registration.register(
    id='gymnasium_env/Blokus-v0',
    entry_point='gymnasium_env:BlokusEnv',
)

# if __name__ == "__main__":
#     env = BlokusEnv()
#     env.reset()
#     env.render()