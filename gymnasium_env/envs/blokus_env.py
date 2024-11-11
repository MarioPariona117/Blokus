import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Any, Dict, List
from gymnasium.core import ObsType, ActType

# ActType = Tuple[int, int, int]

from . import BlokusPieceTransformations
import pygame


"""
    Agents or Players
    How to model actions?

"""


class BlokusEnvAgentInfo:
    """Class to store information about a Blokus agent, including the player ID and available pieces."""
    def __init__(self, player_id, available_pieces:List[str]):
        self.player_id = player_id
        self.available_pieces = set(list(available_pieces))
        self.expander_squares = []
        self.locked_squares = []
        self.started = False
    # def get_pieces():

class BlokusEnv(gym.Env):
    """Custom environment for the Blokus game using the Gymnasium framework.
    
    Attributes:
        metadata (dict): Metadata for the environment, including render modes.
        PIECE_IDS (list): List of piece identifiers used in the game.
    """
    metadata = {'render_modes': ['human', 'console']}
    PIECE_IDS = ['1', '2', 'I3', 'V3', 'I4', 'L4', 'O', 'T4', 'Z4', 'F', 'I5', 'L5', 'N', 'P', 'T5', 'U', 'V5', 'W', 'X', 'Y', 'Z5']
    BAD_MOVE_PUNISHMENT = -100
    
    def __init__(self, render_mode='console', board_size=14, num_players=2, render_scale=5):
        """Initialize the Blokus environment with parameters for rendering, board size, number of players, and render scaling."""
        super(BlokusEnv, self).__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        ##################################### PIECES #####################################
        self.pieces = {id: BlokusPieceTransformations(id=id) for id in BlokusEnv.PIECE_IDS}

        self.different_pieces = sum(map(lambda x: len(x.transformations), self.pieces.values()))
        assert self.different_pieces == 84

        self._get_piece_info = []
        
        for i in range(len(self.pieces)):
            for j in range(len(self.pieces[BlokusEnv.PIECE_IDS[i]].transformations)):
                self.pieces[BlokusEnv.PIECE_IDS[i]].transformations[j].idx = len(self._get_piece_info)
                self._get_piece_info.append((BlokusEnv.PIECE_IDS[i], j))

        assert len(self._get_piece_info) == self.different_pieces

        ## ACTIONS TO INT ##
        self._encode_action_to_int = lambda action: action[0] * self.board_size * self.different_pieces + action[1] * self.different_pieces + self.pieces[action[2]].transformations[action[3]].idx
        self._decode_action = lambda action: (action // (self.different_pieces * self.board_size), (action // self.different_pieces) % self.board_size, *self._get_piece_info[action % self.different_pieces])
        
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
            "expander_squares": spaces.MultiBinary((self.board_size, self.board_size)),  # Binary 20x20 matrix indicating expander squares for the current player
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
        self.window = None
        self.clock = None
        
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
            self.agents_info[1].expander_squares = [(0, 0)]
        elif self.num_players == 2:
            self.agents_info[1].expander_squares = [(0, 0)]
            self.agents_info[2].expander_squares = [(self.board_size - 1, self.board_size - 1)]
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
        ret = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                add, not_add = False, False
                for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    if min(i + x, j + y) >= 0 and max(i + x, j + y) < self.board_size and self.board[i + x, j + y] == player:
                        add = True
                for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if min(i + x, j + y) >= 0 and max(i + x, j + y) < self.board_size and self.board[i + x, j + y] == player:
                        not_add = True
                if not not_add and add and self.board[i, j] != player:
                    ret.append((i, j))
        return ret
    
    def place_piece(self, action_parameters, player, place=True):
        """Place a piece on the board, and update game state."""
        # row, col, piece_id, piece_transformation = action
        row, col, piece_id, piece_transformation = action_parameters

        if not piece_id in self.agents_info[player].available_pieces:
            return False #### PIECE ALREADY USED ####
        piece_shape = self.pieces[piece_id].transformations[piece_transformation].shape

        can_place = False
        for i, j in piece_shape:
            board_row, board_col = i + row, j + col
            if board_row < 0 or board_row >= self.board_size or board_col < 0 or board_col >= self.board_size:
                return False #### OUTSIDE THE BOARD ####
            if (board_row, board_col) in self.agents_info[player].expander_squares:
                can_place = True #### TOUCHES A CORNER PIECE ####
            if self.board[board_row, board_col] != 0:
                return False #### IS COLOURED ALREADY ####
            if (board_row, board_col) in self.agents_info[player].locked_squares:
                return False #### TOUCH OWN COLOR BY SIDE ####
        if not can_place:
            return False
        
        if not place:
            return True
        
        for i, j in piece_shape:
            self.board[i + row, j + col] = player
            
        self.agents_info[player].available_pieces.remove(piece_id)

        ### UPDATE OTHER ATTRIBUTES TO MAKE IT FASTER ###
        ### EXAMPLE: LOCKED/EXPANDER SQUARES ###
        self.agents_info[player].started = True
        for i in range(1, self.num_players + 1):
            self.agents_info[i].locked_squares = self.get_locked_squares(i)
            if self.agents_info[i].started:
                self.agents_info[i].expander_squares = self.get_expander_squares(i)

        return True

    def possible_actions(self, player: int) -> list[Tuple[int, int, str, int]]:
        """Generate all possible moves for a player by checking each piece and transformation."""
        pass 
    
    def possible_actions_efficient(self, player: int) -> ActType:
        """Generate all possible moves for a player by checking each piece and transformation efficiently."""
        actions = set()
        for piece_id in self.agents_info[player].available_pieces:
            for piece_transformation in range(len(self.pieces[piece_id].transformations)):
                piece_shape = self.pieces[piece_id].transformations[piece_transformation].shape
                for x, y in piece_shape:
                    for i, j in self.agents_info[player].expander_squares:
                        if self.place_piece((i - x, j - y, piece_id, piece_transformation), player, place=False):
                            actions.add(self._encode_action_to_int((i - x, j - y, piece_id, piece_transformation)))
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
                            actions.append(self._encode_action_to_int((i, j, piece_id, piece_transformation)))
        return sorted(list(actions))

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Execute a step by making the player’s move and updating the current player."""
        row, col, piece = (action // (self.different_pieces * self.board_size), (action // self.different_pieces) % self.board_size, action % self.different_pieces)
        # print(row, col, piece)
        piece_id, piece_transformation = self._get_piece_info[piece]

        if(not self.place_piece(action_parameters=(row, col, piece_id, piece_transformation), player=self.current_player)):
            self.reset()
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
        screen = pygame.display.set_mode((self.board_size * self.render_scale * 4, self.board_size * self.render_scale * 4))
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
            if self.board[i, j] == 0:
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
        pygame.display.quit() 
        pygame.quit()
        print("Stopped")

    def _render_console(self):
        for row in self.board:
            print(''.join(str(cell) for cell in row))

    def close(self):
        """Close the environment and perform any cleanup."""
        # TODO: Add any additional cleanup code here if needed
        if self.render_mode == 'human':
            pygame.display.quit()
            pygame.quit()
        # pass
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# Register the environment with Gymnasium
gym.envs.registration.register(
    id='gymnasium_env/Blokus-v0',
    entry_point='gymnasium_env:BlokusEnv',
)

if __name__ == "__main__":
    env = BlokusEnv()
    env.reset()
    env.render()