import pygame
import sys
import numpy as np

from game_state import GameState
from gymnasium_env import BlokusEnv, SingleAgentBlokusEnv, BlokusAction, BlokusPieceManager, PIECE_SHAPE_IDS, PIECE_ORDER, MultipleColorsEncoding
from src.agents import Agent, RandomAgent, HeuristicAgent, ABPruningAgent, QNetworkAgent
from gymnasium_env import BlokusTheme

from src.agents.heuristic.heuristics import max_our_expander_diff_1_3, max_my_expanders, min_his_expanders

from src.agents.qnetwork.archis import ColorfulArch, UnicolorArch

class BlokusGameUI:
    def __init__(self, board_size: int = 10, player_turn: int = 2, opponent_agent: Agent = None):
        # Initialize Pygame
        pygame.init()

        self.board_size = board_size
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Blokus Interactive Demo")

        self.opponent_agent = opponent_agent
        self.opponent_agent.eval()
        self.player_turn = player_turn
        env = BlokusEnv(
            board_size=self.board_size,
            num_players=2,
            render_mode="console",
        )
        env = MultipleColorsEncoding(env)
        self.env = SingleAgentBlokusEnv(
            base_env=env,
            player_turn=self.player_turn,
            hidden_agents=[self.opponent_agent],
        )
        obs, info = self.env.reset()
        self.board = obs["state"]
        self.current_actions(obs)

        self.current_board = obs["state"]

        # Game State
        self.selected_piece_index = 0
        self.selected_transformation_index = 0
        self.selected_action_index = 0
        self.game_state = GameState.CHOOSING_PIECE  # Start in piece selection mode
        self.theme = BlokusTheme(render_scale=30)

    def current_actions(self, obs):
        self.make_structured_actions(obs)

    def make_structured_actions(self, obs):
        action_ids = obs["possible_actions"]
        actions = [BlokusAction(board_size=self.board_size, action_id=action_id) for action_id in action_ids]
        
        # Convert actions into a tuple representation
        tupled_actions = [(action.piece.shape_id, action.piece.transform, action.x, action.y, action) for action in actions]
        
        # Sort actions based on shape order (ensures consistent ordering)
        tupled_actions = sorted(tupled_actions, key=lambda x: PIECE_ORDER[x[0]])

        shape_map = {}
        
        for shape_id, transform, x, y, action in tupled_actions:
            if shape_id not in shape_map:
                shape_map[shape_id] = {}

            if transform not in shape_map[shape_id]:
                shape_map[shape_id][transform] = []
            
            shape_map[shape_id][transform].append(action)

        # Convert dictionary to ordered list of lists of lists
        self.structured_actions = [
            [shape_map[shape_id][transform] for transform in sorted(shape_map[shape_id].keys())]
            for shape_id in sorted(shape_map.keys(), key=lambda sid: PIECE_ORDER[sid])
        ]

    ## TEST AGAINST HEURISTICS

    def draw_board(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                player = self.current_board[x, y]
                pygame.draw.rect(self.screen, self.theme.cell_color(player_id=player), pygame.Rect(x * self.theme.render_scale, y * self.theme.render_scale, self.theme.render_scale, self.theme.render_scale))
                pygame.draw.rect(self.screen, self.theme.grid_color, pygame.Rect(x * self.theme.render_scale, y * self.theme.render_scale, self.theme.render_scale, self.theme.render_scale), 1)  # Black border
        
        if self.structured_actions:
            action: BlokusAction = self.structured_actions[self.selected_piece_index][self.selected_transformation_index][self.selected_action_index]
            for x, y in action.body:
                pygame.draw.rect(self.screen, self.theme.expander_color(player_id=self.player_turn), pygame.Rect(x * self.theme.render_scale, y * self.theme.render_scale, self.theme.render_scale, self.theme.render_scale))

    def run(self):
        running = True
        while running:
            self.screen.fill(self.theme.background_color)  # Clear screen
            for event in pygame.event.get():
                # Old value
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.game_state = GameState.CHOOSING_PIECE
                        self.selected_piece_index = 0
                        self.selected_transformation_index = 0
                        self.selected_action_index = 0 
                        
                        print(f"I am resetting the state")
                    elif self.game_state == GameState.WAITING:
                        if event.key == pygame.K_RETURN:
                            obs, info = self.env.reset()
                            self.board = obs["state"]
                            self.current_actions(obs)

                            self.current_board = obs["state"]
                            self.game_state = GameState.CHOOSING_PIECE
                            # running = False
                    elif self.game_state == GameState.CHOOSING_PIECE:
                        if event.key == pygame.K_DOWN:
                            self.selected_piece_index = (self.selected_piece_index + 1) % len(self.structured_actions)
                        elif event.key == pygame.K_UP:
                            self.selected_piece_index = (self.selected_piece_index - 1) % len(self.structured_actions)
                        elif event.key == pygame.K_RETURN:
                            self.game_state = GameState.CHOOSING_TRANSFORMATION
                        elif event.key == pygame.K_DELETE:
                            # self.game_state = GameState.WAITING
                            pass  # Handle going back if needed

                    elif self.game_state == GameState.CHOOSING_TRANSFORMATION:
                        if event.key == pygame.K_DOWN:
                            self.selected_transformation_index = (self.selected_transformation_index + 1) % len(self.structured_actions[self.selected_piece_index])
                        elif event.key == pygame.K_UP:
                            self.selected_transformation_index = (self.selected_transformation_index - 1) % len(self.structured_actions[self.selected_piece_index])
                        elif event.key == pygame.K_RETURN:
                            self.game_state = GameState.PLACING_PIECE
                        elif event.key == pygame.K_DELETE:
                            self.game_state = GameState.CHOOSING_PIECE

                    elif self.game_state == GameState.PLACING_PIECE:
                        if event.key == pygame.K_DOWN:
                            self.selected_action_index = (self.selected_action_index + 1) % len(self.structured_actions[self.selected_piece_index][self.selected_transformation_index])
                        elif event.key == pygame.K_UP:
                            self.selected_action_index = (self.selected_action_index - 1) % len(self.structured_actions[self.selected_piece_index][self.selected_transformation_index])
                        elif event.key == pygame.K_RETURN:
                            print(self.selected_piece_index, self.selected_transformation_index, self.selected_action_index)
                            action = self.structured_actions[self.selected_piece_index][self.selected_transformation_index][self.selected_action_index]
                            print(action.piece.shape_id, action.piece.transform, action.x, action.y)
                            obs, reward, terminated, truncated, info = self.env.step(action.action_id)

                            self.selected_piece_index, self.selected_transformation_index, self.selected_action_index = 0, 0, 0
                            self.current_board = obs["state"]
                            self.current_actions(obs)

                            if terminated:
                                print("Game Over")
                                scores = obs["points"]
                                print(f"Player 1: {scores[1]}")
                                print(f"Player 2: {scores[2]}")
                                if scores[1] > scores[2]:
                                    print("Player 1 wins!")
                                elif scores[2] > scores[1]:
                                    print("Player 2 wins!")
                                else:
                                    print("It's a tie!")
                                self.game_state = GameState.WAITING
                            else:
                                self.game_state = GameState.CHOOSING_PIECE
                            # self.game_state = GameState.WAITING  # Reset for next turn
                        elif event.key == pygame.K_DELETE:
                            self.game_state = GameState.CHOOSING_TRANSFORMATION

                # Render based on game state
                self.draw_board()
                pygame.display.flip()  # Update display

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    BOARD_SIZE = 10
    # player_turn = int(input("Do you want to play first or second? (1/2): "))
    player_turn = 2

    # opponent_agent = HeuristicAgent(
    #     board_size=BOARD_SIZE,
    #     func=max_our_expander_diff_1_3,
    # )
    # opponent_agent = ABPruningAgent(
    #     board_size=BOARD_SIZE,
    #     depth=2,
    #     use_cache=True
    # )
    opponent_agent = QNetworkAgent(
        device="cpu",
        board_size=BOARD_SIZE,
        model_class=ColorfulArch,
        model_folder="first_working",
    ) 
    game = BlokusGameUI(
        board_size = BOARD_SIZE,
        player_turn = player_turn,
        opponent_agent = opponent_agent
    )
    game.run()