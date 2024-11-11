import os
import matplotlib.pyplot as plt

from gymnasium_env.envs.blokus_piece import BlokusPieceTransformations, BlokusPiece

def save_piece_image(piece: BlokusPiece, folder_path, filename):
    N = 5
    grid_size = 15  # Fixed grid size for all images
    grid = [[0 for _ in range(grid_size * N)] for _ in range(grid_size * N)]
    
    piece_x = [pos[0] for pos in piece.shape]
    piece_y = [pos[1] for pos in piece.shape]
    min_x, max_x = min(piece_x), max(piece_x)
    min_y, max_y = min(piece_y), max(piece_y)
    
    offset_x = (grid_size - (max_x - min_x + 1)) // 2
    offset_y = (grid_size - (max_y - min_y + 1)) // 2
    
    for x, y in piece.shape:
        for i in range(N):
            for j in range(N):
                grid[(x + offset_x) * N + i][(y + offset_y) * N + j] = 1
    
    fig, ax = plt.subplots()  # Adjust figure size for better resolution
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    # ax.set_xlim(-1, grid_size * N)
    # ax.set_ylim(grid_size * N, -1)
    # Add grid lines
    # ax.set_xticks([i - 0.5 for i in range(1, grid_size * N)], minor=True)
    # ax.set_yticks([i - 0.5 for i in range(1, grid_size * N)], minor=True)
    # ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    # ax.tick_params(which='minor', size=0)  # Hide minor tick marks
    
    ax.axis('off')
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    plt.savefig(os.path.join(folder_path, filename), bbox_inches='tight', pad_inches=0.1)
    plt.close()

def save_board_image(board, folder_path="board_images", filename="board.png"):
    N = 5
    grid_size = 20  # Fixed grid size for the board
    grid = [[0 for _ in range(grid_size * N)] for _ in range(grid_size * N)]

    for x in range(grid_size):
        for y in range(grid_size):
            if board[x][y] != 0:
                for i in range(N):
                    for j in range(N):
                        grid[x * N + i][y * N + j] = 1
                        color = board[x][y]
                        for i in range(N):
                            for j in range(N):
                                grid[x * N + i][y * N + j] = color

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.axis('off')

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, filename), bbox_inches='tight', pad_inches=0.1)
    plt.close()

folder_path = 'gymnasium_env/piece_images'

PIECE_IDS = ['1', '2', 'I3', 'V3', 'I4', 'L4', 'O', 'T4', 'Z4', 'F', 'I5', 'L5', 'N', 'P', 'T5', 'U', 'V5', 'W', 'X', 'Y', 'Z5']

pieces = {id: BlokusPieceTransformations(id=id) for id in PIECE_IDS}

j = 0
for id, piece in pieces.items():
    for i, transformation in enumerate(piece.transformations):
        save_piece_image(transformation, folder_path, f'{j:03d}_{id}_transformation_{i}.png')
        j += 1
