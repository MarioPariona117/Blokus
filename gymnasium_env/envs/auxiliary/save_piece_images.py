import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

from gymnasium_env.envs.blokus_piece import BlokusPieceTransformations, BlokusPiece, BlokusPieceManager, PIECE_SHAPE_IDS
from gymnasium_env.envs.blokus_theme import BlokusTheme

theme = BlokusTheme()

grid_width = 10
scale = 111

## TODO: fix this code

def draw_piece(grid, piece: BlokusPiece, offset_x, offset_y, player_id: int = 1, corners=False):
    for x, y in piece.body:
        for i in range(grid_width * 2, scale):
            for j in range(grid_width * 2, scale):
                grid[(x + offset_x) * scale + i][(y + offset_y) * scale + j] = theme.cell_color(player_id=player_id)

    if corners:
        for expander in piece.expanders:
            for i in range(grid_width * 2, scale):
                for j in range(grid_width * 2, scale):
                    if (i - scale // 2) ** 2 + (j - scale // 2) ** 2 <= (scale // 2) ** 2:
                        grid[(expander[0] + offset_x) * scale + i][(expander[1] + offset_y) * scale + j] = theme.expander_color(player_id=player_id)
        
        for locked in piece.locked:
            for i in range(grid_width * 2, scale):
                for j in range(grid_width * 2, scale):
                    if (i == j or i + j == scale - 1):
                        grid[(locked[0] + offset_x) * scale + i][(locked[1] + offset_y) * scale + j] = theme.expander_color(player_id=player_id)

def save_piece_image_to_pdf(pdf, piece: BlokusPiece):
    grid_size = 15  # Fixed grid size for all images
    
    piece_x = [pos[0] for pos in piece.body]
    piece_y = [pos[1] for pos in piece.body]
    min_x, max_x = min(piece_x), max(piece_x)
    min_y, max_y = min(piece_y), max(piece_y)
    
    offset_x = (grid_size - (max_x - min_x + 1)) // 2
    offset_y = (grid_size - (max_y - min_y + 1)) // 2
    grid = [[theme.background_color for _ in range(grid_size * scale)] for _ in range(grid_size * scale)]
    
    draw_piece(grid, piece, offset_x, offset_y, player_id=1, corners=True)
    
    fig, ax = plt.subplots()
    ax.imshow(grid, interpolation='nearest')
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def save_board_image_to_pdf(pdf, dic):
    grid_size = (18, 18)  # Fixed grid size for all images
    grid = [[theme.background_color for _ in range(grid_size[0] * scale + grid_width)] for _ in range(grid_size[1] * scale + grid_width)]
    pieces = [BlokusPieceManager.get_piece(shape_id=piece_id, transform=dic[piece_id][0]) for piece_id in PIECE_SHAPE_IDS if piece_id in dic]
    for piece in pieces:
        xy = dic[piece.shape_id][1]
        draw_piece(grid, piece, xy[0], xy[1], corners=False)
    
    fig, ax = plt.subplots()
    ax.imshow(grid, interpolation='nearest')
    ax.set_xticks([i * scale for i in range(grid_size[0] + 1)], minor=True)
    ax.set_yticks([i * scale for i in range(grid_size[1] + 1)], minor=True)
    ax.grid(which='minor', color=theme.grid_color_norm, linestyle='-', linewidth=grid_width * 2)
    ax.tick_params(which='minor', size=0) 
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

folder_path = 'gymnasium_env/piece_images2'
pdf_path = os.path.join(folder_path, 'blokus_pieces.pdf')

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

DICTIONARY = {
    '1': (0, (7, 8)), #
    '2': (0, (10, 11)), #
    'I3': (0, (0, 0)), #
    'V3': (0, (7, 10)), #
    'I4': (0, (9, 0)), #
    'L4': (0, (0, 8)), 
    'O': (0, (11, 0)),
    'T4': (0, (4, 4)),
    'Z4': (0, (14, 0)),
    'F': (0, (8, 13)),
    'I5': (0, (7, 0)),
    'L5': (0, (3, 14)),
    'N': (0, (0, 12)),
    'P': (0, (1, 4)),
    'T5': (0, (3, 0)),
    'U': (0, (4, 8)),
    'V5': (0, (7, 6)), #
    'W': (0, (11, 9)), #
    'X': (0, (12, 13)), #
    'Y': (0, (12, 7)), #
    'Z5': (0, (11, 3)) #
}

with PdfPages(os.path.join(folder_path, 'board.pdf')) as pdf:
    save_board_image_to_pdf(pdf, DICTIONARY)

for id, transform_list in BlokusPieceManager.pieces.items():
    for i, piece in enumerate(transform_list.transformations):
        with PdfPages(os.path.join(folder_path, f'{piece.idx:03d}_{id}_transformation_{i}.pdf')) as pdf:
            save_piece_image_to_pdf(pdf, piece)