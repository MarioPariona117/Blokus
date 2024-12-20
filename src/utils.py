import numpy as np

def encode_board_string(board: np.ndarray) -> str:
    return ''.join(map(str, board.flatten()))

def decode_board_string(encoded_board : str, n: int) -> np.ndarray:
    board = []
    board = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            board[i, j] = int(encoded_board[i * n + j])
    return board