from contextlib import contextmanager
import numpy as np
import time 


def encode_board_string(board: np.ndarray) -> str:
    return ''.join(map(str, board.flatten()))

def decode_board_string(encoded_board : str, n: int) -> np.ndarray:
    board = []
    board = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            board[i, j] = int(encoded_board[i * n + j])
    return board

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        timed = end_time - start_time
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result, timed
    return wrapper

@contextmanager
def env_action_context(env, action):
    state = env.capture_state()
    try:
        new_obs, _, _, _, _ = env.step(action)
        yield new_obs  
    finally:
        env.restore_state(state)