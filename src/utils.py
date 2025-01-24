from contextlib import contextmanager
import numpy as np
import time 
from gymnasium_env.envs import BlokusEnv, BlokusAction
import inspect
import torch


def encode_board_string(board: np.ndarray) -> str:
    return ''.join(map(str, board.flatten()))

def decode_board_string(encoded_board : str, n: int) -> np.ndarray:
    board = []
    board = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            board[i, j] = int(encoded_board[i * n + j])
    return board

def encode_board_bytes(board: np.ndarray) -> bytes:
    flattened_board = board.flatten()
    packed_values = np.zeros((len(flattened_board) + 3) // 4, dtype=np.uint8)  # Allocate bytes
    for i, value in enumerate(flattened_board):
        packed_values[i // 4] |= (value & 0b11) << (2 * (i % 4))  # Pack each 2-bit value
    return packed_values.tobytes()

def decode_board_bytes(encoded_board: bytes, n: int) -> np.ndarray:
    encoded_board = np.frombuffer(encoded_board, dtype=np.uint8)
    unpacked_values = []
    for byte in encoded_board:
        for i in range(4):  # Each byte has up to 4 values (2 bits each)
            unpacked_values.append((byte >> (2 * i)) & 0b11)  # Extract 2 bits
    return np.array(unpacked_values[:n * n]).reshape((n, n))

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
def env_action_context(env: BlokusEnv, action: BlokusAction):
    state = env.capture_state()
    try:
        new_obs, _, _, _, _ = env.step(action.action_id)
        yield new_obs  
    finally:
        env.restore_state(state)

def dbg(variable):
    frame = inspect.currentframe().f_back
    try:
        var_name = next(name for name, val in frame.f_locals.items() if val is variable)
    except StopIteration:
        var_name = "<unknown>"  # Fallback if no variable name is found
    print(f"{var_name} = {repr(variable)}")

def dbg_tensor(tensor):
    assert isinstance(tensor, torch.Tensor)
    frame = inspect.currentframe().f_back
    try:
        var_name = next(name for name, val in frame.f_locals.items() if val is tensor)
    except StopIteration:
        var_name = "<unknown>"  # Fallback if no variable name is found
    print(f"{var_name} = shape = {tensor.shape}")