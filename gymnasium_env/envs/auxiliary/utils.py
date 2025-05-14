import numpy as np
import time

def encode(x: int, y: int, idx: int) -> int:
    return ((-x) << 13) | ((-y & 7) << 10) | idx  # Mask to avoid sign extension

def decode(action):
    x = -(action >> 13)  # Extract top 3 bits and negate
    y = -((action >> 10) & 7)  # Extract next 3 bits and negate
    idx = action & 1023  # Mask to get last 10 bits (idx)
    return x, y, idx

def encode_batch(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return ((-x) << 13) | ((-y & 7) << 10) | idx  # Mask to avoid sign extension

def decode_batch(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = -(actions >> 13)
    y = -((actions >> 10) & 7)
    idx = actions & 1023
    return np.stack((x, y, idx), axis=-1)

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper