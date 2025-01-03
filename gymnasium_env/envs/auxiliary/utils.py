import numpy as np
import time

PIECE_IDS = ['1', '2', 'I3', 'V3', 'I4', 'L4', 'O', 'T4', 'Z4', 'F', 'I5', 'L5', 'N', 'P', 'T5', 'U', 'V5', 'W', 'X', 'Y', 'Z5']

def encode(x, y, idx):
    x *= -1
    y *= -1
    return (x << 13) + (y << 10) + idx

def decode(action):
    x = (action >> 13) * -1
    y = ((action >> 10) & 7) * -1
    idx = action & 1023
    return x, y, idx

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper