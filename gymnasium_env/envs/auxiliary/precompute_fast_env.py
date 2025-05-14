from tqdm import tqdm
import numpy as np
import os
import time

from gymnasium_env.envs.auxiliary.utils import encode, decode
from gymnasium_env.envs.blokus_piece import BlokusPieceManager
from gymnasium_env.env_config import PRECOMPUTED_DIR, NEIGHBOR_POS_FILENAME, COMPUTE_ACTIONS_FILENAME

def compute_neighborhood_positions():
    """ Compute the neighborhood positions for all pieces.  """
    cnt = 0
    neighbor_idx, neighbor_pos = {}, []
    for i in range(-2, 1):
        for j in range(1, 4 - (-i // 2)):
            neighbor_idx[i, j] = cnt
            neighbor_pos.append((i, j))
            cnt += 1

    for i in range(1, 3):
        for j in range(-2, 5 - i):
            neighbor_idx[i, j] = cnt
            neighbor_pos.append((i, j))
            cnt += 1

    for i in range(-1, 2):
        neighbor_idx[3, i] = cnt
        neighbor_pos.append((3, i))
        cnt += 1

    return neighbor_idx, neighbor_pos

def compute_neighborhood_actions():
    """ Compute the neighborhood actions for all pieces.  """
    to_idx, to_pos = compute_neighborhood_positions()
    compute_actions = [[] for _ in range(1 << 22)]

    for i in tqdm(range(91)):
        piece = BlokusPieceManager.get_piece(piece_id=i)
        for h, w in piece.body:
            good_cover = True
            ids = []
            for xx, yy in piece.body:
                x, y = xx - h, yy - w
                if not ((x, y) in to_idx) and not (x == 0 and y == 0):
                    good_cover = False
                    break
                elif x != 0 or y != 0:
                    ids.append(to_idx[(x, y)])
                pass

            if not good_cover:
                continue

            for j in range(1 << 22):
                all_ids = True
                for k in ids:
                    if not (j & (1 << k)):
                        all_ids = False
                if all_ids:
                    compute_actions[j].append(encode(-h, -w, i))
                    
    compute_actions = np.array([np.array(actions, dtype=np.int16) for actions in compute_actions], dtype=object)
    return compute_actions

def load_precomputed():
    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)

    compute_actions_path = os.path.join(PRECOMPUTED_DIR, COMPUTE_ACTIONS_FILENAME)
    neighborhood_pos_path = os.path.join(PRECOMPUTED_DIR, NEIGHBOR_POS_FILENAME)
    start_time = time.time()
    if not os.path.exists(compute_actions_path):
        print("Computing neighborhood actions...")
        compute_actions = compute_neighborhood_actions()
        np.save(compute_actions_path, compute_actions)
    else:
        compute_actions = np.load(compute_actions_path, allow_pickle=True)

    if not os.path.exists(neighborhood_pos_path):
        print("Computing neighborhood positions...")
        neighbor_idx, neighbor_pos = compute_neighborhood_positions()
        np.save(neighborhood_pos_path, neighbor_pos)
    else:
        neighbor_pos = np.load(neighborhood_pos_path, allow_pickle=True)

    print(f"Precomputation time: {time.time() - start_time:.2f} seconds")
    return compute_actions, neighbor_pos