from pathlib import Path
import os

# Project root (always consistent, wherever it's imported)
BASE_DIR = Path(__file__).resolve().parent

assert (BASE_DIR / "envs").exists(), "This is not the correct working root."

PRECOMPUTED_DIR = Path(os.path.join(BASE_DIR, "envs", "precomputed"))

NEIGHBOR_POS_FILENAME = "neighbor_pos.npy"
COMPUTE_ACTIONS_FILENAME = "compute_actions.npy"
