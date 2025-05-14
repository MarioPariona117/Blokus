from pathlib import Path
import os

# Project root (always consistent, wherever it's imported)
BASE_DIR = Path(__file__).resolve().parent

assert (BASE_DIR / "src").exists(), "This is not the correct working root."

MODEL_DIR = Path(os.path.join(BASE_DIR, "models"))
MODEL_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path(os.path.join(BASE_DIR, "cache"))
CACHE_DIR.mkdir(exist_ok=True)

# ENV_DIR = Path(os.path.join(BASE_DIR, "gymnasium_env"))
# ENV_DIR.mkdir(exist_ok=True)