from pathlib import Path
import sys


def fodlaai_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_on_path() -> Path:
    root = fodlaai_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
