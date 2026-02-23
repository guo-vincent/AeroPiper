import sys
from pathlib import Path

# Calculate roots relative to this file
_UTIL_PATH_DIR = Path(__file__).resolve().parent
UTILS_DIR      = _UTIL_PATH_DIR.parent
TELEOP_ROOT    = UTILS_DIR.parent
REPO_ROOT      = TELEOP_ROOT.parent

def setup_sys_path():
    """Injects the repo and teleop roots into sys.path if not present."""
    for p in (REPO_ROOT, TELEOP_ROOT):
        if str(p) not in sys.path:
            # prioritize local modules over globals
            sys.path.insert(0, str(p))

# Export paths
__all__ = ["REPO_ROOT", "TELEOP_ROOT", "setup_sys_path"]