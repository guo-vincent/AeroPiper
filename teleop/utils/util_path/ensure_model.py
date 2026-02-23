import urllib.request
from pathlib import Path
from .path_config import TELEOP_ROOT

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

def ensure_model(model_name: str = "pose_landmarker_full.task") -> Path:
    """
    Checks if the specified MediaPipe model exists in teleop/models/.
    Downloads it if missing.
    """
    model_dir = TELEOP_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / model_name
    
    if not model_path.exists():
        print(f"[INFO] Downloading {model_name} (~29 MB) to {model_path} ...")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print("[INFO] Download complete.")
        
    return model_path