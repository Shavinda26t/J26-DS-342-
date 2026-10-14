import os
from pathlib import Path

class Settings:
    PROJECT_NAME: str = "HDD Failure Prediction & Self-Healing Backend"
    API_V1_STR: str = "/api/v1"
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", 8000))
    
    @property
    def HDD_PROJECT_ROOT(self) -> Path:
        env_root = os.getenv("HDD_PROJECT_ROOT")
        if env_root and Path(env_root).exists():
            return Path(env_root).resolve()
        
        # Infer relative to app directory (app/config.py -> app -> 05_backend -> project root)
        inferred_root = Path(__file__).resolve().parent.parent.parent
        if (inferred_root / "03_component_3_causal_xai").exists():
            return inferred_root
        
        # Development fallback if necessary
        fallback_root = Path(r"D:\Research\research_project_v1")
        if fallback_root.exists():
            return fallback_root
            
        return inferred_root

settings = Settings()

