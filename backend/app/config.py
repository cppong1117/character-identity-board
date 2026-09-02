"""Application configuration for Character Identity Board."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os


def _default_data_dir() -> Path:
    env = os.environ.get("CIB_DATA_DIR")
    if env:
        return Path(env)
    return Path.home() / "character-identity-board-data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CIB_", env_file=".env", extra="ignore"
    )

    app_name: str = "Character Identity Board"
    version: str = "0.1.0"

    # Network
    api_host: str = "127.0.0.1"
    api_port: int = 8322
    ui_port: int = 8321

    # Storage roots (under CIB_DATA_DIR)
    data_dir: Path = _default_data_dir()
    project_dir: Path = _default_data_dir() / "projects"
    cache_dir: Path = _default_data_dir() / "cache"
    model_dir: Path = _default_data_dir() / "cache" / "models"
    log_dir: Path = _default_data_dir() / "logs"

    # Database
    db_path: Path = _default_data_dir() / "cib.sqlite3"
    # SQLAlchemy URL; default SQLite, switch to postgres for prod:
    #   postgresql+psycopg://user:pass@localhost/cib
    database_url: str = ""

    # Processing resources
    device: str = ""  # "" => auto detect cuda/cpu
    use_gpu: bool = True

    # Shot detection defaults (PySceneDetect)
    shot_detector: str = "ContentDetector"
    shot_threshold: float = 27.0
    shot_min_len_frames: int = 12

    # Tracking / detection
    track_iou_threshold: float = 0.30
    max_track_frames_lost: int = 12

    # Clustering (Discovery Mode)
    cluster_method: str = "hdbscan"
    hdbscan_min_cluster_size: int = 15
    # Face recognition engine: SFace (128-dim) or ArcFace R100 (512-dim)
    # ArcFace has 6.99x better cross-shot separation
    use_arcface: bool = False
    
    # ArcFace cosine similarity thresholds (512-dim embeddings)
    # A/B test: same-person mean=0.888, diff-person mean=0.615, separation=0.273
    arcface_identity_threshold: float = 0.75
    arcface_merge_threshold: float = 0.80
    arcface_unknown_threshold: float = 0.65
    
    # SFace cosine similarity thresholds (128-dim embeddings, legacy)
    identity_threshold: float = 0.85
    merge_threshold: float = 0.88
    unknown_threshold: float = 0.80

    # Face quality
    # SFace embeddings are less reliable below this size. The processor still
    # records such faces, but leaves them unembedded for conservative assignment.
    min_face_size_px: int = 90
    # Keep the legacy OR gate by default; a face passing either dimension is
    # accepted. Set to "both" for stricter portrait-quality benchmarking.
    face_size_gate: str = "either"
    blur_threshold: float = 28.0
    occlusion_penalty: bool = True
    # Max recommended face size (px) for stable embedding
    max_face_size_px: int = 600
    # Aggregate several high-quality observations before reference matching.
    reference_top_k: int = 5
    reference_margin: float = 0.10

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.db_path}"


settings = Settings()

# Ensure dirs exist
for d in (
    settings.data_dir,
    settings.project_dir,
    settings.cache_dir,
    settings.model_dir,
    settings.log_dir,
):
    d.mkdir(parents=True, exist_ok=True)
