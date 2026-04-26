"""Manage the MMS-1B-FL102 model snapshot under ComfyUI/models/mms/."""

from __future__ import annotations

from pathlib import Path

MODEL_REPO = "facebook/mms-1b-fl102"


def _models_root() -> Path:
    """Return ComfyUI/models/mms (creating it if needed)."""
    try:
        import folder_paths  # provided by ComfyUI at runtime
        base = Path(folder_paths.models_dir)
    except Exception:
        here = Path(__file__).resolve()
        repo_root = here.parent.parent
        if repo_root.parent.name == "custom_nodes":
            base = repo_root.parent.parent / "models"
        else:
            base = repo_root / "models"
    target = base / "mms"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _local_dir() -> Path:
    return _models_root() / MODEL_REPO.replace("/", "__")


def _is_populated(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def ensure_model() -> str:
    """Ensure MMS-1B-FL102 is present locally; return the local directory path."""
    local_dir = _local_dir()
    if _is_populated(local_dir):
        return str(local_dir)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. pip install huggingface_hub"
        ) from e

    print(f"[MMS-Aligner] Downloading {MODEL_REPO} to {local_dir} ...")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print("[MMS-Aligner] Download complete.")

    if not _is_populated(local_dir):
        raise RuntimeError(
            f"Model download to {local_dir} did not produce expected files. "
            "Check network and Hugging Face access."
        )
    return str(local_dir)
