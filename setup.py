from pathlib import Path
import subprocess
import sys
import os

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent

REQUIREMENTS = ROOT / "requirements.txt"
MODELS_DIR = ROOT / "models"

EMBEDDING_DIR = MODELS_DIR / "bge-m3"
RERANKER_DIR = MODELS_DIR / "bge-reranker-v2-m3"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def install_dependencies():
    """Install Python dependencies."""
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS),
        ]
    )


def download_model(model_name: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {model_name}...")
    print(f"Target directory: {local_dir}")

    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
        # ускорение и устойчивость:
        max_workers=4,
        # не качать дубликаты форматов:
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
    )
    print(f"Downloaded: {model_name}")


def download_models():
    """Download all models required at runtime."""
    download_model(
        EMBEDDING_MODEL,
        EMBEDDING_DIR,
    )

    download_model(
        RERANKER_MODEL,
        RERANKER_DIR,
    )


def setup():
    """Install dependencies and download models."""
    print("=== Installing dependencies ===")
    install_dependencies()

    print("\n=== Downloading models ===")
    download_models()

    print("\nSetup completed.")


if __name__ == "__main__":
    setup()