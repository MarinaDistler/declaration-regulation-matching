from pathlib import Path
import subprocess
import sys


EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def install_dependencies(requirements_path=None):
    """Install project dependencies from requirements.txt."""
    if requirements_path is None:
        root = Path(__file__).resolve().parent
        requirements_path = root / "requirements.txt"

    requirements_path = Path(requirements_path)

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ]
    )


def download_model(model_name, local_dir, revision=None):
    """Download a Hugging Face model to local_dir."""
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_name}")
    print(f"Destination: {local_dir}")

    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
        revision=revision,
    )

    print(f"Finished: {model_name}\n")


def download_models(
    models_dir,
    download_embedding=True,
    download_reranker=True,
    embedding_revision=None,
    reranker_revision=None,
):
    """Download models required by the project."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if download_embedding:
        download_model(
            model_name=EMBEDDING_MODEL,
            local_dir=models_dir / "bge-m3",
            revision=embedding_revision,
        )

    if download_reranker:
        download_model(
            model_name=RERANKER_MODEL,
            local_dir=models_dir / "bge-reranker-v2-m3",
            revision=reranker_revision,
        )


def setup(
    models_dir=None,
    install_deps=True,
    download_embedding=True,
    download_reranker=True,
):
    """Install dependencies and download required models."""
    root = Path(__file__).resolve().parent

    if models_dir is None:
        models_dir = root / "models"

    if install_deps:
        install_dependencies(root / "requirements.txt")

    download_models(
        models_dir=models_dir,
        download_embedding=download_embedding,
        download_reranker=download_reranker,
    )


if __name__ == "__main__":
    setup()