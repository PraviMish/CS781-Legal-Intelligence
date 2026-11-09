"""Download ONLY the seven official Phase-3 RR JSON files from Hugging Face.

Usage: python -m phase3.prepare --output /datasets
       [--token-env HF_TOKEN]

The resulting dataset is stored in /datasets/Phase_3_RR/{Train,Test}/.
No credentials are embedded, and datasets must not be committed to Git.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from .data import load_official

REPOSITORY = "Exploration-Lab/CS-781-Capstone"
FILES = (
    "Train/CL_train.json", "Train/IT_train.json",
    "Train/CL_dev.json", "Train/IT_dev.json",
    "Train/label_vocab.json", "Test/CL_test.json", "Test/IT_test.json",
)


def download_dataset(output, repository=REPOSITORY, token=None):
    """Fetch verified dataset paths and validate the actual schema on completion."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install huggingface_hub") from exc
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = "Phase_3_RR"
    paths = [f"{root}/{relative}" for relative in FILES]
    published = set(HfApi().list_repo_files(repository, repo_type="dataset", token=token))
    missing = set(paths) - published
    if missing:
        raise FileNotFoundError("Missing official Phase-3 files: " + ", ".join(sorted(missing)))
    for path in paths:
        hf_hub_download(repository, path, repo_type="dataset", token=token, local_dir=str(output))
    dataset_root = output / root
    documents, labels = load_official(dataset_root)
    print(f"Verified {len(documents)} splits and {len(labels)} labels under {dataset_root}")
    return dataset_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Local output parent")
    parser.add_argument("--repo", default=REPOSITORY, help="Official Hugging Face dataset ID")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing token if required")
    args = parser.parse_args(argv)
    download_dataset(args.output, args.repo, os.environ.get(args.token_env))


if __name__ == "__main__":
    main()
