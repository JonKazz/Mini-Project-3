"""Download Zalando Fashion-MNIST from Kaggle and load the images.

Dataset: https://www.kaggle.com/datasets/zalando-research/fashionmnist

The Kaggle copy ships either the original IDX files or label-plus-pixels CSVs.
This script accepts both, then writes NumPy arrays under data/.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

DATASET = "zalando-research/fashionmnist"
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data"

CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049


def download(slug: str = DATASET) -> Path:
    """Download the dataset and return the local folder Kaggle Hub cached."""
    import kagglehub

    return Path(kagglehub.dataset_download(slug))


def _open_binary(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def read_idx_images(path: Path) -> np.ndarray:
    with _open_binary(path) as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != IMAGE_MAGIC:
            raise ValueError(f"{path.name} is not an IDX image file (magic {magic})")
        pixels = np.frombuffer(handle.read(), dtype=np.uint8)
    return pixels.reshape(count, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with _open_binary(path) as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != LABEL_MAGIC:
            raise ValueError(f"{path.name} is not an IDX label file (magic {magic})")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"{path.name} has {labels.size} labels, header says {count}")
    return labels.copy()


def read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Fashion-MNIST CSV: first column is the label, the rest are pixels."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    labels = raw[:, 0].astype(np.uint8)
    images = raw[:, 1:].astype(np.uint8).reshape(-1, 28, 28)
    return images, labels


def _data_files(folder: Path) -> list[Path]:
    return [path for path in folder.rglob("*") if path.is_file() and not path.name.startswith(".")]


def load(folder: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {'train': (images, labels), 'test': (images, labels)}."""
    files = _data_files(folder)
    by_name = {path.name.lower(): path for path in files}

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    idx_pairs = {
        "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
        "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
    }
    for split, (image_stem, label_stem) in idx_pairs.items():
        images = _match(by_name, image_stem)
        labels = _match(by_name, label_stem)
        if images and labels:
            splits[split] = (read_idx_images(images), read_idx_labels(labels))

    if splits:
        return splits

    csvs = [path for path in files if path.suffix.lower() == ".csv"]
    for path in csvs:
        name = path.name.lower()
        if "train" in name:
            splits["train"] = read_csv(path)
        elif "test" in name or "t10k" in name:
            splits["test"] = read_csv(path)

    if not splits:
        names = ", ".join(path.name for path in files) or "(empty folder)"
        raise FileNotFoundError(f"no Fashion-MNIST files in {folder}: {names}")
    return splits


def _match(by_name: dict[str, Path], stem: str) -> Path | None:
    for suffix in ("", ".gz"):
        path = by_name.get(stem + suffix)
        if path is not None:
            return path
    return None


def save(splits: dict[str, tuple[np.ndarray, np.ndarray]], out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, (images, labels) in splits.items():
        np.save(out_dir / f"{split}_images.npy", images)
        np.save(out_dir / f"{split}_labels.npy", labels)
    return out_dir


def pull(slug: str = DATASET, out_dir: Path = OUT_DIR):
    """Download Fashion-MNIST and save train/test arrays. Returns the loaded splits."""
    folder = download(slug)
    print(f"downloaded to {folder}")
    splits = load(folder)
    saved = save(splits, out_dir)
    for split, (images, labels) in splits.items():
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        print(f"{split}: images {images.shape} {images.dtype}, labels {labels.shape}")
        print("  " + ", ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(len(CLASS_NAMES))))
    print(f"saved arrays in {saved}")
    return splits


if __name__ == "__main__":
    pull()
