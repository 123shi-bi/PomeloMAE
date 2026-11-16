import os
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset


VIEW_SUFFIXES = ("a", "b", "c", "d")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_fruit_views(split_root, view_suffixes=VIEW_SUFFIXES):
    """Yield each fruit once, with image paths in the declared view order."""
    if not split_root.is_dir():
        raise FileNotFoundError(f"Dataset split directory does not exist: {split_root}")

    seen = set()
    for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        for fruit_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            fruit_id = fruit_dir.name
            if fruit_id in seen:
                raise ValueError(f"Duplicate fruit ID {fruit_id!r} in {split_root}")
            images = [path for path in fruit_dir.iterdir()
                      if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
            by_stem = {path.stem: path for path in images}
            expected = [f"{fruit_id}_{suffix}" for suffix in view_suffixes]
            if len(images) != len(expected) or by_stem.keys() != set(expected):
                found = ", ".join(sorted(path.name for path in images)) or "none"
                raise ValueError(f"Expected one image for each of {', '.join(expected)} "
                                 f"in {fruit_dir}. Found {found}.")
            seen.add(fruit_id)
            yield class_dir.name, fruit_id, [by_stem[stem] for stem in expected]
    if not seen:
        raise ValueError(f"No complete fruit groups found in {split_root}")


class FourViewPomeloDataset(Dataset):
    """Group one pomelo's a/b/c/d images as a single pretraining sample.

    Expected layout:
        root/train/class_name/pomelo_id/pomelo_id_a.jpg
        root/train/class_name/pomelo_id/pomelo_id_b.jpg
        root/train/class_name/pomelo_id/pomelo_id_c.jpg
        root/train/class_name/pomelo_id/pomelo_id_d.jpg
    """

    def __init__(self, root, transform=None, view_suffixes=VIEW_SUFFIXES):
        self.root = Path(root)
        self.transform = transform
        self.view_suffixes = tuple(view_suffixes)
        fruits = list(iter_fruit_views(self.root, self.view_suffixes))
        self.classes = [path.name for path in sorted(self.root.iterdir()) if path.is_dir()]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = [(views, self.class_to_idx[class_name], fruit_id)
                        for class_name, fruit_id, views in fruits]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        view_paths, label, pomelo_id = self.samples[index]
        views = []
        for path in view_paths:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                views.append(img)
        return torch.stack(views, dim=0), label, pomelo_id

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(root={os.fspath(self.root)!r}, "
            f"samples={len(self.samples)}, classes={self.classes}, "
            f"views={self.view_suffixes})"
        )
