"""Paired dataset: X-ray (input) -> bone-suppressed X-ray (target)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common.io import imread_gray


class PairedXrayDataset(Dataset):
    """Reads <root>/source/<id>.png and <root>/target/<id>.png for one split.

    Augmentations are applied identically to input and target:
    small affine (rotation / scale / shift), horizontal flip and gamma.
    `crop` > 0 returns random crops (needed for heavy full-resolution models
    such as ResNet-BS); evaluation always uses the full image.
    """

    def __init__(self, root: str | Path, split: str, augment: bool = False,
                 crop: int = 0, cache: bool = True):
        self.root = Path(root)
        df = pd.read_csv(self.root / "splits.csv", dtype={"id": str})
        self.ids = df.loc[df["split"] == split, "id"].tolist()
        if not self.ids:
            raise ValueError(f"No images for split '{split}' in {self.root}/splits.csv")
        self.augment = augment
        self.crop = crop
        self.cache = {} if cache else None

    def __len__(self) -> int:
        return len(self.ids)

    def _load(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        if self.cache is not None and pid in self.cache:
            return self.cache[pid]
        x = imread_gray(self.root / "source" / f"{pid}.png")
        y = imread_gray(self.root / "target" / f"{pid}.png")
        if self.cache is not None:
            self.cache[pid] = (x, y)
        return x, y

    @staticmethod
    def _random_affine(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        h, w = x.shape
        angle = rng.uniform(-5, 5)
        scale = rng.uniform(0.92, 1.08)
        tx, ty = rng.uniform(-0.04, 0.04, size=2) * (w, h)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        m[:, 2] += (tx, ty)
        kw = dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return cv2.warpAffine(x, m, (w, h), **kw), cv2.warpAffine(y, m, (w, h), **kw)

    def __getitem__(self, idx: int):
        pid = self.ids[idx]
        x, y = self._load(pid)
        if self.augment:
            rng = np.random.default_rng()
            x, y = self._random_affine(x, y, rng)
            if rng.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            g = rng.uniform(0.8, 1.25)
            x, y = np.clip(x, 0, 1) ** g, np.clip(y, 0, 1) ** g
        if self.crop and self.crop < x.shape[0]:
            rng = np.random.default_rng()
            i = rng.integers(0, x.shape[0] - self.crop + 1)
            j = rng.integers(0, x.shape[1] - self.crop + 1)
            x = x[i:i + self.crop, j:j + self.crop]
            y = y[i:i + self.crop, j:j + self.crop]
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None]
        return x, y, pid
