"""Download and convert published bone-suppression weights to this repo's PyTorch format.

After conversion the checkpoints work with every tool in src/bone_suppression
(infer.py, evaluate.py), exactly like the models we train ourselves.

| name      | source                                                     | trained on              | input   | needs        |
|-----------|------------------------------------------------------------|-------------------------|---------|--------------|
| rajaraman | ResNet-BS, Rajaraman et al. 2021 (resnet_bs.h5, Keras)     | JSRT/BSE-JSRT (augm.)   | 256x256 | h5py         |
| hmchuong  | AE of Gusarev et al. 2017, re-implemented by hmchuong (TF1)| JSRT/BSE-JSRT (augm.)   | 1024x1024 | tensorflow (one-off) |

The hmchuong model was trained on NEGATIVE images (lungs bright, bones dark); the converted
checkpoint stores invert_input=True so infer.py flips VinDr images before/after automatically.

Usage
  python scripts/convert_pretrained.py --which rajaraman          # downloads + converts
  python scripts/convert_pretrained.py --which hmchuong            # needs: pip install tensorflow-cpu
  python scripts/convert_pretrained.py --which all

Output: checkpoints/pretrained/<name>.pt   (git-ignored: share it via Drive / GitHub Releases)
Note: the Rajaraman repository states no license -> use for research/education and cite the paper.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bone_suppression.models import build_model  # noqa: E402

EXTERNAL = ROOT / "data" / "external"
OUT_DIR = ROOT / "checkpoints" / "pretrained"

# Pinned to the commits that were verified (see docs/PLAN.md).
RAJARAMAN_URL = ("https://raw.githubusercontent.com/sivaramakrishnan-rajaraman/CXR-bone-suppression/"
                 "fca7025e183852065b39f0774ab9cc307dd90d28/resnet_bs.h5")
HMCHUONG_BASE = ("https://raw.githubusercontent.com/hmchuong/ML-BoneSuppression/"
                 "0f9de5ff4cc1f793e27f9db52af35e631614a4c1/model/")
HMCHUONG_FILES = ["model.index", "model.data-00000-of-00001", "model.meta"]


def download(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}")
    urllib.request.urlretrieve(url, dst)
    return dst


def save(model, out: Path, meta: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {**meta, "state_dict": model.state_dict()}
    torch.save(ckpt, out)
    print(f"saved {out}")


def convert_rajaraman(h5_path: Path, out: Path) -> None:
    """Keras ResNet-BS (weights only) -> ResNetBS(n_blocks=16), direct output."""
    import h5py

    model = build_model("resnet_bs", residual=False, n_blocks=16)
    bb = model.backbone
    # Keras layer order: conv2d_1 = head, (conv2d_2k, conv2d_2k+1) = res-block k, conv2d_34 = body tail, conv2d_35 = output
    targets = [bb.head]
    for blk in bb.body[:16]:
        targets += [blk.body[0], blk.body[2]]
    targets += [bb.body[16], bb.tail]
    with h5py.File(h5_path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        for i, conv in enumerate(targets, start=1):
            name = f"conv2d_{i}"
            k = np.asarray(g[name][name]["kernel:0"])  # (kh, kw, in, out)
            b = np.asarray(g[name][name]["bias:0"])
            assert k.shape[::-1][:2] == tuple(conv.weight.shape[:2]), (name, k.shape, conv.weight.shape)
            conv.weight.data = torch.from_numpy(k.transpose(3, 2, 0, 1).copy())
            conv.bias.data = torch.from_numpy(b.copy())
    save(model, out, {
        "model_name": "resnet_bs", "model_kwargs": {"n_blocks": 16}, "residual": False, "data_size": 256,
        "source": "Rajaraman et al., Diagnostics 2021 - github.com/sivaramakrishnan-rajaraman/CXR-bone-suppression",
        "preprocess": "grayscale / 255, resized to 256x256",
    })


def convert_hmchuong(ckpt_prefix: Path, out: Path) -> None:
    """TF1 checkpoint (tied-weight AE) -> HmchuongAE, direct output."""
    try:
        import tensorflow as tf
    except ImportError:
        sys.exit("Converting the hmchuong model needs TensorFlow once: pip install tensorflow-cpu")
    reader = tf.train.load_checkpoint(str(ckpt_prefix))
    ws = [reader.get_tensor(f"encoder/layer/{i}/W") for i in range(3)]  # (5, 5, in, out)
    model = build_model("hmchuong_ae", residual=False)
    bb = model.backbone
    for conv, w in zip(bb.enc, ws):
        conv.weight.data = torch.from_numpy(w.transpose(3, 2, 0, 1).copy())
    for deconv, w in zip(bb.dec, reversed(ws)):  # decoder re-uses encoder kernels in reverse order
        deconv.weight.data = torch.from_numpy(w.transpose(3, 2, 0, 1).copy())
    save(model, out, {
        "model_name": "hmchuong_ae", "model_kwargs": {}, "residual": False, "data_size": 1024,
        "source": "Gusarev et al. 2017 AE, implementation github.com/hmchuong/ML-BoneSuppression (MIT)",
        "preprocess": "grayscale / 255, 1024x1024, NEGATIVE images (lungs bright)",
        "invert_input": True,  # trained on inverted JSRT images -> infer.py flips polarity automatically
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", default="all", choices=["rajaraman", "hmchuong", "all"])
    ap.add_argument("--rajaraman-h5", default=None, help="local resnet_bs.h5 (skip download)")
    ap.add_argument("--hmchuong-ckpt", default=None, help="local TF checkpoint prefix, e.g. .../model/model")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    if args.which in ("rajaraman", "all"):
        h5 = Path(args.rajaraman_h5) if args.rajaraman_h5 else download(RAJARAMAN_URL, EXTERNAL / "rajaraman" / "resnet_bs.h5")
        convert_rajaraman(h5, out_dir / "rajaraman_resnet_bs.pt")
    if args.which in ("hmchuong", "all"):
        if args.hmchuong_ckpt:
            prefix = Path(args.hmchuong_ckpt)
        else:
            for name in HMCHUONG_FILES:
                download(HMCHUONG_BASE + name, EXTERNAL / "hmchuong" / name)
            prefix = EXTERNAL / "hmchuong" / "model"
        convert_hmchuong(prefix, out_dir / "hmchuong_ae.pt")


if __name__ == "__main__":
    main()
