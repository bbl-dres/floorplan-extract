"""Stage 3a, segmentation: the U-Net and tiled inference with flip averaging.

Two checkpoint formats load through load_model():
- v1: a bare state dict (data/model/segmenter.pt of pilot v2): six classes, softmax.
- v2: a dict {"format": "fpx-segmenter", "version": 2, "classes", "heads", "state_dict", "meta"} written by train.py:
  the same U-Net/ResNet-34 with the six classes (softmax) plus one sigmoid channel per extra head (model.HEADS:
  interior, boundary, void, text). segment() writes them to sheet.<head>_prob.
sheet.prob and sheet.label are computed exactly as in v1 for both formats.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from .config import DEFAULT
from .model import CLASSES

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
FORMAT = "fpx-segmenter"


def build_model(pretrained=True, heads=()):
    """U-Net/ResNet-34 with len(CLASSES) softmax channels followed by one sigmoid channel per extra head.
    pretrained=True starts the encoder from ImageNet weights (licence question: MODEL_CARD.md)."""
    return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, classes=len(CLASSES) + len(heads))


def normalise(x):
    return (x.float() - MEAN.to(x.device)) / STD.to(x.device)


def checkpoint(model, heads, **meta):
    """v2 checkpoint: state dict plus what is needed to rebuild and interpret the model. meta is reduced to plain JSON
    values (anything else becomes a string), so that torch.load(weights_only=True) accepts the file."""
    meta = json.loads(json.dumps(meta, default=str))
    return {"format": FORMAT, "version": 2, "arch": "smp.Unet/resnet34", "classes": list(CLASSES), "heads": list(heads),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "meta": meta}


def load_model(path, device=None):
    """Trained segmenter for CPU inference, v1 or v2 checkpoint. The model carries fpx_version, fpx_heads, fpx_meta."""
    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and obj.get("format") == FORMAT:
        if list(obj["classes"]) != list(CLASSES):
            raise ValueError(f"{path}: classes {obj['classes']} differ from {CLASSES}")
        version, heads, state, meta = int(obj["version"]), list(obj["heads"]), obj["state_dict"], obj.get("meta", {})
    else:                                                   # v1: a bare state dict, six classes, no heads
        version, heads, state, meta = 1, [], obj, {}
    model = build_model(pretrained=False, heads=heads)
    model.load_state_dict(state)
    model.fpx_version, model.fpx_heads, model.fpx_meta = version, heads, meta
    # device: "cpu" (default) or "cuda"; FPX_DEVICE sets it for scripts that do not pass it (e.g. benchmarks on a GPU pod)
    device = device or os.environ.get("FPX_DEVICE", "cpu")
    return model.to(device).eval()


@torch.no_grad()
def segment(sheet, model, cfg=DEFAULT):
    tile, overlap = cfg.seg_tile, cfg.seg_overlap
    dev = next(model.parameters(), torch.empty(0)).device      # parameterless stand-ins (tests) run on CPU
    heads = list(getattr(model, "fpx_heads", ()))
    k = len(CLASSES)
    img = sheet.img
    H, W = img.shape[:2]
    acc = np.zeros((len(CLASSES), H, W), np.float32)
    hacc = np.zeros((len(heads), H, W), np.float32) if heads else None
    wsum = np.zeros((H, W), np.float32)
    ramp = np.minimum(np.arange(tile) + 1, overlap) / overlap
    win = np.minimum.outer(np.minimum(ramp, ramp[::-1]), np.minimum(ramp, ramp[::-1])).astype(np.float32)
    step = tile - overlap
    for y in range(0, max(H - overlap, 1), step):
        for x in range(0, max(W - overlap, 1), step):
            t = img[y:y + tile, x:x + tile]
            h, w = t.shape[:2]
            ph, pw = (-h) % 32, (-w) % 32
            t = np.pad(t, ((0, ph), (0, pw), (0, 0)), constant_values=255)
            x_ = normalise(torch.from_numpy(t).permute(2, 0, 1)[None].to(dev))
            p = 0
            flips = ([], [3], [2], [2, 3]) if cfg.seg_tta else ([],)
            for dims in flips:                                 # test-time augmentation: average over flips
                xf = torch.flip(x_, dims) if dims else x_
                if heads:                                      # v2: softmax over the classes, sigmoid per head
                    out = model(xf)
                    pf = torch.cat([torch.softmax(out[:, :k], 1), torch.sigmoid(out[:, k:])], 1)
                else:                                          # v1: unchanged
                    pf = torch.softmax(model(xf), 1)
                p = p + (torch.flip(pf, dims) if dims else pf)
            p = (p / len(flips))[0, :, :h, :w].cpu().numpy()
            acc[:, y:y + h, x:x + w] += p[:k] * win[:h, :w]
            if heads:
                hacc[:, y:y + h, x:x + w] += p[k:] * win[:h, :w]
            wsum[y:y + h, x:x + w] += win[:h, :w]
    sheet.prob = acc / np.maximum(wsum, 1e-6)
    sheet.label = sheet.prob.argmax(0).astype(np.uint8)
    for i, name in enumerate(heads):
        if hasattr(sheet, f"{name}_prob"):
            setattr(sheet, f"{name}_prob", hacc[i] / np.maximum(wsum, 1e-6))
        else:
            warnings.warn(f"segmenter head '{name}' has no Sheet field; ignored")
    return sheet.label
