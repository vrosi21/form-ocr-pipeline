"""
digit_model.py — a dedicated single-digit classifier for the comb cells.

Comb digit cells (dob/dov/ssn/phone/emergency_phone/age + the 8 digit positions
of insurance) are pre-segmented single characters. That's a degenerate case for
EasyOCR's CRNN+CTC sequence recognizer but the natural case for a small CNN
classifier. This module:

  * collects the digit cells + their labels from the crops,
  * builds a small MNIST-style CNN,
  * provides generic train / eval loops (used for MNIST pretrain AND fine-tune),
  * reassembles a comb field from per-cell CNN predictions (for downstream eval
    and, later, the app's L-pipeline).

Letters are NOT handled here. Insurance positions 0-2 are letters -> route those
to EasyOCR with an A-Z allowlist; this classifier only does the digit cells.

The data-collection half (collect_digit_cells / digit_histogram) is pure
PIL/numpy and imports no torch, so it runs anywhere. Everything that needs torch
imports it lazily inside the function.
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image

# Comb fields whose cells are digits (insurance is included; its first 3 letter
# cells are filtered out because their label is not a digit).
DIGIT_COMB_FIELDS = ["date_of_birth", "date_of_visit", "age", "ssn",
                     "phone_number", "emergency_contact_phone", "insurance_number"]

IMG = 28  # classifier input is 28x28 to match MNIST


# --------------------------------------------------------------------------- #
# Data collection (no torch)
# --------------------------------------------------------------------------- #
def collect_digit_cells(crops_root, field_map, labels_by_pid, pids):
    """
    Return [(crop_path, digit_int), ...] for every non-empty DIGIT comb cell over
    the given pids. Uses finetune.derive_cell_chars to get the ground-truth
    character per cell; keeps only cells whose char is a digit (so insurance's
    3 letter cells are dropped automatically).
    """
    from finetune import derive_cell_chars
    out = []
    for pid in pids:
        d = os.path.join(crops_root, pid)
        row = labels_by_pid.get(pid, {})
        for field in DIGIT_COMB_FIELDS:
            cfg = field_map["fields"][field]
            cells = derive_cell_chars(field, cfg, row.get(field, ""))
            if cells is None:
                continue
            for i, ch in enumerate(cells):
                if ch == "" or not ch.isdigit():
                    continue
                p = os.path.join(d, f"{field}_{i}.png")
                if os.path.exists(p):
                    out.append((p, int(ch)))
    return out


def digit_histogram(pairs):
    """Count samples per digit class -> {0:n0, ..., 9:n9}."""
    h = {d: 0 for d in range(10)}
    for _, y in pairs:
        h[y] = h.get(y, 0) + 1
    return h


# --------------------------------------------------------------------------- #
# Preprocessing: form cell (white bg, dark ink) -> MNIST convention
# (black bg, bright ink), 28x28, MNIST-normalized.
# --------------------------------------------------------------------------- #
def cell_to_array(pil_or_path):
    """PIL/path of a form digit cell -> float32 [28,28], MNIST-like + normalized."""
    img = pil_or_path if isinstance(pil_or_path, Image.Image) else Image.open(pil_or_path)
    g = img.convert("L").resize((IMG, IMG), Image.BILINEAR)
    arr = 255.0 - np.asarray(g, dtype=np.float32)   # invert -> ink bright like MNIST
    arr = (arr / 255.0 - 0.1307) / 0.3081           # MNIST mean/std
    return arr


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_cnn():
    """Small MNIST-grade CNN (~99% on MNIST)."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28->14
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14->7
        nn.Flatten(),
        nn.Dropout(0.25),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10),
    )


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def _make_dataset(pairs, augment=False, seed=42):
    """torch Dataset over (crop_path, label) form cells with optional augmentation."""
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as T

    aug = T.RandomAffine(degrees=8, translate=(0.1, 0.1),
                         scale=(0.9, 1.1), fill=255) if augment else None

    class DS(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            img = Image.open(path).convert("L").resize((IMG, IMG), Image.BILINEAR)
            if aug is not None:
                img = aug(img)              # operates on white-bg PIL, fill=255
            arr = 255.0 - np.asarray(img, dtype=np.float32)
            arr = (arr / 255.0 - 0.1307) / 0.3081
            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # [1,28,28]
            return x, label

    return DS(pairs)


# --------------------------------------------------------------------------- #
# Generic train / eval (used for MNIST pretrain and for fine-tune)
# --------------------------------------------------------------------------- #
def train_loop(model, train_loader, epochs, lr, device,
               val_loader=None, ckpt_path=None, patience=None, log_every=100):
    """
    Train `model` with Adam + cross-entropy. If val_loader + ckpt_path are given,
    saves the best-by-val-accuracy weights and (optionally) early-stops.
    Returns a history list of {epoch, train_loss, val_acc}.
    """
    import torch
    import torch.nn as nn

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    history, best_acc, wait = [], -1.0, 0

    for ep in range(epochs):
        model.train()
        run, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run += float(loss.item()); nb += 1
            if nb % log_every == 0:
                print(f"  epoch {ep+1}/{epochs} step {nb} loss {run/nb:.4f}", flush=True)
        tr = run / max(1, nb)

        val_acc = None
        if val_loader is not None:
            val_acc = accuracy(model, val_loader, device)
        history.append({"epoch": ep + 1, "train_loss": tr, "val_acc": val_acc})
        msg = f"epoch {ep+1}/{epochs}  train_loss {tr:.4f}"
        if val_acc is not None:
            msg += f"  val_acc {100*val_acc:.2f}%"
        if val_acc is not None and val_acc > best_acc + 1e-4:
            best_acc, wait = val_acc, 0
            if ckpt_path:
                save_model(model, ckpt_path); msg += "  * best, saved"
        else:
            wait += 1
        print(msg, flush=True)
        if patience and val_loader is not None and wait >= patience:
            print(f"early stop (no val gain for {patience})", flush=True); break

    if ckpt_path and best_acc < 0:   # no val tracking -> save final
        save_model(model, ckpt_path)
    return history


def accuracy(model, loader, device):
    """Top-1 accuracy over a DataLoader."""
    import torch
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += int((pred == y).sum()); total += int(y.numel())
    return correct / max(1, total)


def evaluate(model, pairs, device, batch_size=256):
    """Per-class + overall accuracy and a confusion matrix over (path,label) pairs."""
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(_make_dataset(pairs), batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    conf = np.zeros((10, 10), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            for t, p in zip(y.numpy(), pred):
                conf[t, p] += 1
    per_class = {d: (conf[d, d] / conf[d].sum() if conf[d].sum() else None)
                 for d in range(10)}
    overall = conf.trace() / max(1, conf.sum())
    return {"overall": float(overall), "per_class": per_class, "confusion": conf}


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #
def save_model(model, path):
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path, device="cpu"):
    import torch
    model = build_cnn()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Downstream: read a comb field's digits with the CNN, assemble the string.
# --------------------------------------------------------------------------- #
def predict_comb_digits(model, crops_dir, field, cfg, device, letter_reader=None):
    """
    Reassemble a comb field using the CNN for digit cells. Empty cells are
    skipped (ocr._is_empty_cell). For insurance, cells 0-2 are letters: if a
    `letter_reader` (an EasyOCR reader) is given they're read with an A-Z
    allowlist, otherwise left blank. Returns the assembled string.
    """
    import torch
    import ocr
    n = len(cfg["cells"])
    chars = []
    for i in range(n):
        gray = ocr._load_gray(os.path.join(crops_dir, f"{field}_{i}.png"))
        if ocr._is_empty_cell(gray):
            chars.append("")
            continue
        is_letter_cell = (field == "insurance_number" and i < 3)
        if is_letter_cell:
            if letter_reader is not None:
                chars.append(ocr._ocr_char(letter_reader, gray,
                                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            else:
                chars.append("")
        else:
            arr = cell_to_array(Image.fromarray(gray))
            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                chars.append(str(int(model(x).argmax(1).item())))
    return ocr._assemble_comb(field, cfg, chars)
