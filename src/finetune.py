"""
finetune.py — L1: fine-tune the EasyOCR recognizer on this form's handwriting.

L1 = the SAME L0 pipeline (allow-lists, empty-cell skip, comb assembly, nb-02
checkboxes) but driven by a recognizer fine-tuned on our crops, instead of the
stock pretrained weights. Nothing about cropping or the field logic changes.

This module has two halves:

  (A) Dataset prep (no torch needed — pure PIL/numpy, runs anywhere):
      split_pids()            seeded 80/20 train/val split over the 100 forms
      derive_cell_chars()     ground-truth char per comb cell (right-aligned age)
      iter_base_samples()     (crop_path, label, field, pid, cell) over train pids
      augment()               mild, handwriting-appropriate augmentation
      build_recog_dataset()   materialize data/train_recog/{images,labels.csv}

  (B) Training (imports torch + easyocr lazily, so (A) stays importable on any
      machine):
      train_recognizer()      CTC fine-tune of reader.recognizer (quantize=False!)

Why quantize=False: on CPU, easyocr.Reader builds a dynamically-quantized,
inference-only recognizer. You cannot backprop through it, and you cannot load a
float fine-tuned state-dict back into it. So both training here and inference in
ocr.get_finetuned_reader() build the Reader with quantize=False.

Data hygiene: only the TRAIN pids are ever augmented/trained on. The validation
pids and the crops the L0 eval reads are never touched.
"""

from __future__ import annotations

import csv
import json
import os
import random

import numpy as np
from PIL import Image

# Separator chars per comb field (mirrors ocr.COMB_SEP). Imported lazily so this
# module doesn't hard-depend on import order; falls back to a local copy.
try:
    from ocr import COMB_SEP  # type: ignore
except Exception:  # pragma: no cover - fallback if ocr isn't on path yet
    COMB_SEP = {
        "date_of_birth": "/", "date_of_visit": "/", "ssn": "-",
        "phone_number": "-", "emergency_contact_phone": "-",
        "insurance_number": "", "age": "",
    }


# --------------------------------------------------------------------------- #
# (A) Dataset prep
# --------------------------------------------------------------------------- #
def split_pids(pids, train_frac: float = 0.8, seed: int = 42):
    """Deterministic train/val split. Returns (train_sorted, val_sorted)."""
    pids = sorted(pids)
    shuffled = pids[:]
    random.Random(seed).shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_frac))
    return sorted(shuffled[:n_train]), sorted(shuffled[n_train:])


def save_split(path: str, train, val, seed: int = 42, train_frac: float = 0.8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "train_frac": train_frac,
                   "train": list(train), "val": list(val)}, fh, indent=1)


def load_split(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return d["train"], d["val"]


def derive_cell_chars(field: str, cfg: dict, label: str):
    """
    Ground-truth character for each comb cell, or None if the label can't be
    cleanly mapped (then we skip that field for this form rather than guess).

      age (right-aligned, NNN): "29" -> ["", "2", "9"];  "4" -> ["", "", "4"]
      others (fixed): strip separators, must equal n_cells exactly.
    """
    label = str(label).strip()
    n = len(cfg["cells"])
    if field == "age":
        raw = "".join(ch for ch in label if ch.isdigit())
        if not raw or len(raw) > n:
            return None
        return [""] * (n - len(raw)) + list(raw)
    sep = COMB_SEP.get(field, "")
    raw = label.replace(sep, "") if sep else label
    raw = raw.strip()
    if len(raw) != n:
        return None
    return list(raw)


def iter_base_samples(crops_root: str, field_map: dict, labels_by_pid: dict, pids,
                      text_only: bool = False):
    """
    Yield (crop_path, label, field, pid, cell_idx) for every trainable crop:
      text_box  -> the field crop, label = the field's text (skip empty)
      comb      -> each non-empty cell crop, label = that single character
    Checkbox fields are skipped (not OCR'd). With text_only=True, comb fields are
    skipped too: digits are handled by the dedicated CNN (src/digit_model.py) and
    insurance letters fall back to pretrained EasyOCR, so EasyOCR fine-tunes on
    multi-character text only — avoiding the single-char-digit imbalance that
    previously hurt chief_complaint/insurance.
    """
    for pid in pids:
        d = os.path.join(crops_root, pid)
        row = labels_by_pid.get(pid, {})
        for fname, cfg in field_map["fields"].items():
            t = cfg["type"]
            if t == "checkbox":
                continue
            if text_only and t == "comb":
                continue
            if t == "text_box":
                lab = str(row.get(fname, "")).strip()
                if not lab:
                    continue
                p = os.path.join(d, f"{fname}.png")
                if os.path.exists(p):
                    yield (p, lab, fname, pid, None)
            elif t == "comb":
                cells = derive_cell_chars(fname, cfg, row.get(fname, ""))
                if cells is None:
                    continue
                for i, ch in enumerate(cells):
                    if ch == "":
                        continue
                    p = os.path.join(d, f"{fname}_{i}.png")
                    if os.path.exists(p):
                        yield (p, ch, fname, pid, i)


def augment(img: Image.Image, rng: random.Random, npr: np.random.RandomState) -> Image.Image:
    """
    Mild augmentation suited to scanned handwriting on a white field.
    Rotation, small scale jitter + recentre, brightness, and light gaussian
    noise. Kept gentle so labels stay legible (this is character-level data).
    """
    img = img.convert("L")
    w, h = img.size

    # small rotation, white fill
    angle = rng.uniform(-5.0, 5.0)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)

    # scale jitter, then recentre on a white canvas back to (w, h)
    s = rng.uniform(0.9, 1.1)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    scaled = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (w, h), 255)
    canvas.paste(scaled, ((w - nw) // 2, (h - nh) // 2))
    img = canvas

    # brightness + gaussian noise
    arr = np.asarray(img).astype(np.float32)
    arr *= rng.uniform(0.85, 1.15)
    arr += npr.normal(0.0, 6.0, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


def _safe_name(field: str, pid: str, cell, aug: int) -> str:
    base = field if cell is None else f"{field}_{cell}"
    return f"{pid}__{base}__aug{aug}.png"


def build_recog_dataset(crops_root: str, field_map: dict, labels_by_pid: dict,
                        train_pids, out_dir: str, aug_factor: int = 4,
                        seed: int = 42, verbose: bool = True,
                        text_only: bool = False) -> dict:
    """
    Materialize a self-contained recognition training set for the TRAIN pids:

      out_dir/images/<pid>__<field>[_<cell>]__aug<k>.png   (aug0 = original)
      out_dir/labels.csv  with columns filename,label,field,pid,aug

    aug_factor = total images per base crop (1 original + aug_factor-1 augmented).
    Returns a small stats dict. Idempotent: clears out_dir/images first.
    """
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    # clean any prior run so counts don't accumulate
    for f in os.listdir(img_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(img_dir, f))

    rng = random.Random(seed)
    npr = np.random.RandomState(seed)
    manifest_path = os.path.join(out_dir, "labels.csv")

    n_base = 0
    n_written = 0
    per_field: dict[str, int] = {}
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["filename", "label", "field", "pid", "aug"])
        for src, label, field, pid, cell in iter_base_samples(
                crops_root, field_map, labels_by_pid, train_pids,
                text_only=text_only):
            n_base += 1
            per_field[field] = per_field.get(field, 0) + 1
            base_img = Image.open(src).convert("L")
            for k in range(max(1, aug_factor)):
                out_img = base_img if k == 0 else augment(base_img, rng, npr)
                fn = _safe_name(field, pid, cell, k)
                out_img.save(os.path.join(img_dir, fn))
                wr.writerow([fn, label, field, pid, k])
                n_written += 1

    stats = {"train_forms": len(list(train_pids)), "base_samples": n_base,
             "aug_factor": aug_factor, "images_written": n_written,
             "per_field": per_field, "out_dir": out_dir}
    if verbose:
        print(f"train_recog: {n_base} base crops x{aug_factor} "
              f"-> {n_written} images in {img_dir}")
    return stats


def load_recog_pairs(out_dir: str):
    """Read the manifest -> [(abs_image_path, label), ...] for training."""
    img_dir = os.path.join(out_dir, "images")
    pairs = []
    with open(os.path.join(out_dir, "labels.csv"), "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pairs.append((os.path.join(img_dir, row["filename"]), row["label"]))
    return pairs


# --------------------------------------------------------------------------- #
# (B) Training — CTC fine-tune of the EasyOCR recognizer
# --------------------------------------------------------------------------- #
def train_recognizer(out_dir: str, ckpt_path: str, epochs: int = 30,
                     lr: float = 1e-5, batch_size: int = 32, imgH: int = 64,
                     imgW: int = 320, gpu: bool = False, seed: int = 42,
                     smoke: bool = False, log_every: int = 50,
                     val_frac: float = 0.1, patience: int = 5) -> dict:
    """
    Fine-tune the EasyOCR recognizer with CTC on the materialized train_recog set
    and save the recognizer state-dict to ckpt_path.

    Mirrors easyocr.recognition.recognizer_predict exactly so train/inference
    preprocessing match: AlignCollate(imgH=64, keep_ratio_with_pad=True),
    model(image, dummy_text) -> logits [B, T, C], CTC over converter.encode().

    smoke=True does a tiny 1-epoch/5-batch dry run to verify the loop end to end.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    import easyocr
    from easyocr.recognition import AlignCollate

    torch.manual_seed(seed)
    device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"

    # IMPORTANT: quantize=False -> trainable float recognizer (see module docstring)
    reader = easyocr.Reader(["en"], gpu=gpu, quantize=False)
    model = reader.recognizer.to(device)
    converter = reader.converter
    valid_chars = set(converter.dict.keys())

    pairs = load_recog_pairs(out_dir)
    kept = [(p, l) for (p, l) in pairs if l and all(c in valid_chars for c in l)]
    dropped = len(pairs) - len(kept)
    if not kept:
        raise RuntimeError("No trainable samples (all labels had OOV chars?).")

    batch_max_length = int(imgW / 10)

    # Group all augmentations of one base crop together, then hold out a small
    # validation slice for model selection. Grouping by base prevents the same
    # crop's augmentations from straddling train and val.
    def _base_key(path):
        return os.path.basename(path).rsplit("__aug", 1)[0]

    bases = {}
    for p, l in kept:
        bases.setdefault(_base_key(p), []).append((p, l))
    base_keys = sorted(bases)
    random.Random(seed).shuffle(base_keys)
    n_val = 1 if smoke else max(1, int(round(len(base_keys) * val_frac)))
    val_items   = [it for k in base_keys[:n_val] for it in bases[k]]
    train_items = [it for k in base_keys[n_val:] for it in bases[k]]

    class CropDS(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            return Image.open(path).convert("L"), label

    align = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)

    def collate(batch):
        return align([b[0] for b in batch]), [b[1] for b in batch]

    train_loader = DataLoader(CropDS(train_items), batch_size=batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(CropDS(val_items), batch_size=batch_size,
                            shuffle=False, collate_fn=collate, num_workers=0)

    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def batch_cost(tensors, labels):
        # zero_infinity=True forces the native (non-cuDNN) CTC kernel, so every
        # tensor (logp, text, lengths) must live on the same device.
        tensors = tensors.to(device)
        text, length = converter.encode(labels, batch_max_length)
        text, length = text.to(device), length.to(device)
        B = tensors.size(0)
        dummy = torch.LongTensor(B, batch_max_length + 1).fill_(0).to(device)
        preds = model(tensors, dummy)                       # [B, T, C] logits
        preds_size = torch.IntTensor([preds.size(1)] * B).to(device)
        logp = preds.log_softmax(2).permute(1, 0, 2)         # [T, B, C]
        return criterion(logp, text, preds_size, length)

    def _save():
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        # save unwrapped (no 'module.' prefix) so it loads on CPU or GPU
        to_save = model.module if hasattr(model, "module") else model
        torch.save(to_save.state_dict(), ckpt_path)

    n_epochs = 1 if smoke else epochs
    history = []
    best_val, best_epoch, patience_left = float("inf"), -1, patience
    print(f"train {len(train_items)} / val {len(val_items)} crop samples", flush=True)

    for ep in range(n_epochs):
        model.train()
        running, nb = 0.0, 0
        for tensors, labels in train_loader:
            cost = batch_cost(tensors, labels)
            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(cost.item())
            nb += 1
            if nb % log_every == 0:
                print(f"  epoch {ep+1}/{n_epochs}  step {nb}  "
                      f"loss {running/nb:.4f}", flush=True)
            if smoke and nb >= 5:
                break

        # validation loss for model selection
        model.eval()
        vrun, vnb = 0.0, 0
        with torch.no_grad():
            for tensors, labels in val_loader:
                vrun += float(batch_cost(tensors, labels).item())
                vnb += 1
                if smoke and vnb >= 2:
                    break
        train_loss = running / max(1, nb)
        val_loss = vrun / max(1, vnb)
        history.append({"epoch": ep + 1, "train": train_loss, "val": val_loss})

        improved = val_loss < best_val - 1e-4
        print(f"epoch {ep+1}/{n_epochs}  train {train_loss:.4f}  "
              f"val {val_loss:.4f}{'  * best, saved' if improved else ''}",
              flush=True)
        if improved:
            best_val, best_epoch, patience_left = val_loss, ep + 1, patience
            _save()
        else:
            patience_left -= 1
            if patience_left <= 0 and not smoke:
                print(f"early stop: no val improvement for {patience} epochs",
                      flush=True)
                break

    if best_epoch < 0:          # smoke / never-improved safety net
        _save()
        best_epoch, best_val = n_epochs, history[-1]["val"] if history else float("nan")
    # dump the loss history next to the checkpoint so the notebook can plot it
    # even on a session where training was skipped (checkpoint already existed).
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    hist_path = os.path.join(os.path.dirname(ckpt_path), "history.json")
    with open(hist_path, "w", encoding="utf-8") as fh:
        json.dump({"history": history, "best_epoch": best_epoch,
                   "best_val": best_val}, fh, indent=1)
    print(f"best epoch {best_epoch}  (val {best_val:.4f})  -> {ckpt_path}", flush=True)
    print(f"loss history -> {hist_path}", flush=True)
    return {"ckpt": ckpt_path, "kept": len(kept), "dropped_oov": dropped,
            "best_epoch": best_epoch, "best_val": best_val,
            "history": history, "smoke": smoke}
