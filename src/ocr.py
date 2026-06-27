"""
ocr.py — L0 OCR extraction for the v2 medical-intake pipeline.

L0 = pretrained EasyOCR + per-field allow-lists + empty-cell skip.
No fuzzy/vocab correction (that is L1, src/vocab_match.py) and no db matching
(L2, src/patient_match.py). This module is import-only logic so both notebook
03 and the Streamlit app can call the same code path.

What it does NOT do:
  * It does NOT crop or clean. Cropping/cleaning is locked in notebook 02; the
    per-field and per-comb-cell crops already live on disk as cleaned grayscale
    PNGs under data/scans/crops/<pid>/.
  * It does NOT OCR checkboxes. gender + blood_type are read by pixel detection
    in nb 02 and stored in data/scans/preprocess_report.json — we just look them up.

Crop file naming (written by nb 02):
  text_box field   ->  <field>.png
  comb field       ->  <field>_<cellindex>.png   (one box per character)

Public API:
  get_reader(gpu=False)                  -> cached easyocr.Reader (loaded once)
  extract_form(crops_dir, field_map,
               checkboxes=None, reader=None, gpu=False) -> {field_name: value}
  load_field_map(path)                   -> dict
  load_checkbox_report(path)             -> {pid: {"gender":[...], "blood_type":[...]}}
  predict_all(crops_root, field_map, report, gpu=False) -> [{patient_id, **fields}, ...]
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Allow-lists (EasyOCR `allowlist=`) — derived from the generation vocab.
# Each field can only emit legal characters, which is most of L0's leverage.
# --------------------------------------------------------------------------- #
_UP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LO = "abcdefghijklmnopqrstuvwxyz"
_LETTERS = _UP + _LO
_DIGITS = "0123456789"

ALLOWLISTS = {
    # --- text_box fields ---------------------------------------------------- #
    "last_name":               _LETTERS + "-' ",
    "first_name":              _LETTERS + "-' ",
    "emergency_contact_name":  _LETTERS + "-' ",
    "address":                 _LETTERS + _DIGITS + " ,",
    "email":                   _LO + _DIGITS + "@.",          # lowercase only
    "department":              _LETTERS + " &",
    "doctor_name":             _LETTERS + " .",
    "chief_complaint":         _LETTERS + " ",
    "medical_history":         _LETTERS + " ,",
    "allergies":               _LETTERS + " ,",
    "current_medications":     _LETTERS + _DIGITS + " ,",
    # --- comb fields (allow-list applied per single-char cell) -------------- #
    "date_of_birth":           _DIGITS,
    "date_of_visit":           _DIGITS,
    "age":                     _DIGITS,
    "ssn":                     _DIGITS,
    "phone_number":            _DIGITS,
    "emergency_contact_phone": _DIGITS,
    "insurance_number":        _UP + _DIGITS,                 # 3 letters + 8 digits
}

# Separator inserted between comb groups when assembling the final string.
# (insurance_number + age are stored contiguous in labels.csv -> no separator.)
COMB_SEP = {
    "date_of_birth":           "/",
    "date_of_visit":           "/",
    "ssn":                     "-",
    "phone_number":            "-",
    "emergency_contact_phone": "-",
    "insurance_number":        "",
    "age":                     "",
}

# Ink-density threshold for the empty-cell skip. Cleaned crops are white-bg
# grayscale; a truly empty cell (incl. blank leading age cells) has almost no
# dark ink. Anything below this fraction of dark pixels is treated as empty and
# never OCR'd — the backstop for residual edge specks.
INK_DARK = 110          # pixel < INK_DARK counts as ink
EMPTY_MAX_INK_FRAC = 0.02

# --------------------------------------------------------------------------- #
# Reader singleton (the model is heavy — load it exactly once).
# --------------------------------------------------------------------------- #
_READER = None


def get_reader(gpu: bool = False):
    """Return a cached pretrained EasyOCR reader (L0), constructing it on first use."""
    global _READER
    if _READER is None:
        import easyocr  # imported lazily so importing this module stays cheap
        _READER = easyocr.Reader(["en"], gpu=gpu)
    return _READER


# Cache of fine-tuned readers, keyed by checkpoint path (L1).
_FT_READERS: dict = {}


def get_finetuned_reader(recog_ckpt: str, gpu: bool = False):
    """
    Return an EasyOCR reader whose recognizer weights are loaded from a fine-tuned
    checkpoint (produced by finetune.train_recognizer) — this is the L1 reader.

    Built with quantize=False because a float fine-tuned state-dict cannot load
    into the default CPU-quantized recognizer. Cached per checkpoint path.
    """
    import os as _os
    key = _os.path.abspath(recog_ckpt)
    if key not in _FT_READERS:
        import easyocr
        import torch
        from collections import OrderedDict
        rdr = easyocr.Reader(["en"], gpu=gpu, quantize=False)
        sd = torch.load(recog_ckpt, map_location="cpu")
        # normalize checkpoint keys to be prefix-free (a GPU/DataParallel-trained
        # checkpoint carries a 'module.' prefix; a CPU one does not)
        sd = OrderedDict((k[7:] if k.startswith("module.") else k, v)
                         for k, v in sd.items())
        # load into the underlying module whether or not reader.recognizer is
        # wrapped in DataParallel (GPU builds wrap it; CPU builds don't)
        target = rdr.recognizer
        target = target.module if hasattr(target, "module") else target
        target.load_state_dict(sd)
        _FT_READERS[key] = rdr
    return _FT_READERS[key]


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _load_gray(path: str) -> np.ndarray | None:
    """Load a crop as a 2-D uint8 grayscale array, or None if missing."""
    if not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert("L"))


def _is_empty_cell(gray: np.ndarray) -> bool:
    """True if a comb cell has too little ink to be a character."""
    if gray is None or gray.size == 0:
        return True
    ink_frac = float(np.mean(gray < INK_DARK))
    return ink_frac < EMPTY_MAX_INK_FRAC


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    """EasyOCR wants an HxWx3 array."""
    return np.stack([gray, gray, gray], axis=-1)


def _ocr_text(reader, gray: np.ndarray, allowlist: str) -> str:
    """OCR a multi-character text crop -> single joined string (reading order)."""
    if gray is None:
        return ""
    results = reader.readtext(
        _to_rgb(gray), allowlist=allowlist, detail=1, paragraph=False
    )
    # results: [(bbox, text, conf), ...]; order left-to-right by box x.
    results = sorted(results, key=lambda r: r[0][0][0])
    parts = [r[1].strip() for r in results if r[1].strip()]
    return " ".join(parts).strip()


def _ocr_char(reader, gray: np.ndarray, allowlist: str) -> str:
    """OCR a single comb cell -> at most one allowed character."""
    if gray is None:
        return ""
    results = reader.readtext(
        _to_rgb(gray), allowlist=allowlist, detail=0, paragraph=False
    )
    legal = set(allowlist)
    for token in results:
        for ch in token:
            if ch in legal:
                return ch
    return ""


# --------------------------------------------------------------------------- #
# Field-type readers
# --------------------------------------------------------------------------- #
def _normalize_text(field: str, text: str) -> str:
    """Light, deterministic cleanup of an OCR'd text field (still L0)."""
    text = re.sub(r"\s+", " ", text).strip()
    if field == "email":
        text = text.replace(" ", "").lower()
    return text


def read_text_field(reader, crops_dir: str, field: str) -> str:
    gray = _load_gray(os.path.join(crops_dir, f"{field}.png"))
    raw = _ocr_text(reader, gray, ALLOWLISTS[field])
    return _normalize_text(field, raw)


def read_comb_field(reader, crops_dir: str, field: str, cfg: dict) -> str:
    """Read each cell crop, skipping empties, then assemble the formatted value."""
    allowlist = ALLOWLISTS[field]
    n_cells = len(cfg["cells"])
    chars: list[str] = []
    for i in range(n_cells):
        gray = _load_gray(os.path.join(crops_dir, f"{field}_{i}.png"))
        if _is_empty_cell(gray):           # empty-cell skip FIRST
            chars.append("")
        else:
            chars.append(_ocr_char(reader, gray, allowlist))
    return _assemble_comb(field, cfg, chars)


def _assemble_comb(field: str, cfg: dict, chars: list[str]) -> str:
    """Join per-cell chars into the field's canonical string format."""
    if field == "age":
        # right-aligned NNN; concatenate, strip leading zeros (_29/029/29 -> 29)
        joined = "".join(chars).strip()
        joined = joined.lstrip("0")
        return joined if joined else ("0" if "".join(chars) else "")

    sep = COMB_SEP.get(field, "")
    if not sep:
        return "".join(chars)

    # Insert the separator at fixed group boundaries (robust to empty cells:
    # positions are preserved, so a dropped cell doesn't shift the slashes).
    groups = cfg.get("groups", [len(chars)])
    out, idx = [], 0
    for gi, g in enumerate(groups):
        if gi > 0:
            out.append(sep)
        out.append("".join(chars[idx:idx + g]))
        idx += g
    if idx < len(chars):                   # any trailing cells (shouldn't happen)
        out.append("".join(chars[idx:]))
    return "".join(out)


def read_checkbox(field: str, checkboxes: dict | None) -> str:
    """Map nb-02 pixel-detection output to the labels.csv string form."""
    if not checkboxes:
        return ""
    if field == "gender":
        sel = checkboxes.get("gender", [])
        m = {"M": "Male", "F": "Female", "Other": "Other"}
        return m.get(sel[0], sel[0]) if sel else ""
    if field == "blood_type":
        sel = checkboxes.get("blood_type", [])
        group = rh = ""
        for tok in sel:
            if tok.startswith("group:"):
                group = tok.split(":", 1)[1]
            elif tok.startswith("rh:"):
                rh = tok.split(":", 1)[1]
        return f"{group}{rh}"
    return ""


# --------------------------------------------------------------------------- #
# Top-level: one form
# --------------------------------------------------------------------------- #
def extract_form(crops_dir: str, field_map: dict,
                 checkboxes: dict | None = None,
                 reader=None, gpu: bool = False) -> dict:
    """
    L0 extraction for a single form.

    crops_dir   : data/scans/crops/<pid>/
    field_map   : parsed field_map.json (defines field order + types + cells)
    checkboxes  : that form's entry from preprocess_report.json
                  e.g. {"gender": ["M"], "blood_type": ["group:B", "rh:+"]}
    returns     : {field_name: value} in field_map order.
    """
    reader = reader or get_reader(gpu)
    out: dict[str, str] = {}
    for fname, cfg in field_map["fields"].items():
        ftype = cfg["type"]
        if ftype == "text_box":
            out[fname] = read_text_field(reader, crops_dir, fname)
        elif ftype == "comb":
            out[fname] = read_comb_field(reader, crops_dir, fname, cfg)
        elif ftype == "checkbox":
            out[fname] = read_checkbox(fname, checkboxes)
        else:
            out[fname] = ""
    return out


# --------------------------------------------------------------------------- #
# Loaders + batch driver
# --------------------------------------------------------------------------- #
def load_field_map(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_checkbox_report(path: str) -> dict:
    """preprocess_report.json is a list of per-form dicts -> index by patient_id."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {e["patient_id"]: e.get("checkboxes", {}) for e in data}


def predict_all(crops_root: str, field_map: dict, report: dict,
                gpu: bool = False, pids: list[str] | None = None,
                progress: bool = True, reader=None) -> list[dict]:
    """
    Run extraction over form folders under crops_root.

    reader=None  -> L0 (pretrained reader via get_reader).
    reader=<ft>  -> L1 (pass a get_finetuned_reader(...) instance); same pipeline,
                    fine-tuned recognizer.

    Returns a list of {"patient_id": pid, **fields} dicts, sorted by patient_id.
    """
    reader = reader or get_reader(gpu)
    if pids is None:
        pids = sorted(d for d in os.listdir(crops_root)
                      if os.path.isdir(os.path.join(crops_root, d)))
    rows = []
    for n, pid in enumerate(pids, 1):
        crops_dir = os.path.join(crops_root, pid)
        fields = extract_form(crops_dir, field_map,
                              checkboxes=report.get(pid), reader=reader)
        rows.append({"patient_id": pid, **fields})
        if progress:
            print(f"  [{n:>3}/{len(pids)}] {pid}", flush=True)
    return rows
