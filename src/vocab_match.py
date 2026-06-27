"""
vocab_match.py — L2: snap form-authoritative text fields to the known vocab.

OCR (even fine-tuned) rarely nails a free-text field exactly, but most of these
fields are drawn from a *finite* list in data/vocab/. Fuzzy-matching a noisy read
to the nearest legal value turns "Rhematology" into "Rheumatology" — a near-miss
into an exact hit.

Scope (form-authoritative, small-vocab fields):
    department          -> departments.json      (single value)
    chief_complaint     -> chief_complaints.json  (single value)
    allergies           -> allergies.json         (comma-separated multi)
    medical_history     -> history_options.json   (comma-separated multi)
    current_medications -> medications.json        (comma-separated multi; name[+dosage])

Names are deliberately NOT snapped here — they're large-vocab identity fields,
handled by L3 (db match), which also tolerates noisy reads for matching.
Pure stdlib (difflib) so this runs anywhere, no torch / no extra deps.
"""

from __future__ import annotations

import difflib
import json
import os

FIELD_VOCAB = {
    "department":          "departments",
    "chief_complaint":     "chief_complaints",
    "allergies":           "allergies",
    "medical_history":     "history_options",
    "current_medications": "medications",
}
MULTI = {"allergies", "medical_history", "current_medications"}
# "no value" tokens kept verbatim instead of being snapped to a vocab entry.
NONE_TOKENS = {"", "none", "none known", "nil", "n/a", "na", "no", "unknown"}


def doctor_choices(db) -> list:
    """Doctor pool = the unique doctor_names seen across the db's visit history."""
    return sorted({v["doctor_name"] for p in db for v in p.get("visits", [])
                   if v.get("doctor_name")})


def load_vocab(vocab_dir: str, db=None) -> dict:
    """Load each field's candidate list. medications.json is [{name,dosages}] ->
    expand to 'Name' and 'Name dose' strings. If `db` is given, also builds a
    doctor_name vocab from the db's visit history (lets L2 snap doctor names)."""
    vocab = {}
    for field, fname in FIELD_VOCAB.items():
        data = json.load(open(os.path.join(vocab_dir, fname + ".json"),
                              encoding="utf-8"))
        if field == "current_medications":
            choices = []
            for m in data:
                choices.append(m["name"])
                for dose in m.get("dosages", []):
                    choices.append(f'{m["name"]} {dose}')
            vocab[field] = choices
        else:
            vocab[field] = list(data)
    if db is not None:
        vocab["doctor_name"] = doctor_choices(db)
    return vocab


def _snap_token(tok: str, choices: list, lower_map: dict, threshold: float) -> str:
    tok = tok.strip()
    if tok.lower() in NONE_TOKENS:
        return tok
    hit = difflib.get_close_matches(tok.lower(), list(lower_map), n=1, cutoff=threshold)
    return lower_map[hit[0]] if hit else tok


def snap_field(field: str, value: str, vocab: dict, threshold: float = 0.6) -> str:
    """Snap one field's value to its vocab (case-insensitive); unmatched stays raw."""
    value = (value or "").strip()
    if field not in vocab or not value or value.lower() in NONE_TOKENS:
        return value
    choices = vocab[field]
    lower_map = {c.lower(): c for c in choices}   # match lower, return canonical
    if field in MULTI:
        toks = [t for t in (p.strip() for p in value.split(",")) if t]
        return ", ".join(_snap_token(t, choices, lower_map, threshold) for t in toks)
    return _snap_token(value, choices, lower_map, threshold)


def apply_l2(fields: dict, vocab: dict, threshold: float = 0.6) -> dict:
    """Return a copy of `fields` with the vocab-backed fields snapped (L2).
    Also snaps doctor_name when a doctor vocab was loaded (db-derived)."""
    out = dict(fields)
    snap_fields = list(FIELD_VOCAB) + (["doctor_name"] if "doctor_name" in vocab else [])
    for f in snap_fields:
        if f in out:
            out[f] = snap_field(f, out.get(f, ""), vocab, threshold)
    return out


def in_vocab(field: str, value: str, vocab: dict) -> bool:
    """True if value (single field) is a legal vocab entry — used for review flags."""
    if field not in vocab:
        return True
    return (value or "").strip() in set(vocab[field])
