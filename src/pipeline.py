"""
pipeline.py — the hybrid base read (L1) and the full L0->L3 stack.

Hybrid base read = the production "raw" read, routing each field to the engine
that's best for it:
    comb digit cells          -> digit CNN          (digit_model, ~98%/digit)
    insurance letter cells 0-2 -> EasyOCR  A-Z       (via predict_comb_digits)
    text fields               -> EasyOCR (fine-tuned on text)
    checkboxes                -> nb-02 pixel detection (from preprocess_report)

On top of that:
    L2 = vocab snap   (vocab_match.apply_l2)
    L3 = db match     (patient_match.apply_l3)

This is the module the Streamlit app imports to run the chosen level.
"""

from __future__ import annotations

import ocr
import digit_model
import vocab_match
import patient_match


def extract_hybrid(crops_dir: str, field_map: dict, checkboxes,
                   text_reader, dmodel, device: str = "cpu") -> dict:
    """L1 hybrid read of a single form -> {field: value}."""
    out = {}
    for fname, cfg in field_map["fields"].items():
        t = cfg["type"]
        if t == "text_box":
            out[fname] = ocr.read_text_field(text_reader, crops_dir, fname)
        elif t == "comb":
            out[fname] = digit_model.predict_comb_digits(
                dmodel, crops_dir, fname, cfg, device, letter_reader=text_reader)
        elif t == "checkbox":
            out[fname] = ocr.read_checkbox(fname, checkboxes)
        else:
            out[fname] = ""
    return out


def run_levels(crops_dir, field_map, checkboxes, text_reader, dmodel,
               vocab, db, device="cpu", l3_threshold=0.55):
    """
    Run the hybrid base read then stack L2 and L3 for one form.
    Returns {"L1": {...}, "L2": {...}, "L3": {...}, "match": {...}}.
    """
    l1 = extract_hybrid(crops_dir, field_map, checkboxes, text_reader, dmodel, device)
    l2 = vocab_match.apply_l2(l1, vocab)
    l3, info = patient_match.apply_l3(l2, db, threshold=l3_threshold)
    return {"L1": l1, "L2": l2, "L3": l3, "match": info}
