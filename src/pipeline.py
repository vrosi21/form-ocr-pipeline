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
import validate

# Field behaviour tiers (drive provenance + review flags).
CHECKBOX = {"gender", "blood_type"}
IMMUTABLE = {"last_name", "first_name", "date_of_birth", "age", "ssn",
             "insurance_number", "gender", "blood_type"}
PATIENT_MUTABLE = {"address", "email", "phone_number", "allergies",
                   "current_medications", "medical_history",
                   "emergency_contact_name", "emergency_contact_phone"}
VISIT = {"date_of_visit", "department", "doctor_name", "chief_complaint"}


def _norm(s):
    return " ".join(str(s if s is not None else "").strip().split()).lower()


def review_record(l2: dict, l3: dict, info: dict, vocab: dict):
    """
    Build (provenance, flags) for the app's review queue.

    provenance[field] -> "db" | "form" | "detected"
    flags[field]      -> reason a human should look (only flagged fields appear):
      * invalid value      (fails its per-field validator: not a real date, bad
                            phone/SSN/email, out-of-vocab department/doctor, ...)
      * possible update    (a DB-backed field where the form confidently disagrees
                            with the record -> maybe the patient changed it, maybe
                            a bad match: admin decides, not auto-accepted)
      * no DB match        (whole record needs manual handling)
    """
    matched = info.get("matched")
    patient = info.get("patient") or {}
    prov, flags = {}, {}
    for f in l3:
        if f in CHECKBOX:
            prov[f] = "detected"
        elif matched and (f in IMMUTABLE or f in PATIENT_MUTABLE):
            prov[f] = "db"
        else:
            prov[f] = "form"

    if not matched:
        flags["__record__"] = "no confident DB match — manual review"

    # per-field validity on the final value
    for f in l3:
        if f in CHECKBOX:
            continue
        ok, reason = validate.validate_field(f, l3.get(f, ""), vocab)
        if not ok:
            flags[f] = reason

    # possible-update: a DB-backed field where the form has a *valid, different* read
    if matched:
        for f in PATIENT_MUTABLE:
            form_v = l2.get(f, "")
            if not form_v or _norm(form_v) == _norm(patient.get(f, "")):
                continue
            ok, _ = validate.validate_field(f, form_v, vocab)
            if ok:
                flags[f] = (f"form '{form_v}' differs from record "
                            f"'{patient.get(f, '')}' — review")
    return prov, flags


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
    provenance, flags = review_record(l2, l3, info, vocab)
    return {"L1": l1, "L2": l2, "L3": l3, "match": info,
            "provenance": provenance, "review": flags}
