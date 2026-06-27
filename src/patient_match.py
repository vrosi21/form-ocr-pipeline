"""
patient_match.py — L3: match a form's identity fields against patient_db.json,
then pull the canonical identity from the matched patient and register the visit.

Why this is the big lever: OCR only has to get the identity *close enough to
match*, not exactly right. We score every patient in the db by a weighted
character-similarity over the identity fields — leaning hardest on the
near-unique keys (SSN + insurance number), with name + DOB + phone as support —
take the best, and (if confident) replace the noisy identity reads with the db's
canonical values. The current visit (date_of_visit/department/doctor/complaint)
is NOT in the db and stays from the form (L1/L2), as do the other mutable fields.

Pure stdlib (difflib) — runs anywhere.
"""

from __future__ import annotations

import difflib
import json

# Identity fields taken FROM THE DB on a confident match (canonical, on file).
IDENTITY = ["last_name", "first_name", "date_of_birth", "age", "gender", "ssn",
            "insurance_number", "phone_number", "blood_type", "address", "email"]

# Slow-changing patient attributes that also live in the DB. Pulled from the DB
# by default (they're far more accurate there), but the app should flag them for
# review when the form disagrees with a confident read (a possible update).
DB_STATIC = ["allergies", "current_medications", "medical_history",
             "emergency_contact_name", "emergency_contact_phone"]

# Per-field weights for the match score (SSN + insurance dominate: near-unique).
WEIGHTS = {
    "ssn": 3.0, "insurance_number": 3.0,
    "phone_number": 2.0, "date_of_birth": 2.0,
    "last_name": 1.5, "first_name": 1.5, "age": 0.5,
}


def load_db(path: str) -> list:
    return json.load(open(path, encoding="utf-8"))


def _sim(a, b) -> float:
    a = ("" if a is None else str(a)).strip().lower()
    b = ("" if b is None else str(b)).strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def score(form: dict, patient: dict) -> float:
    """Weighted mean similarity over the match keys (0..1)."""
    s = w = 0.0
    for f, wt in WEIGHTS.items():
        s += wt * _sim(form.get(f, ""), patient.get(f, ""))
        w += wt
    return s / w if w else 0.0


def match(form: dict, db: list):
    """Return (best_patient, best_score, margin_over_2nd)."""
    scored = sorted(((score(form, p), p) for p in db), key=lambda t: t[0],
                    reverse=True)
    best_s, best_p = scored[0]
    second_s = scored[1][0] if len(scored) > 1 else 0.0
    return best_p, best_s, best_s - second_s


def apply_l3(form_fields: dict, db: list, threshold: float = 0.55,
             include_static: bool = True):
    """
    L3: match, and on a confident match overwrite the DB-backed fields with the
    patient's canonical values. IDENTITY is always pulled; DB_STATIC (slow-changing
    attributes) is pulled when include_static=True. The current-visit fields
    (date_of_visit/department/doctor/complaint) always stay from the form.

    Returns (final_fields, info) where info includes the matched db record
    ("patient") so the app/review layer can compare form vs record.
    """
    best, s, margin = match(form_fields, db)
    out = dict(form_fields)
    matched = s >= threshold
    pulled = []
    if matched:
        for f in IDENTITY + (DB_STATIC if include_static else []):
            if f in best:
                out[f] = best[f]
                pulled.append(f)
    return out, {"matched": matched, "patient_id": best.get("patient_id"),
                 "score": round(s, 3), "margin": round(margin, 3),
                 "db_fields": pulled, "patient": best if matched else None}


def register_visit(patient: dict, form_fields: dict) -> dict:
    """Append the form's current visit to the matched patient's visits[] (app use)."""
    visit = {k: form_fields.get(k, "") for k in
             ("date_of_visit", "department", "doctor_name", "chief_complaint")}
    patient.setdefault("visits", []).append(visit)
    return patient
