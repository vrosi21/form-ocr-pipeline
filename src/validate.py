"""
validate.py — per-field plausibility checks for the review queue.

Every field has its own notion of "a real value": a date must be a real calendar
date (not 45/13/2099), a phone must be NNN-NNN-NNNN, an SSN XXX-XX-XXXX, insurance
3 letters + 8 digits, an age in range, a department/complaint/doctor a known
vocab entry, a name alphabetic, etc. validate_field returns (ok, reason); the app
flags any field that fails so an admin reviews it instead of auto-accepting.

Pure stdlib.
"""

from __future__ import annotations

import re
from datetime import datetime

GENDERS = {"Male", "Female", "Other"}
BLOOD = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}
NONE_TOKENS = {"none", "none known", "nil", "n/a", "na", ""}


def is_real_date(s: str, fmt: str = "%d/%m/%Y") -> bool:
    try:
        datetime.strptime((s or "").strip(), fmt)
        return True
    except ValueError:
        return False


def validate_field(field: str, value, vocab: dict | None = None) -> tuple:
    """Return (ok: bool, reason: str). Empty values are treated as OK here
    (presence/empty handling belongs to the caller)."""
    vocab = vocab or {}
    v = ("" if value is None else str(value)).strip()
    if v == "":
        return True, ""

    if field in ("date_of_birth", "date_of_visit"):
        if not is_real_date(v):
            return False, "not a real date (dd/mm/yyyy)"
        year = int(v.strip()[-4:])
        now = datetime.now().year
        if field == "date_of_birth" and not (1900 <= year <= now):
            return False, f"implausible birth year {year}"
        if field == "date_of_visit" and not (2000 <= year <= now + 1):
            return False, f"implausible visit year {year}"
        return True, ""
    if field in ("phone_number", "emergency_contact_phone"):
        return (True, "") if re.fullmatch(r"\d{3}-\d{3}-\d{4}", v) else (False, "bad phone format")
    if field == "ssn":
        return (True, "") if re.fullmatch(r"\d{3}-\d{2}-\d{4}", v) else (False, "bad SSN format")
    if field == "insurance_number":
        return (True, "") if re.fullmatch(r"[A-Za-z]{3}\d{8}", v) else (False, "bad insurance format")
    if field == "email":
        return (True, "") if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v) else (False, "bad email")
    if field == "age":
        return (True, "") if (v.isdigit() and 0 <= int(v) <= 120) else (False, "implausible age")
    if field == "gender":
        return (True, "") if v in GENDERS else (False, "unknown gender")
    if field == "blood_type":
        return (True, "") if v in BLOOD else (False, "unknown blood type")
    if field in ("first_name", "last_name", "emergency_contact_name"):
        ok = any(c.isalpha() for c in v) and all(c.isalpha() or c in " -'." for c in v)
        return (True, "") if ok else (False, "non-alphabetic name")
    if field == "doctor_name":
        if vocab.get("doctor_name"):
            return (True, "") if v in set(vocab["doctor_name"]) else (False, "doctor not in known list")
        return (True, "") if v.lower().startswith("dr") else (False, "missing Dr. prefix")
    if field in ("department", "chief_complaint"):
        choices = set(vocab.get(field, []))
        if choices:
            return (True, "") if v in choices else (False, f"not a known {field.replace('_',' ')}")
        return True, ""
    if field in ("allergies", "current_medications", "medical_history"):
        choices = {c.lower() for c in vocab.get(field, [])}
        toks = [t.strip() for t in v.split(",") if t.strip()]
        bad = [t for t in toks if t.lower() not in choices and t.lower() not in NONE_TOKENS]
        return (True, "") if not bad else (False, "unknown: " + ", ".join(bad))
    # address + anything else: free-form, no strict check
    return True, ""
