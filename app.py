"""
app.py - Streamlit front end for the medical intake OCR pipeline.

Two stages:

  Stage 1 - Source and preprocess
    Upload a scan or pick one of the bundled forms, then see the raw photo and
    the deskewed/cleaned page side by side (pre vs post processing). Uploads run
    the full nb-02 pipeline live (src/preprocess.py): finder-pattern deskew,
    CLAHE, guide dropout, per-field crop and clean.

  Stage 2 - Extract and review
    Pick a pipeline level (L1 hybrid, L2 vocab snap, L3 db match). The left panel
    shows the processed page with every field's crop region outlined; the right
    panel reproduces the same layout as prefilled fields. Fields the pipeline is
    not sure about are highlighted:
      * L1/L2 - value fails its validator or is out of vocab (any field,
        including visit details, not just the visit block);
      * L3    - additionally, a DB-backed field whose form read disagrees with
        the matched patient record.

Run:  streamlit run app.py
Needs the two checkpoints (models/easyocr_ft/recognizer.pth,
models/digit_cnn/digit_cnn.pth) plus easyocr + torch + opencv installed.
"""

import base64
import html
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import ocr, digit_model, vocab_match, patient_match, pipeline, validate, preprocess  # noqa: E402

st.set_page_config(page_title="Medical Intake OCR", layout="wide",
                   initial_sidebar_state="expanded")
CROPS = ROOT / "data" / "scans" / "crops"
IMAGES = ROOT / "data" / "scans" / "images"
PROCESSED = ROOT / "data" / "scans" / "processed"
NONEY = {"", "none", "none known", "nil", "n/a", "na"}

CHECKBOX = pipeline.CHECKBOX
DB_BACKED = pipeline.IMMUTABLE | pipeline.PATIENT_MUTABLE
DB_PATH = ROOT / "data" / "generated" / "patient_db.json"
VISIT_FIELDS = ["date_of_visit", "department", "doctor_name", "chief_complaint"]

# Field groupings for the editable record panel (display order within each group).
EDIT_GROUPS = [
    ("Patient identity", ["last_name", "first_name", "date_of_birth", "age",
                          "gender", "ssn", "insurance_number", "blood_type"]),
    ("Contact", ["address", "email", "phone_number",
                 "emergency_contact_name", "emergency_contact_phone"]),
    ("Visit", VISIT_FIELDS),
    ("Clinical", ["allergies", "current_medications", "medical_history"]),
]

# st.dialog landed under different names across Streamlit versions.
_DIALOG = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)

# Clinical palette (shared by CSS and the SVG-ish HTML overlays).
C_PRIMARY = "#0b5394"
C_TEXT = "#0b5394"     # text-field crop accent
C_DIGIT = "#2e7d57"    # comb/digit crop accent
C_CHECK = "#9a6700"    # checkbox crop accent
C_DB = "#0b5394"       # value sourced from patient DB
C_FORM = "#5a6b7b"     # value sourced from the form
C_OK = "#1f7a4d"
C_WARN = "#9a6700"
C_BAD = "#b3261e"
C_LINE = "#cdd7e1"


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(f"""
    <style>
      :root {{
        --primary:{C_PRIMARY}; --ink:#1f2a36; --muted:#5a6b7b;
        --line:{C_LINE}; --surface:#ffffff; --bg:#eef2f6;
      }}
      .stApp {{ background:var(--bg); }}
      [data-testid="stHeader"] {{ background:transparent; }}
      .block-container {{ padding-top:1.1rem; padding-bottom:2rem; max-width:1500px; }}
      html, body, [class*="css"] {{
        font-family:"Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
      }}
      /* top app bar */
      .appbar {{
        background:linear-gradient(180deg,#0b5394 0%,#0a4a82 100%);
        color:#fff; padding:14px 22px; border-radius:6px;
        display:flex; align-items:center; justify-content:space-between;
        margin-bottom:18px; box-shadow:0 1px 3px rgba(16,42,67,.18);
      }}
      .appbar .brand {{ font-size:1.18rem; font-weight:600; letter-spacing:.2px; }}
      .appbar .brand small {{
        display:block; font-size:.74rem; font-weight:400; opacity:.85;
        letter-spacing:.3px; margin-top:2px;
      }}
      .appbar .meta {{ font-size:.78rem; text-align:right; opacity:.9; line-height:1.5; }}
      /* section header */
      .sec {{ border-bottom:1px solid var(--line); margin:6px 0 14px; padding-bottom:6px; }}
      .sec .t {{ font-size:1.02rem; font-weight:600; color:var(--ink); }}
      .sec .s {{ font-size:.82rem; color:var(--muted); margin-top:2px; }}
      .step {{ font-size:.72rem; font-weight:600; letter-spacing:.8px;
               text-transform:uppercase; color:var(--primary); }}
      /* status banner */
      .banner {{ padding:10px 14px; border-radius:5px; font-size:.9rem;
                 margin:4px 0 14px; border:1px solid var(--line);
                 border-left-width:4px; background:var(--surface); }}
      .banner b {{ font-weight:600; }}
      .pill {{ display:inline-block; padding:1px 9px; border-radius:11px;
               font-size:.72rem; font-weight:600; letter-spacing:.4px; }}
      /* legend */
      .legend {{ font-size:.78rem; color:var(--muted); margin:2px 0 10px; }}
      .legend .lg {{ display:inline-flex; align-items:center; margin-right:16px; }}
      .legend .lg i {{ width:11px; height:11px; border-radius:2px;
                       display:inline-block; margin-right:6px; border:1px solid rgba(0,0,0,.12); }}
      .panel-h {{ font-size:.82rem; font-weight:600; color:var(--muted);
                  text-transform:uppercase; letter-spacing:.5px; margin:2px 0 6px; }}
      /* buttons */
      .stButton>button, .stDownloadButton>button {{
        border-radius:5px; border:1px solid var(--line); font-weight:600;
        font-size:.86rem; padding:.4rem 1rem;
      }}
      .stButton>button[kind="primary"] {{ background:var(--primary); border-color:var(--primary); }}
      /* sidebar */
      [data-testid="stSidebar"] {{ background:#f5f8fb; border-right:1px solid var(--line); }}
      [data-testid="stSidebar"] .sb-brand {{ font-weight:600; color:var(--primary);
        font-size:1rem; padding:2px 0 10px; border-bottom:1px solid var(--line); margin-bottom:12px; }}
      [data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:5px; }}
    </style>
    """, unsafe_allow_html=True)


def section(title, sub="", step=""):
    step_html = f'<div class="step">{html.escape(step)}</div>' if step else ""
    sub_html = f'<div class="s">{html.escape(sub)}</div>' if sub else ""
    st.markdown(f'<div class="sec">{step_html}<div class="t">{html.escape(title)}</div>'
                f'{sub_html}</div>', unsafe_allow_html=True)


def banner(kind, title, detail=""):
    fg = {"ok": C_OK, "warn": C_WARN, "bad": C_BAD, "info": C_PRIMARY}[kind]
    bg = {"ok": "#eef7f1", "warn": "#fbf5e8", "bad": "#fbeeed", "info": "#eef3f9"}[kind]
    pill = {"ok": "MATCH", "warn": "REVIEW", "bad": "ERROR", "info": "INFO"}[kind]
    detail_html = f' &nbsp;{html.escape(detail)}' if detail else ""
    st.markdown(
        f'<div class="banner" style="border-left-color:{fg};background:{bg}">'
        f'<span class="pill" style="background:{fg};color:#fff">{pill}</span> '
        f'<b>{html.escape(title)}</b>{detail_html}</div>', unsafe_allow_html=True)


def legend(items):
    spans = "".join(f'<span class="lg"><i style="background:{c}"></i>{html.escape(l)}</span>'
                    for c, l in items)
    st.markdown(f'<div class="legend">{spans}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models")
def load_all():
    import torch
    gpu = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    field_map = ocr.load_field_map(str(ROOT / "data/template/field_map.json"))
    report = ocr.load_checkbox_report(str(ROOT / "data/scans/preprocess_report.json"))
    db = patient_match.load_db(str(ROOT / "data/generated/patient_db.json"))
    vocab = vocab_match.load_vocab(str(ROOT / "data/vocab"), db=db)
    text_reader = ocr.get_finetuned_reader(
        str(ROOT / "models/easyocr_ft/recognizer.pth"), gpu=gpu)
    dmodel = digit_model.load_model(
        str(ROOT / "models/digit_cnn/digit_cnn.pth"), device)
    return field_map, report, db, vocab, text_reader, dmodel, device


field_map, report, db, vocab, text_reader, dmodel, device = load_all()
CANVAS_W = field_map["canvas"]["w"]
CANVAS_H = field_map["canvas"]["h"]


@st.cache_data(show_spinner="Reading form")
def process(workdir, checkboxes):
    """Run the L1 to L3 stack on a crops dir. Cached per (workdir, checkboxes)."""
    fm, _rep, _db, vc, reader, dm, dev = load_all()
    return pipeline.run_levels(workdir, fm, checkboxes, reader, dm, vc, _db, device=dev)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def show_none(v):
    return "None" if str(v).strip().lower() in NONEY else v


def field_bbox(spec):
    """Bounding box (x, y, w, h) of a field in template-canvas pixels: the whole
    comb run / the whole checkbox group, not a single cell."""
    t = spec["type"]
    if t == "text_box":
        return tuple(spec["box"])
    if t == "comb":
        cells = spec["cells"]
        x = min(c[0] for c in cells)
        y = cells[0][1]
        right = max(c[0] + c[2] for c in cells)
        return x, y, right - x, cells[0][3]
    boxes = (list(spec["options"].values()) if "options" in spec
             else list(spec.get("group", {}).values()) + list(spec.get("rh", {}).values()))
    x = min(b[0] for b in boxes)
    y = min(b[1] for b in boxes)
    right = max(b[0] + b[2] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    return x, y, right - x, bottom - y


def _norm(s):
    return " ".join(str(s if s is not None else "").strip().split()).lower()


def level_flags(level, res, record=None):
    """Fields to highlight at the chosen level -> {field: reason}.

    `record` defaults to the pipeline output for `level`, but the app passes the
    operator's edited values so highlights clear/appear as fields are corrected.

    Every level: a value that fails its validator (bad format, out of vocab).
    L3 only: a DB-backed field whose current value differs from the matched
    patient record (a change to persist, or a bad match)."""
    record = res[level] if record is None else record
    flags = {}
    for f, v in record.items():
        if f in CHECKBOX:
            continue
        ok, reason = validate.validate_field(f, v, vocab)
        if not ok:
            flags[f] = reason
    info = res["match"]
    if level == "L3" and info.get("matched"):
        patient = info.get("patient") or {}
        for f in DB_BACKED:
            if f in CHECKBOX:
                continue
            v = record.get(f, "")
            if not v or _norm(v) == _norm(patient.get(f, "")):
                continue
            ok, _ = validate.validate_field(f, v, vocab)
            if ok:
                flags.setdefault(
                    f, f"differs from record '{patient.get(f, '')}'")
    return flags


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


TYPE_COLOR = {"text_box": C_TEXT, "comb": C_DIGIT, "checkbox": C_CHECK}


def overlay_crops_html(proc_path, scale):
    """Processed page with each field's crop region outlined (left panel)."""
    dw, dh = int(CANVAS_W * scale), int(CANVAS_H * scale)
    boxes = []
    for fname, spec in field_map["fields"].items():
        x, y, w, h = field_bbox(spec)
        col = TYPE_COLOR.get(spec["type"], "#666")
        boxes.append(
            f'<div title="{fname}" style="position:absolute;'
            f'left:{x*scale:.1f}px;top:{y*scale:.1f}px;'
            f'width:{w*scale:.1f}px;height:{h*scale:.1f}px;'
            f'border:1.5px solid {col};box-sizing:border-box;'
            f'background:{col}12;"></div>')
    return (
        f'<div style="position:relative;width:{dw}px;height:{dh}px;'
        f'background-image:url(data:image/png;base64,{_b64(proc_path)});'
        f'background-size:{dw}px {dh}px;border:1px solid {C_LINE};border-radius:4px;">'
        + "".join(boxes) + "</div>")


def overlay_fields_html(proc_path, record, flags, prov, scale):
    """Same layout reproduced as prefilled fields (right panel); flagged ones red."""
    dw, dh = int(CANVAS_W * scale), int(CANVAS_H * scale)
    cells = []
    for fname, spec in field_map["fields"].items():
        x, y, w, h = field_bbox(spec)
        val = html.escape(str(show_none(record.get(fname, ""))))
        flagged = fname in flags
        source = prov.get(fname, "form")
        border, bg = ("#d68a86", "#fbeeed") if flagged else (C_LINE, "#ffffff")
        accent = C_DB if source == "db" else (C_FORM if source == "form" else C_CHECK)
        tip = html.escape(flags.get(fname, f"{fname} - source: {source}"))
        fs = max(7, min(13, h * scale * 0.5))
        cells.append(
            f'<div title="{tip}" style="position:absolute;'
            f'left:{x*scale:.1f}px;top:{y*scale:.1f}px;'
            f'width:{w*scale:.1f}px;height:{h*scale:.1f}px;'
            f'border:1px solid {border};border-left:3px solid {accent};'
            f'background:{bg};box-sizing:border-box;overflow:hidden;border-radius:2px;'
            f'display:flex;align-items:center;padding:0 4px;'
            f'font-size:{fs:.0f}px;line-height:1.1;color:#1f2a36;'
            f'font-family:"Segoe UI",system-ui,sans-serif;white-space:nowrap;">'
            f'{val}</div>')
    return (
        f'<div style="position:relative;width:{dw}px;height:{dh}px;'
        f'border:1px solid {C_LINE};border-radius:4px;'
        f'background-image:url(data:image/png;base64,{_b64(proc_path)});'
        f'background-size:{dw}px {dh}px;">'
        f'<div style="position:absolute;inset:0;background:rgba(255,255,255,0.82);">'
        + "".join(cells) + "</div></div>")


# --------------------------------------------------------------------------- #
# Database write dialogs (L3)
# --------------------------------------------------------------------------- #
def open_modal(title, body, *args):
    """Render `body(*args)` in a modal dialog, or an inline bordered block on
    Streamlit builds without st.dialog."""
    if _DIALOG:
        @_DIALOG(title)
        def _m():
            body(*args)
        _m()
    else:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            body(*args)


def update_record_body(patient, edited):
    """Overwrite changed patient-record fields (identity + slow-changing) on disk."""
    pid = patient.get("patient_id")
    changes = {f: (patient.get(f, ""), edited.get(f, ""))
               for f in patient_match.PATIENT_FIELDS
               if f in edited and str(edited.get(f, "")) != str(patient.get(f, ""))}
    if not changes:
        st.info(f"No changes to the record for patient {pid}.")
        return
    st.caption(f"These stored fields for patient {pid} will be overwritten:")
    st.dataframe(
        pd.DataFrame([{"Field": f.replace("_", " ").title(), "Current": o, "New": n}
                     for f, (o, n) in changes.items()]),
        hide_index=True, use_container_width=True)
    if st.button("Apply changes", type="primary"):
        patient_match.update_patient(patient, edited, keys=list(changes))
        patient_match.save_db(db, str(DB_PATH))
        process.clear()
        st.success("Patient record updated.")
        st.rerun()


def register_visit_body(patient, edited):
    """Append the current (held-out) visit to the matched patient's visits[]."""
    pid = patient.get("patient_id")
    visit = {k: edited.get(k, "") for k in VISIT_FIELDS}
    st.caption(f"Register this visit for patient {pid} "
               f"({len(patient.get('visits', []))} already on record):")
    st.dataframe(
        pd.DataFrame([{"Field": k.replace("_", " ").title(), "Value": v}
                     for k, v in visit.items()]),
        hide_index=True, use_container_width=True)
    if st.button("Register visit", type="primary"):
        patient_match.register_visit(patient, edited)
        patient_match.save_db(db, str(DB_PATH))
        process.clear()
        st.success("Visit registered.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #
inject_css()
ss = st.session_state
forms = sorted(p.name for p in CROPS.iterdir() if p.is_dir())

st.markdown(
    '<div class="appbar"><div class="brand">Medical Intake OCR'
    '<small>Handwritten intake digitisation and patient record match</small></div>'
    f'<div class="meta">Recognition: fine-tuned EasyOCR + digit CNN<br>'
    f'Records on file: {len(db)}</div></div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sb-brand">Intake source</div>', unsafe_allow_html=True)
src = st.sidebar.radio("Source", ["Bundled form", "Upload scan"],
                       label_visibility="collapsed")

if src == "Bundled form":
    pid = st.sidebar.selectbox("Scanned form", forms)
    raw = IMAGES / f"{pid}.jpg"
    if not raw.exists():
        cands = list(IMAGES.glob(f"{pid}.*"))
        raw = cands[0] if cands else None
    proc = PROCESSED / f"{pid}.png"
    loaded = {"pid": pid, "raw": str(raw) if raw else None,
              "proc": str(proc), "workdir": str(CROPS / pid),
              "checkboxes": report.get(pid, {})}
    if ss.get("loaded", {}).get("pid") != pid:
        ss["loaded"] = loaded
        ss["stage"] = "preprocess"
else:
    up = st.sidebar.file_uploader("Form image", type=["jpg", "jpeg", "png"])
    if up is not None and ss.get("up_name") != up.name:
        img = preprocess.decode_upload(up.getvalue())
        if img is None:
            st.sidebar.error("Could not decode that image.")
        else:
            tmp = Path(tempfile.mkdtemp(prefix="formocr_"))
            (tmp / f"raw_{up.name}").write_bytes(up.getvalue())
            with st.spinner("Preprocessing scan"):
                summary = preprocess.process_form(img, field_map, str(tmp))
            if not summary["ok"]:
                st.sidebar.error(f"Preprocess failed: {summary['reason']}")
            else:
                ss["up_name"] = up.name
                ss["loaded"] = {"pid": up.name, "raw": str(tmp / f"raw_{up.name}"),
                                "proc": summary["processed"], "workdir": str(tmp),
                                "checkboxes": summary["checkboxes"]}
                ss["stage"] = "preprocess"

if "loaded" not in ss:
    section("No form loaded", "Select a bundled form or upload a scan to begin.",
            step="Getting started")
    st.stop()

L = ss["loaded"]
stage = ss.get("stage", "preprocess")
st.sidebar.markdown("---")
st.sidebar.caption(f"Loaded: {L['pid']}")

# --------------------------------------------------------------------------- #
# Stage 1 - preprocess
# --------------------------------------------------------------------------- #
if stage == "preprocess":
    section("Preprocessing", "Finder-pattern deskew, contrast normalisation, "
            "guide dropout, per-field crop.", step="Stage 1 of 2")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel-h">Raw scan</div>', unsafe_allow_html=True)
        if L["raw"]:
            st.image(L["raw"], use_container_width=True)
        else:
            st.info("No raw image on file for this form.")
    with c2:
        st.markdown('<div class="panel-h">Processed page</div>', unsafe_allow_html=True)
        st.image(L["proc"], use_container_width=True)
    st.markdown("---")
    if st.button("Continue to extraction", type="primary"):
        ss["stage"] = "extract"
        st.rerun()
    st.stop()

# --------------------------------------------------------------------------- #
# Stage 2 - extract and review
# --------------------------------------------------------------------------- #
section("Extraction and review",
        "Crop regions on the left, extracted record on the right. "
        "Highlighted fields need an operator's attention.", step="Stage 2 of 2")

ctrl = st.columns([1.3, 3])
with ctrl[0]:
    if st.button("Back to preprocessing"):
        ss["stage"] = "preprocess"
        st.rerun()
with ctrl[1]:
    level_label = st.radio(
        "Pipeline level",
        ["L3 - DB match (final)", "L2 - vocab snap", "L1 - raw hybrid"],
        horizontal=True, label_visibility="collapsed")
LV = {"L3 - DB match (final)": "L3", "L2 - vocab snap": "L2",
      "L1 - raw hybrid": "L1"}[level_label]

res = process(L["workdir"], L["checkboxes"])
record = res[LV]
info = res["match"]
prov = res["provenance"] if LV == "L3" else {f: "form" for f in record}


def ekey(f):
    """Session-state key for a field's editable value (per form + level)."""
    return f"edit::{L['pid']}::{LV}::{f}"


# Seed each editable value once, then read the operator's current edits back so
# the overlay and the highlights reflect changes live.
for f in record:
    ss.setdefault(ekey(f), "" if record.get(f) is None else str(record.get(f)))
edited = {f: ss[ekey(f)] for f in record}
flags = level_flags(LV, res, edited)

if info.get("matched"):
    banner("ok", f"Patient {info['patient_id']}",
           f"match score {info['score']}, margin {info['margin']}")
else:
    banner("warn", "No confident DB match",
           "Treat as a new patient or complete manual entry.")

legend([(C_TEXT, "Text crop"), (C_DIGIT, "Digit crop"), (C_CHECK, "Checkbox"),
        (C_DB, "From record"), (C_FORM, "From form"), ("#d68a86", "Needs review")])

scale = 560 / CANVAS_W
view_h = int(CANVAS_H * scale) + 12
gl, gr = st.columns(2)
with gl:
    st.markdown('<div class="panel-h">Crop regions</div>', unsafe_allow_html=True)
    components.html(overlay_crops_html(L["proc"], scale), height=view_h)
with gr:
    st.markdown(f'<div class="panel-h">Extracted record ({LV})</div>',
                unsafe_allow_html=True)
    components.html(overlay_fields_html(L["proc"], edited, flags, prov, scale),
                    height=view_h)

st.markdown("---")
flagged_n = sum(1 for f in flags if f != "__record__")
section(f"Record ({flagged_n} flagged)",
        "Every field is editable. Highlighted fields failed a check or differ "
        "from the record; edits update the view above instantly.")

seen = set()
for gtitle, gfields in EDIT_GROUPS:
    present = [f for f in gfields if f in record]
    if not present:
        continue
    st.markdown(f'<div class="panel-h">{gtitle}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, f in enumerate(present):
        label = f.replace("_", " ").title() + (" - review" if f in flags else "")
        with cols[i % 3]:
            st.text_input(label, key=ekey(f), help=flags.get(f))
        seen.add(f)
rest = [f for f in record if f not in seen]
if rest:
    st.markdown('<div class="panel-h">Other</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, f in enumerate(rest):
        label = f.replace("_", " ").title() + (" - review" if f in flags else "")
        with cols[i % 3]:
            st.text_input(label, key=ekey(f), help=flags.get(f))

if LV == "L3":
    st.markdown("---")
    st.markdown('<div class="panel-h">Database actions</div>', unsafe_allow_html=True)
    if not info.get("matched"):
        st.info("Database actions unlock once a patient is confidently matched.")
    else:
        patient = info["patient"]
        a1, a2 = st.columns(2)
        with a1:
            if st.button("Update patient record", use_container_width=True):
                open_modal("Update patient record", update_record_body, patient, edited)
        with a2:
            if st.button("Register visit", type="primary", use_container_width=True):
                open_modal("Register visit", register_visit_body, patient, edited)

with st.expander("Level progression (L1 to L2 to L3)"):
    prog = [{"field": f,
             "L1": show_none(res["L1"].get(f, "")),
             "L2": show_none(res["L2"].get(f, "")),
             "L3": show_none(res["L3"].get(f, ""))}
            for f in field_map["fields"]]
    st.dataframe(pd.DataFrame(prog), hide_index=True, use_container_width=True)
