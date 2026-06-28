"""
app.py — Streamlit front end for the medical-intake OCR pipeline.

Two stages:

  Stage 1 · Source & preprocess
    Upload a scan OR pick one of the bundled forms, then see the raw photo and
    the deskewed/cleaned page side by side (pre vs post processing). Uploads run
    the full nb-02 pipeline live (src/preprocess.py): finder-pattern deskew ->
    CLAHE -> guide dropout -> per-field crop + clean.

  Stage 2 · Extract & review
    Pick a pipeline level (L1 hybrid -> L2 vocab snap -> L3 db match). The left
    panel shows the processed page with every field's crop region outlined; the
    right panel reproduces the same layout as prefilled fields. Fields the
    pipeline isn't sure about are highlighted in red:
      * L1/L2 — value fails its validator / is out of vocab (any field, incl.
        visit details, not just the visit block);
      * L3    — additionally, a DB-backed field whose form read disagrees with
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

st.set_page_config(page_title="Medical Intake OCR", layout="wide")
CROPS = ROOT / "data" / "scans" / "crops"
IMAGES = ROOT / "data" / "scans" / "images"
PROCESSED = ROOT / "data" / "scans" / "processed"
NONEY = {"", "none", "none known", "nil", "n/a", "na"}

# Field tiers reused for provenance + per-level highlighting.
CHECKBOX = pipeline.CHECKBOX
DB_BACKED = pipeline.IMMUTABLE | pipeline.PATIENT_MUTABLE


@st.cache_resource(show_spinner="Loading models…")
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


@st.cache_data(show_spinner="Reading form…")
def process(workdir, checkboxes):
    """Run the L1->L3 stack on a crops dir. Cached per (workdir, checkboxes)."""
    fm, _rep, _db, vc, reader, dm, dev = load_all()
    return pipeline.run_levels(workdir, fm, checkboxes, reader, dm, vc, _db, device=dev)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def show_none(v):
    return "None" if str(v).strip().lower() in NONEY else v


def field_bbox(spec):
    """Bounding box (x, y, w, h) of a field in template-canvas pixels —
    the whole comb run / the whole checkbox group, not a single cell."""
    t = spec["type"]
    if t == "text_box":
        return tuple(spec["box"])
    if t == "comb":
        cells = spec["cells"]
        x = min(c[0] for c in cells)
        y = cells[0][1]
        right = max(c[0] + c[2] for c in cells)
        return x, y, right - x, cells[0][3]
    # checkbox: span the option boxes
    boxes = (list(spec["options"].values()) if "options" in spec
             else list(spec.get("group", {}).values()) + list(spec.get("rh", {}).values()))
    x = min(b[0] for b in boxes)
    y = min(b[1] for b in boxes)
    right = max(b[0] + b[2] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    return x, y, right - x, bottom - y


def _norm(s):
    return " ".join(str(s if s is not None else "").strip().split()).lower()


def level_flags(level, res):
    """Fields to highlight at the chosen level -> {field: reason}.

    Every level: a value that fails its validator (bad format, out of vocab).
    L3 only: a DB-backed field whose form read confidently differs from the
    matched patient record (possible update, or a bad match)."""
    record = res[level]
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
            form_v = res["L2"].get(f, "")
            if not form_v or _norm(form_v) == _norm(patient.get(f, "")):
                continue
            ok, _ = validate.validate_field(f, form_v, vocab)
            if ok:
                flags.setdefault(
                    f, f"form '{form_v}' differs from record '{patient.get(f, '')}'")
    return flags


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


TYPE_COLOR = {"text_box": "#2563eb", "comb": "#16a34a", "checkbox": "#d97706"}


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
            f'background:{col}14;"></div>')
    return (
        f'<div style="position:relative;width:{dw}px;height:{dh}px;'
        f'background-image:url(data:image/png;base64,{_b64(proc_path)});'
        f'background-size:{dw}px {dh}px;border:1px solid #ccc;">'
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
        if flagged:
            border, bg = "#dc2626", "#fee2e2"
        else:
            border, bg = "#cbd5e1", "#ffffff"
        accent = "#2563eb" if source == "db" else (
            "#64748b" if source == "form" else "#d97706")  # db / form / detected
        tip = html.escape(flags.get(fname, f"{fname} · source: {source}"))
        fs = max(7, min(13, h * scale * 0.5))
        cells.append(
            f'<div title="{tip}" style="position:absolute;'
            f'left:{x*scale:.1f}px;top:{y*scale:.1f}px;'
            f'width:{w*scale:.1f}px;height:{h*scale:.1f}px;'
            f'border:1.5px solid {border};border-left:4px solid {accent};'
            f'background:{bg};box-sizing:border-box;overflow:hidden;'
            f'display:flex;align-items:center;padding:0 3px;'
            f'font-size:{fs:.0f}px;line-height:1.1;color:#111;'
            f'font-family:system-ui,sans-serif;white-space:nowrap;">'
            f'{val}</div>')
    return (
        f'<div style="position:relative;width:{dw}px;height:{dh}px;'
        f'background:#fafafa;border:1px solid #ccc;'
        f'background-image:url(data:image/png;base64,{_b64(proc_path)});'
        f'background-size:{dw}px {dh}px;background-blend-mode:lighten;">'
        f'<div style="position:absolute;inset:0;background:rgba(255,255,255,0.78);">'
        + "".join(cells) + "</div></div>")


# --------------------------------------------------------------------------- #
# Sidebar — source selection
# --------------------------------------------------------------------------- #
forms = sorted(p.name for p in CROPS.iterdir() if p.is_dir())
ss = st.session_state

st.sidebar.title("🏥 Intake OCR")
src = st.sidebar.radio("Source", ["Pick existing form", "Upload a scan"])

if src == "Pick existing form":
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
            with st.spinner("Preprocessing scan…"):
                summary = preprocess.process_form(img, field_map, str(tmp))
            if not summary["ok"]:
                st.sidebar.error(f"Preprocess failed — {summary['reason']}")
            else:
                ss["up_name"] = up.name
                ss["loaded"] = {"pid": up.name, "raw": str(tmp / f"raw_{up.name}"),
                                "proc": summary["processed"], "workdir": str(tmp),
                                "checkboxes": summary["checkboxes"]}
                ss["stage"] = "preprocess"

if "loaded" not in ss:
    st.title("🏥 Medical intake form — OCR & patient match")
    st.info("Pick a bundled form or upload a scan from the sidebar to begin.")
    st.stop()

L = ss["loaded"]
stage = ss.get("stage", "preprocess")

# --------------------------------------------------------------------------- #
# Stage 1 — preprocess (pre vs post)
# --------------------------------------------------------------------------- #
if stage == "preprocess":
    st.title("Stage 1 · Preprocessing")
    st.caption(f"`{L['pid']}` — raw scan vs deskewed + cleaned page "
               "(finder-pattern warp → CLAHE → guide dropout).")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Raw scan")
        if L["raw"]:
            st.image(L["raw"], use_container_width=True)
        else:
            st.info("No raw image on file for this form.")
    with c2:
        st.subheader("Preprocessed")
        st.image(L["proc"], use_container_width=True)
    st.divider()
    if st.button("Continue to extraction ▸", type="primary"):
        ss["stage"] = "extract"
        st.rerun()
    st.stop()

# --------------------------------------------------------------------------- #
# Stage 2 — extract & review
# --------------------------------------------------------------------------- #
st.title("Stage 2 · Extraction & review")
top = st.columns([2, 3])
with top[0]:
    if st.button("◂ Back to preprocessing"):
        ss["stage"] = "preprocess"
        st.rerun()
with top[1]:
    level_label = st.radio(
        "Pipeline level",
        ["L3 — DB match (final)", "L2 — vocab snap", "L1 — raw hybrid"],
        horizontal=True)
LV = {"L3 — DB match (final)": "L3", "L2 — vocab snap": "L2",
      "L1 — raw hybrid": "L1"}[level_label]

res = process(L["workdir"], L["checkboxes"])
record = res[LV]
info = res["match"]
flags = level_flags(LV, res)
prov = res["provenance"] if LV == "L3" else {f: "form" for f in record}

if info.get("matched"):
    st.success(f"Matched patient **{info['patient_id']}**  ·  score "
               f"{info['score']}  ·  margin {info['margin']}")
else:
    st.warning("No confident DB match — treat as a new patient / manual entry.")

st.caption(
    "Left: crop regions on the processed page "
    "(🟦 text · 🟩 digits · 🟧 checkbox). "
    "Right: extracted values in the same layout — "
    "left accent 🟦 from DB · ⬜ from form · 🟧 detected; "
    "**red = unlikely** (bad format / out of vocab"
    + (" / disagrees with DB record)." if LV == "L3" else ").")
)

scale = 560 / CANVAS_W
view_h = int(CANVAS_H * scale) + 12
gl, gr = st.columns(2)
with gl:
    st.markdown("**Crop overlay**")
    components.html(overlay_crops_html(L["proc"], scale), height=view_h)
with gr:
    st.markdown(f"**Extracted fields · {LV}**")
    components.html(
        overlay_fields_html(L["proc"], record, flags, prov, scale), height=view_h)

st.divider()
st.subheader(f"Review queue · {len(flags)}")
if not flags:
    st.info("Nothing flagged — every field looks plausible at this level.")
else:
    with st.form("review"):
        for f, reason in flags.items():
            if f == "__record__":
                st.error(reason)
                continue
            st.text_input(f"{f} — {reason}", value=str(record.get(f, "")),
                          key=f"fix_{f}")
        if st.form_submit_button("✅ Confirm & register visit"):
            st.success("Visit registered and corrections saved (demo).")

with st.expander("Level progression  (L1 → L2 → L3)"):
    prog = [{"field": f,
             "L1": show_none(res["L1"].get(f, "")),
             "L2": show_none(res["L2"].get(f, "")),
             "L3": show_none(res["L3"].get(f, ""))}
            for f in field_map["fields"]]
    st.dataframe(pd.DataFrame(prog), hide_index=True, use_container_width=True)
