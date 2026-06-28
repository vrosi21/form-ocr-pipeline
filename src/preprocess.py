"""
preprocess.py — importable port of notebook 02's scan→crops pipeline.

Notebook 02 (`02_preprocessing.ipynb`) is the locked source of truth for how a
raw photographed form becomes the cleaned per-field crops the OCR stack reads.
That logic lived only in notebook cells, so the Streamlit app could process the
100 forms already on disk but could NOT process a freshly uploaded scan. This
module lifts those exact functions out — verbatim behaviour — and parameterises
them by `field_map` (the notebook used module-level globals W/H/FIELDS).

Pipeline for one form (BGR image in):
    detect_finders -> deskew (homography to the 1654x2339 template canvas)
    -> normalize (CLAHE, kills yellow cast) -> drop_guides (light gray -> white)
    -> per-field crop + clean_field (+ trim_x for text) -> checkbox pixel reads

`process_form(raw, field_map, out_dir)` writes `<out_dir>/processed.png` plus the
`<field>.png` / `<field>_<cell>.png` crops the rest of the pipeline already
expects, and returns the checkbox summary in `preprocess_report.json` shape.

Mirrors nb02 cells 3-5; thresholds kept identical so on-disk and freshly
processed forms read the same.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# Thresholds — identical to nb02 cell 1.
DROP_THRESH = 150     # pixels lighter than this -> white (guide/background dropout)
CHECK_THRESH = 0.08   # inner dark-pixel fraction above which a checkbox is "ticked"


# --------------------------------------------------------------------------- #
# Geometry: finder-pattern detection + deskew  (nb02 cell 3)
# --------------------------------------------------------------------------- #
def detect_finders(img, W, H):
    """Locate the 3 QR-style finder patterns -> their centres (unordered)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, hier = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    cand = []
    for i, c in enumerate(cnts):
        if cv2.contourArea(c) < 0.0005 * W * H:        # ignore tiny (incl. QR mini-finders)
            continue
        approx = cv2.approxPolyDP(c, 0.05 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if not (0.75 < w / float(h) < 1.33):
            continue
        depth, child = 0, hier[i][2]                   # nested squares: black->white->black
        while child != -1:
            depth += 1
            child = hier[child][2]
        if depth >= 2:
            cand.append((cv2.contourArea(c), (x + w / 2.0, y + h / 2.0)))
    cand.sort(reverse=True)
    return [c[1] for c in cand[:3]]


def order_TL_TR_BL(pts):
    pts = sorted(pts, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    return [top[0], top[1], pts[2]]


def deskew(img, field_map):
    """Warp a raw scan onto the canonical template canvas, or None if 3 finders
    aren't found (caller should treat that as a failed scan)."""
    W = field_map["canvas"]["w"]
    H = field_map["canvas"]["h"]
    tpl_find = [(f["cx"], f["cy"]) for f in field_map["finder_patterns"]]  # [TL, TR, BL]
    f = detect_finders(img, W, H)
    if len(f) != 3:
        return None
    M = cv2.getAffineTransform(np.float32(order_TL_TR_BL(f)), np.float32(tpl_find))
    return cv2.warpAffine(img, M, (W, H), borderValue=(255, 255, 255))


# --------------------------------------------------------------------------- #
# Tone + guide dropout  (nb02 cell 4)
# --------------------------------------------------------------------------- #
def normalize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def drop_guides(gray, thresh=DROP_THRESH):
    # keep dark ink as-is; everything lighter (guide boxes, paper) -> white
    return np.where(gray < thresh, gray, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Per-field crop + clean  (nb02 cell 5)
# --------------------------------------------------------------------------- #
def crop(img, box):
    x, y, w, h = (int(round(v)) for v in box)   # field_map may hold float coords
    return img[y:y + h, x:x + w]


def clean_field(gray, is_text=False, min_area=8, lt=6, depth=0.16, frac=0.62):
    """Strip residual box edges/frames + dust from a field crop, keep handwriting.
    Verbatim from nb02 cell 5 (rim whiten -> border darkness scan -> text underline
    subtraction -> connected-component edge/frame/dust pass)."""
    h, w = gray.shape
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw[:3, :] = 0; bw[-3:, :] = 0; bw[:, :3] = 0; bw[:, -3:] = 0          # rim whiten

    dh, dw = max(2, int(h * depth)), max(2, int(w * depth))
    for r in list(range(dh)) + list(range(h - dh, h)):
        if bw[r, :].mean() >= frac * 255:
            bw[r, :] = 0
    for c in list(range(dw)) + list(range(w - dw, w)):
        if bw[:, c].mean() >= frac * 255:
            bw[:, c] = 0

    if is_text:
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
        rf = closed.mean(axis=1) / 255.0
        bw[rf >= 0.80, :] = 0

    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    keep = np.zeros_like(bw)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        fill = area / float(cw * ch)
        elong = max(cw, ch) / max(1, min(cw, ch))
        t = (x <= 1) + (y <= 1) + (x + cw >= w - 1) + (y + ch >= h - 1)
        if (y <= 1 and y + ch >= h - 1) and cw <= lt:                    # straight full-height edge
            continue
        if (x <= 1 and x + cw >= w - 1) and ch <= lt:                    # straight full-width edge
            continue
        if cw <= lt and ch >= 0.5 * h and (x <= 0.10 * w or x + cw >= 0.90 * w):   # thin tall side edge
            continue
        if ((x <= 3) + (y <= 3) + (x + cw >= w - 3) + (y + ch >= h - 3)) >= 2 and area < 120:  # corner fragment
            continue
        if t >= 3 and fill < 0.25:                                       # hollow box frame
            continue
        if t >= 2 and fill < 0.18 and elong > 4 and min(cw, ch) <= 2 * lt:   # diagonal/corner fragment
            continue
        if not is_text:
            if cw >= 0.85 * w and ch <= lt:                              # cell top/bottom edge
                continue
            if ch >= 0.85 * h and cw <= lt and (x <= 0.18 * w or x + cw >= 0.82 * w):  # cell side edge
                continue
        keep[lab == i] = 255
    return 255 - keep


def trim_x(gray, margin=16, min_run=10):
    """Trim a TEXT crop on the x-axis to the handwriting + a small margin
    (vertical-ink-run test so faint underlines/specks don't widen the crop)."""
    h, w = gray.shape
    bw = gray < 128
    run = np.zeros(w, int)
    mx = np.zeros(w, int)
    for r in range(h):
        run = np.where(bw[r], run + 1, 0)
        mx = np.maximum(mx, run)
    cols = np.where(mx >= min_run)[0]
    if len(cols) == 0:
        return gray[:, :min(w, 60)]                # empty field -> small strip
    return gray[:, max(0, cols[0] - margin):min(w, cols[-1] + 1 + margin)]


def checkbox_state(img, box, thresh=CHECK_THRESH, inset=0.28):
    """True/dark-fraction for a checkbox; measures only the inner area so the
    light-gray outline never counts as a mark."""
    x, y, w, h = (int(round(v)) for v in box)
    mx, my = int(w * inset), int(h * inset)
    cell = img[y + my:y + h - my, x + mx:x + w - mx]
    if cell.size == 0:
        return False, 0.0
    dark = float((cell < 110).mean())
    return dark > thresh, round(dark, 3)


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
def to_processed_page(raw, field_map):
    """raw BGR -> (processed_gray | None). The deskewed, normalized, guide-dropped
    full page in template space; None if finder detection failed."""
    desk = deskew(raw, field_map)
    if desk is None:
        return None
    return drop_guides(normalize(desk))


def crops_from_page(proc, field_map):
    """Cut every field crop out of an already-processed page (in memory).
    Returns {field: gray} for text_box, {field: [gray,...]} for comb cells,
    plus a `checkboxes` dict matching preprocess_report.json shape."""
    crops, checkboxes = {}, {}
    for field, spec in field_map["fields"].items():
        t = spec["type"]
        if t == "text_box":
            crops[field] = trim_x(clean_field(crop(proc, spec["box"]), is_text=True))
        elif t == "comb":
            crops[field] = [clean_field(crop(proc, cell), is_text=False)
                            for cell in spec["cells"]]
        elif t == "checkbox":
            groups = ({"options": spec.get("options", {})} if "options" in spec
                      else {"group": spec.get("group", {}), "rh": spec.get("rh", {})})
            picked = []
            for gname, opts in groups.items():
                for opt, box in opts.items():
                    checked, _ = checkbox_state(proc, box)
                    if checked:
                        picked.append(f"{gname}:{opt}" if gname != "options" else opt)
            checkboxes[field] = picked
    return crops, checkboxes


def process_form(raw, field_map, out_dir):
    """Full scan->disk pipeline for a single uploaded form.

    raw      : BGR image (e.g. cv2.imread / decoded upload bytes)
    out_dir  : directory to write `processed.png` + per-field crops into
    returns  : {"ok": bool, "reason"?: str, "processed": path, "checkboxes": {...}}
               on success — the same crop layout the OCR pipeline reads on disk.
    """
    proc = to_processed_page(raw, field_map)
    if proc is None:
        return {"ok": False, "reason": "finder detection failed (need all 3 corner patterns)"}

    os.makedirs(out_dir, exist_ok=True)
    proc_path = os.path.join(out_dir, "processed.png")
    cv2.imwrite(proc_path, proc)

    crops, checkboxes = crops_from_page(proc, field_map)
    for field, data in crops.items():
        if isinstance(data, list):
            for i, cell in enumerate(data):
                cv2.imwrite(os.path.join(out_dir, f"{field}_{i}.png"), cell)
        else:
            cv2.imwrite(os.path.join(out_dir, f"{field}.png"), data)
    return {"ok": True, "processed": proc_path, "checkboxes": checkboxes}


def decode_upload(file_bytes):
    """Decode raw uploaded image bytes -> BGR ndarray (None if not an image)."""
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
