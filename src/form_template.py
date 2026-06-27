"""
form_template_v2.py — Scannable medical intake form (v2) generator.

Renders a BLANK form template plus a field-coordinate map (field_map_v2.json).
Design follows OCR scannable-form best practice:
  * 3 QR-style corner finder patterns -> perspective/deskew registration
  * a scannable form-ID QR (template self-identification)
  * comb (one-box-per-character) cells for fixed-format fields
  * checkboxes for categorical fields (blood type, gender)
  * single light boxes for free text
  * all field guides in light gray "dropout" tone -> removed in preprocessing
  * labels sit ABOVE the entry area, never inside it

Outputs (to ../data/template/):
  blank_form.png         high-res blank template (raster)
  blank_form.pdf         print-ready A4 PDF
  field_map.json         every field's type + pixel boxes (post-deskew template space)
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import qrcode

# ---------------------------------------------------------------- canvas / theme
W, H = 1654, 2339
BG          = (255, 255, 255)
INK         = (20, 20, 20)
LABEL       = (55, 55, 55)
SECTION_BG  = (232, 232, 232)
SECTION_TXT = (35, 35, 35)
GUIDE       = (176, 176, 176)
GUIDE_FILL  = (245, 245, 245)
SEP         = (150, 150, 150)

MARGIN = 150
CX0, CX1 = MARGIN, W - MARGIN
CW = CX1 - CX0

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
def font(sz, bold=False):
    f = f"{FONT_DIR}/DejaVuSans{'-Bold' if bold else ''}.ttf"
    try:    return ImageFont.truetype(f, sz)
    except Exception: return ImageFont.load_default()

F_TITLE = font(46, True)
F_SUB   = font(21)
F_SEC   = font(26, True)
F_LABEL = font(22, True)
F_HINT  = font(16)
F_CB    = font(24, True)
F_SEP   = font(34, True)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
field_map = {}

# ---------------------------------------------------------------- helpers
def text_center(cx, y, s, fnt, fill=INK):
    w = d.textlength(s, font=fnt)
    d.text((cx - w/2, y), s, font=fnt, fill=fill)

def finder(cx, cy, size=104):
    h = size/2
    d.rectangle([cx-h, cy-h, cx+h, cy+h], fill=INK)
    w2 = size*0.71/2
    d.rectangle([cx-w2, cy-w2, cx+w2, cy+w2], fill=BG)
    w3 = size*0.43/2
    d.rectangle([cx-w3, cy-w3, cx+w3, cy+w3], fill=INK)
    return {"cx": cx, "cy": cy, "size": size}

def anchor_tick(x, y):
    d.line([x, y, x+14, y], fill=GUIDE, width=3)
    d.line([x, y, x, y+14], fill=GUIDE, width=3)

def label(x, y, s):
    d.text((x, y), s.upper(), font=F_LABEL, fill=LABEL)
    return y + 30

def section(y, s):
    d.rounded_rectangle([CX0, y, CX1, y+40], radius=6, fill=SECTION_BG)
    d.text((CX0+16, y+7), s.upper(), font=F_SEC, fill=SECTION_TXT)
    return y + 40 + 22

def text_box(x, y, w, h, hint=""):
    anchor_tick(x, y)
    d.rounded_rectangle([x, y, x+w, y+h], radius=8, outline=GUIDE, width=2, fill=GUIDE_FILL)
    if hint:
        d.text((x+12, y+h-24), hint, font=F_HINT, fill=GUIDE)
    return [x, y, w, h]

def comb(x, y, groups, seps=None, cell=42, ch=58, gap=9, gpad=20):
    seps = seps or []
    anchor_tick(x, y)            # per-field registration tick (matches text_box)
    cells = []
    cx = x
    for gi, n in enumerate(groups):
        for _ in range(n):
            d.rounded_rectangle([cx, y, cx+cell, y+ch], radius=6,
                                 outline=GUIDE, width=2, fill=GUIDE_FILL)
            cells.append([cx, y, cell, ch])
            cx += cell + gap
        if gi < len(groups)-1:
            s = seps[gi] if gi < len(seps) else ""
            if s:
                d.text((cx-gap+3, y+ch/2-22), s, font=F_SEP, fill=SEP)
                cx += gpad
            else:
                cx += gpad - gap
    return cells, cx

def checkboxes(x, y, options, box=34, gap_after_box=10, group_gap=46, tick=True):
    out = {}
    if tick:
        anchor_tick(x, y)            # per-field registration tick (matches text_box/comb)
    cx = x
    for opt in options:
        d.rounded_rectangle([cx, y, cx+box, y+box], radius=5, outline=GUIDE, width=2, fill=GUIDE_FILL)
        out[opt] = [cx, y, box, box]
        cx += box + gap_after_box
        d.text((cx, y+box/2-15), opt, font=F_CB, fill=LABEL)
        cx += d.textlength(opt, font=F_CB) + group_gap
    return out, cx

# ---------------------------------------------------------------- registration
finders = [
    finder(CX0+10, MARGIN-30),
    finder(CX1-10, MARGIN-30),
    finder(CX0+10, H-MARGIN+30),
]
qr = qrcode.make("FORMv2|medical_intake|tmpl-0001", box_size=4, border=2).convert("RGB")
qr = qr.resize((150, 150), Image.NEAREST)
qr_x, qr_y = CX1-150+10, H-MARGIN-90
img.paste(qr, (qr_x, qr_y))
d.text((qr_x-2, qr_y+152), "FORM v2 · tmpl-0001", font=F_HINT, fill=LABEL)

# ---------------------------------------------------------------- header
text_center(W/2, MARGIN-55, "MEDICAL INTAKE FORM", F_TITLE)
text_center(W/2, MARGIN-2,
            "CONFIDENTIAL · FOR MEDICAL USE ONLY — Protected Health Information",
            F_SUB, LABEL)

y = MARGIN + 70

# ============================================================ SECTION A — IDENTITY
y = section(y, "A · Patient Identification")

half = (CW - 40) // 2
ny = label(CX0, y, "Last Name")
field_map["last_name"] = {"type": "text_box", "box": text_box(CX0, ny, half, 60)}
label(CX0+half+40, y, "First Name")
field_map["first_name"] = {"type": "text_box", "box": text_box(CX0+half+40, ny, half, 60)}
y = ny + 60 + 22

y = label(CX0, y, "Mailing Address")
field_map["address"] = {"type": "text_box", "box": text_box(CX0, y, CW, 60)}
y += 60 + 22

y_email = label(CX0, y, "Email")
field_map["email"] = {"type": "text_box", "box": text_box(CX0, y_email, half, 60)}
label(CX0+half+40, y, "Department")
field_map["department"] = {"type": "text_box", "box": text_box(CX0+half+40, y_email, half, 60)}
y = y_email + 60 + 26

# DOB | Age | Gender row
row_y = y
ly = label(CX0, row_y, "Date of Birth")
dob_cells, _ = comb(CX0, ly, [2, 2, 4], seps=["/", "/"])
field_map["date_of_birth"] = {"type": "comb", "format": "DD/MM/YYYY", "groups": [2,2,4], "cells": dob_cells}

age_x = CX0 + 560
label(age_x, row_y, "Age")
age_cells, _ = comb(age_x, ly, [3])
field_map["age"] = {"type": "comb", "format": "NNN", "groups": [3], "cells": age_cells}

g_x = age_x + 240
label(g_x, row_y, "Gender")
g_boxes, _ = checkboxes(g_x, ly+4, ["M", "F", "Other"])
field_map["gender"] = {"type": "checkbox", "options": g_boxes}
y = ly + 58 + 30

# SSN | Insurance row
row_y = y
ly = label(CX0, row_y, "Social Security Number")
ssn_cells, _ = comb(CX0, ly, [3, 2, 4], seps=["-", "-"])
field_map["ssn"] = {"type": "comb", "format": "XXX-XX-XXXX", "groups": [3,2,4], "cells": ssn_cells}

ins_x = CX0 + 700
label(ins_x, row_y, "Insurance Number")
ins_cells, _ = comb(ins_x, ly, [3, 4, 4], seps=[" ", " "])
field_map["insurance_number"] = {"type": "comb", "format": "11 chars", "groups": [3,4,4], "cells": ins_cells}
y = ly + 58 + 30

# Phone | Blood type row
row_y = y
ly = label(CX0, row_y, "Phone Number")
ph_cells, _ = comb(CX0, ly, [3, 3, 4], seps=["-", "-"])
field_map["phone_number"] = {"type": "comb", "format": "NNN-NNN-NNNN", "groups": [3,3,4], "cells": ph_cells}

bt_x = CX0 + 700
label(bt_x, row_y, "Blood Type")
grp_boxes, after = checkboxes(bt_x, ly+4, ["A", "B", "AB", "O"], group_gap=30)
rh_boxes, _ = checkboxes(after+20, ly+4, ["+", "-"], group_gap=30, tick=False)  # one tick per field
field_map["blood_type"] = {"type": "checkbox", "group": grp_boxes, "rh": rh_boxes}
y = ly + 58 + 34

# ============================================================ SECTION B — VISIT
y = section(y, "B · Visit Details")

row_y = y
ly = label(CX0, row_y, "Date of Visit")
dv_cells, _ = comb(CX0, ly, [2, 2, 4], seps=["/", "/"])
field_map["date_of_visit"] = {"type": "comb", "format": "DD/MM/YYYY", "groups": [2,2,4], "cells": dv_cells}

doc_x = CX0 + 560
doc_w = CX1 - doc_x
label(doc_x, row_y, "Attending Physician")
field_map["doctor_name"] = {"type": "text_box", "box": text_box(doc_x, ly, doc_w, 58)}
y = ly + 58 + 26

y = label(CX0, y, "Chief Complaint / Primary Symptom")
field_map["chief_complaint"] = {"type": "text_box", "box": text_box(CX0, y, CW, 96)}
y += 96 + 26

# ============================================================ SECTION C — MEDICAL
y = section(y, "C · Medical Information")

y = label(CX0, y, "Medical History")
field_map["medical_history"] = {"type": "text_box", "box": text_box(CX0, y, CW, 110)}
y += 110 + 22

y = label(CX0, y, "Known Allergies")
field_map["allergies"] = {"type": "text_box", "box": text_box(CX0, y, CW, 60)}
y += 60 + 22

y = label(CX0, y, "Current Medications")
field_map["current_medications"] = {"type": "text_box", "box": text_box(CX0, y, CW, 60)}
y += 60 + 22

# Emergency contact — name (text) + phone (comb)
row_y = y
ly = label(CX0, row_y, "Emergency Contact Name")
field_map["emergency_contact_name"] = {"type": "text_box", "box": text_box(CX0, ly, 720, 60)}
ecp_x = CX0 + 760
label(ecp_x, row_y, "Emergency Contact Phone")
ecp_cells, _ = comb(ecp_x, ly, [3, 3, 4], seps=["-", "-"])
field_map["emergency_contact_phone"] = {"type": "comb", "format": "NNN-NNN-NNNN", "groups": [3,3,4], "cells": ecp_cells}
y = ly + 60 + 22

# ---------------------------------------------------------------- save
OUT = Path(__file__).resolve().parent.parent / "data" / "template"
OUT.mkdir(parents=True, exist_ok=True)
img.save(OUT / "blank_form.png", dpi=(200, 200))
img.save(OUT / "blank_form.pdf", "PDF", resolution=200.0)   # print-ready A4 (8.27 x 11.69 in)

meta = {
    "form_version": "v2",
    "template_id": "tmpl-0001",
    "canvas": {"w": W, "h": H},
    "finder_patterns": finders,
    "qr": {"box": [qr_x, qr_y, 150, 150], "payload": "FORMv2|medical_intake|tmpl-0001"},
    "guide_color_gray": GUIDE[0],
    "note": "All field boxes are in deskewed template pixel space. After detecting the "
            "3 finder patterns, warp the scan to this canvas, then crop fields by these boxes.",
    "fields": field_map,
}
with open(OUT / "field_map.json", "w") as f:
    json.dump(meta, f, indent=2)

print("wrote", OUT / "blank_form.png")
print("wrote", OUT / "blank_form.pdf")
print("wrote", OUT / "field_map.json")
print("fields:", len(field_map), "| last y cursor:", y, "/", H)
