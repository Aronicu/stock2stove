import json
import os
import sys

import cv2
import docx
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor
import win32com.client


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFENSE_DIR = os.path.join(PROJECT_ROOT, "docs", "defense")
M4_DIR = os.path.join(PROJECT_ROOT, "docs", "milestone4")
LOGO_PATH = os.path.join(PROJECT_ROOT, "images", "logo.png")

BG_LIGHT = (250, 250, 249)
BG_DARK = (17, 24, 39)
INK_DARK = (24, 24, 27)
INK_MUTED = (100, 116, 139)
INK_LIGHT = (248, 250, 252)
INK_LIGHT_MUTED = (148, 163, 184)
ACCENT = (13, 148, 136)
ACCENT_SOFT = (240, 253, 250)
CARD_BG = (255, 255, 255)
BORDER_LIGHT = (226, 232, 240)
DANGER_INK = (185, 28, 28)
DANGER_SOFT = (254, 242, 242)
FONT_MAIN = "Segoe UI"


def ole_rgb(rgb):
    return rgb[0] + (rgb[1] << 8) + (rgb[2] << 16)


def load_summary():
    summary_path = os.path.join(M4_DIR, "milestone4_evaluation_summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_slide_bg(slide, rgb):
    bg = slide.Shapes.AddShape(1, 0, 0, 960, 540)
    bg.Fill.ForeColor.RGB = ole_rgb(rgb)
    bg.Line.Visible = 0
    return bg


def add_rect(slide, left, top, width, height, fill_rgb, border_rgb=None, border_pt=1.0, rounded=False):
    shape_type = 5 if rounded else 1
    shp = slide.Shapes.AddShape(shape_type, left, top, width, height)
    if rounded:
        shp.Adjustments[1] = 0.04
    shp.Fill.ForeColor.RGB = ole_rgb(fill_rgb)
    if border_rgb is None:
        shp.Line.Visible = 0
    else:
        shp.Line.Visible = -1
        shp.Line.ForeColor.RGB = ole_rgb(border_rgb)
        shp.Line.Weight = border_pt
    return shp


def add_text(slide, left, top, width, height, text, size=14, bold=False, color=INK_DARK, align=1, line_spacing=1.2):
    tb = slide.Shapes.AddTextbox(1, left, top, width, height)
    tf = tb.TextFrame
    tf.WordWrap = -1
    tf.MarginLeft = 2
    tf.MarginRight = 2
    tf.MarginTop = 2
    tf.MarginBottom = 2
    tr = tf.TextRange
    tr.Text = text
    tr.Font.Name = FONT_MAIN
    tr.Font.Size = size
    tr.Font.Bold = -1 if bold else 0
    tr.Font.Color.RGB = ole_rgb(color)
    tr.ParagraphFormat.Alignment = align
    tr.ParagraphFormat.SpaceWithin = line_spacing
    return tb


def add_minimal_header(slide, kicker, title, slide_num, total_slides):
    set_slide_bg(slide, BG_LIGHT)
    add_text(slide, 56, 30, 500, 20, kicker.upper(), size=10, bold=True, color=ACCENT)
    add_text(slide, 800, 30, 104, 20, f"{slide_num:02d} / {total_slides:02d}", size=10, bold=False, color=INK_MUTED, align=3)
    add_text(slide, 56, 52, 848, 44, title, size=26, bold=True, color=INK_DARK, line_spacing=1.08)
    add_rect(slide, 56, 104, 848, 1.2, BORDER_LIGHT)
    add_text(slide, 56, 508, 400, 18, "Stock-to-Stove  |  CSST 106 Final Project Defense", size=9, bold=False, color=INK_MUTED)
    add_text(slide, 600, 508, 304, 18, "Team Kaggleing", size=9, bold=False, color=INK_MUTED, align=3)


def add_fitted_image(slide, img_path, left, top, max_w, max_h, center_in_box=True):
    img = cv2.imread(img_path)
    ih, iw = img.shape[:2]
    scale = min(max_w / float(iw), max_h / float(ih))
    w = iw * scale
    h = ih * scale
    x = left + (max_w - w) / 2.0 if center_in_box else left
    y = top + (max_h - h) / 2.0 if center_in_box else top
    return slide.Shapes.AddPicture(os.path.abspath(img_path), 0, -1, x, y, w, h)


def add_bullet_list(slide, left, top, width, height, items, size=14.5, color=INK_DARK, space_after=10):
    tb = slide.Shapes.AddTextbox(1, left, top, width, height)
    tf = tb.TextFrame
    tf.WordWrap = -1
    tf.MarginLeft = 2
    tf.MarginRight = 2
    tf.MarginTop = 2
    tf.MarginBottom = 2
    tr = tf.TextRange
    tr.Text = "\r".join([f"•  {item}" for item in items])
    for i in range(1, tr.Paragraphs().Count + 1):
        p = tr.Paragraphs(i)
        p.Font.Name = FONT_MAIN
        p.Font.Size = size
        p.Font.Color.RGB = ole_rgb(color)
        p.ParagraphFormat.SpaceAfter = space_after
        p.ParagraphFormat.SpaceWithin = 1.22
    return tb


def add_kpi_card(slide, left, top, width, height, value, label, sublabel):
    add_rect(slide, left, top, width, height, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
    add_rect(slide, left, top, 4, height, ACCENT)
    add_text(slide, left + 18, top + 14, width - 26, 44, value, size=32, bold=True, color=INK_DARK)
    add_text(slide, left + 18, top + 60, width - 26, 22, label, size=12, bold=True, color=ACCENT)
    add_text(slide, left + 18, top + 82, width - 26, 24, sublabel, size=10.5, bold=False, color=INK_MUTED)


def add_minimal_table(slide, left, top, width, height, headers, rows, col_widths, font_size=10.5, row_h=28):
    tbl_shape = slide.Shapes.AddTable(len(rows) + 1, len(headers), left, top, width, height)
    tbl = tbl_shape.Table
    for c_idx, cw in enumerate(col_widths, start=1):
        tbl.Columns(c_idx).Width = cw
    for c_idx, h_text in enumerate(headers, start=1):
        cell = tbl.Cell(1, c_idx)
        cell.Shape.Fill.ForeColor.RGB = ole_rgb(BG_DARK)
        tr = cell.Shape.TextFrame.TextRange
        tr.Text = h_text
        tr.Font.Name = FONT_MAIN
        tr.Font.Size = font_size
        tr.Font.Bold = -1
        tr.Font.Color.RGB = ole_rgb(INK_LIGHT)
        tr.ParagraphFormat.Alignment = 2 if c_idx > 1 else 1
    tbl.Rows(1).Height = row_h + 2
    for r_idx, row_vals in enumerate(rows, start=2):
        tbl.Rows(r_idx).Height = row_h
        bg = CARD_BG if r_idx % 2 == 0 else (244, 246, 248)
        for c_idx, val in enumerate(row_vals, start=1):
            cell = tbl.Cell(r_idx, c_idx)
            cell.Shape.Fill.ForeColor.RGB = ole_rgb(bg)
            tr = cell.Shape.TextFrame.TextRange
            tr.Text = str(val)
            tr.Font.Name = FONT_MAIN
            tr.Font.Size = font_size
            tr.Font.Bold = -1 if c_idx == 4 else 0
            tr.Font.Color.RGB = ole_rgb(ACCENT if c_idx == 4 else INK_DARK)
            tr.ParagraphFormat.Alignment = 2 if c_idx > 1 else 1
    return tbl_shape


def build_slides(summary):
    total_slides = 12
    pptx_path = os.path.join(DEFENSE_DIR, "Stock2Stove_Defense_Slides.pptx")
    pdf_path = os.path.join(DEFENSE_DIR, "Stock2Stove_Defense_Slides.pdf")

    pp = win32com.client.Dispatch("PowerPoint.Application")
    pp.Visible = 1
    prs = pp.Presentations.Add()
    prs.PageSetup.SlideWidth = 960
    prs.PageSetup.SlideHeight = 540

    s1 = prs.Slides.Add(1, 12)
    set_slide_bg(s1, BG_DARK)
    add_rect(s1, 64, 96, 54, 4, ACCENT)
    add_text(s1, 64, 116, 620, 22, "CSST 106  •  COMPUTER VISION FINAL PROJECT DEFENSE", size=11, bold=True, color=ACCENT)
    add_text(s1, 64, 148, 650, 72, "Stock-to-Stove", size=48, bold=True, color=INK_LIGHT)
    add_text(
        s1, 64, 222, 620, 68,
        "Automated Pantry Inventory Detection\n& Zero-Waste Recipe Application",
        size=20, bold=False, color=INK_LIGHT_MUTED, line_spacing=1.25
    )
    add_rect(s1, 64, 346, 832, 1, (51, 65, 85))
    add_text(
        s1, 64, 368, 560, 74,
        "Team Kaggleing\nAron Gabriel L. Ogayon (Lead)  •  Allain A. Kumar (Docs)  •  Janvher Lucas J. Sarmiento (Dev)",
        size=13, bold=False, color=INK_LIGHT, line_spacing=1.45
    )
    add_text(
        s1, 64, 456, 640, 24,
        "YOLOv8n (6.2 MB)  |  0.819 mAP@0.50  |  23.7 FPS CPU  |  Web + Android + CLI",
        size=11.5, bold=True, color=ACCENT
    )
    if os.path.exists(LOGO_PATH):
        add_rect(s1, 718, 114, 188, 188, CARD_BG, None, 0, rounded=True)
        add_fitted_image(s1, LOGO_PATH, 730, 126, 164, 164)

    s2 = prs.Slides.Add(2, 12)
    add_minimal_header(s2, "01 / Problem", "Why Pantry Inventory Needs Computer Vision", 2, total_slides)
    pillars = [
        ("01", "Hidden Food Waste", "Rear-shelf items are forgotten, causing duplicate buys and expired groceries."),
        ("02", "Decision Fatigue", "Looking at cluttered shelves makes meal planning slow and pushes takeout."),
        ("03", "Barcodes Fail", "Manual scanning breaks on loose produce, unlabeled bags, and dark cupboards."),
    ]
    for idx, (num, head, body) in enumerate(pillars):
        x = 56 + idx * 292
        add_rect(s2, x, 150, 264, 290, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
        add_text(s2, x + 24, 174, 80, 36, num, size=24, bold=True, color=ACCENT)
        add_text(s2, x + 24, 224, 216, 32, head, size=18, bold=True, color=INK_DARK)
        add_text(s2, x + 24, 268, 216, 140, body, size=14, bold=False, color=INK_MUTED, line_spacing=1.35)
    add_text(
        s2, 56, 464, 848, 28,
        "Goal: Pan a phone camera across a shelf -> detect & count items -> generate 3 zero-waste recipes.",
        size=13.5, bold=True, color=INK_DARK
    )

    s3 = prs.Slides.Add(3, 12)
    add_minimal_header(s3, "02 / Objectives", "Engineering Targets & Scope", 3, total_slides)
    goals = [
        ("Unified 8-Class Detection", "Detect 5 pantry staples and 3 refrigerator essentials under real cupboard shadows (<100 lux) and clutter."),
        ("Rigorous Baseline Benchmark", "Quantify classical ORB & SIFT + RANSAC failure modes against fine-tuned YOLOv8n on a frozen test set."),
        ("Real-Time CPU Application", "Achieve >= 15 FPS on laptop CPU via one cached model serving Web UI, Android CameraX, and CLI."),
        ("Actionable Recipe Output", "Feed temporally confirmed inventory counts into an LLM to produce 3 practical zero-waste recipes."),
    ]
    for idx, (g_title, g_desc) in enumerate(goals):
        col = idx % 2
        row = idx // 2
        x = 56 + col * 438
        y = 140 + row * 172
        add_rect(s3, x, y, 410, 148, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
        add_rect(s3, x, y, 410, 4, ACCENT)
        add_text(s3, x + 22, y + 20, 366, 30, f"{idx + 1}.  {g_title}", size=16.5, bold=True, color=INK_DARK)
        add_text(s3, x + 22, y + 58, 366, 76, g_desc, size=13, bold=False, color=INK_MUTED, line_spacing=1.3)

    s4 = prs.Slides.Add(4, 12)
    add_minimal_header(s4, "03 / Dataset", "8-Class Unified Dataset & Frozen Splits", 4, total_slides)
    add_kpi_card(s4, 56, 132, 198, 112, "813", "Annotated Images", "2,480 bounding boxes")
    add_kpi_card(s4, 272, 132, 198, 112, "104", "Locked Test Set", "261 GT boxes (frozen)")
    add_kpi_card(s4, 488, 132, 198, 112, "115", "Safety Negatives", "Non-meat pantry set")
    add_kpi_card(s4, 704, 132, 200, 112, "60", "Video Pan Frames", "640x480 @ 15 FPS")

    add_rect(s4, 56, 274, 412, 204, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
    add_text(s4, 80, 294, 364, 28, "Pantry Zone (5 Classes)", size=16, bold=True, color=ACCENT)
    add_bullet_list(
        s4, 80, 330, 364, 136,
        ["pasta_noodles", "canned_tomato", "canned_tuna_meat", "cereal_grains", "coffee_condiment"],
        size=13.5, space_after=6
    )

    add_rect(s4, 492, 274, 412, 204, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
    add_text(s4, 516, 294, 364, 28, "Refrigerator Zone (3 Classes)", size=16, bold=True, color=ACCENT)
    add_bullet_list(
        s4, 516, 330, 364, 96,
        ["milk_dairy", "eggs", "fresh_produce"],
        size=13.5, space_after=8
    )
    add_text(
        s4, 516, 426, 364, 38,
        "Split: 605 Train  |  104 Valid  |  104 Held-Out Test\nSources: Freiburg, Kitchen-YOLO, Fridge, Pantry & Negatives",
        size=11, bold=False, color=INK_MUTED, line_spacing=1.25
    )

    s5 = prs.Slides.Add(5, 12)
    add_minimal_header(s5, "04 / Methodology", "Classical Baselines vs. Fine-Tuned YOLOv8n", 5, total_slides)
    add_rect(s5, 56, 136, 408, 344, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
    add_text(s5, 80, 160, 360, 28, "Classical Baselines (ORB & SIFT)", size=17, bold=True, color=INK_DARK)
    add_text(s5, 80, 190, 360, 22, "Planar Keypoint Matching + RANSAC Homography", size=11.5, bold=False, color=INK_MUTED)
    add_bullet_list(
        s5, 80, 228, 360, 226,
        [
            "1,000 keypoints + Lowe's ratio (0.75) + RANSAC.",
            "Breaks on curved cans, shadow gradients, and blur.",
            "Normal mAP@0.50: 0.011 (ORB) and 0.025 (SIFT).",
            "Too slow for live scanning: 0.5 to 1.9 FPS on CPU.",
        ],
        size=13.5, space_after=12
    )

    add_rect(s5, 496, 136, 408, 344, ACCENT_SOFT, ACCENT, 1.2, rounded=True)
    add_text(s5, 520, 160, 360, 28, "Main Pipeline (YOLOv8n + Gated CLAHE)", size=17, bold=True, color=ACCENT)
    add_text(s5, 520, 190, 360, 22, "Single-Stage Deep Detector + Adaptive Preprocessing", size=11.5, bold=False, color=INK_DARK)
    add_bullet_list(
        s5, 520, 228, 360, 226,
        [
            "Fine-tuned YOLOv8n (6.2 MB PT / 12.19 MB ONNX).",
            "Luminance-gated Gamma + CLAHE triggers only if L < 45.",
            "Class-wise NMS (0.45) + meat gate (>= 0.76).",
            "5-frame temporal voter (K >= 3) prevents flicker.",
        ],
        size=13.5, space_after=12
    )

    s6 = prs.Slides.Add(6, 12)
    add_minimal_header(s6, "05 / Architecture", "End-to-End Computer Vision Pipeline", 6, total_slides)
    wf_img = os.path.join(M4_DIR, "workflow_diagram.png")
    add_rect(s6, 56, 128, 848, 250, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
    if os.path.exists(wf_img):
        add_fitted_image(s6, wf_img, 72, 138, 816, 230)

    stages = [
        ("1. Preprocess", "640x640 letterbox + gated CLAHE (L < 45)"),
        ("2. Infer", "Cached YOLOv8n / Gemini / ORB / SIFT"),
        ("3. Postprocess", "NMS (0.45) + meat gate + K>=3 voter"),
        ("4. Render & LLM", "Annotated HUD + 3 zero-waste recipes"),
    ]
    for idx, (st_t, st_d) in enumerate(stages):
        x = 56 + idx * 218
        add_rect(s6, x, 396, 196, 88, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
        add_text(s6, x + 14, 408, 168, 22, st_t, size=12.5, bold=True, color=ACCENT)
        add_text(s6, x + 14, 434, 168, 42, st_d, size=11, bold=False, color=INK_MUTED, line_spacing=1.2)

    s7 = prs.Slides.Add(7, 12)
    add_minimal_header(s7, "06 / Application", "Integrated Multi-Platform Interface", 7, total_slides)
    ui_img = os.path.join(M4_DIR, "application_interface_overview.png")
    add_rect(s7, 56, 130, 520, 356, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
    if os.path.exists(ui_img):
        add_fitted_image(s7, ui_img, 68, 142, 496, 332)

    apps = [
        ("Web Dashboard (GET /)", "FastAPI browser UI for image upload, video scan, webcam live feed, detector switching, and recipe cards."),
        ("Android CameraX App", "Native Kotlin + Jetpack Compose client streaming frames over WS /ws/scan with real-time box overlays."),
        ("Unified CLI (src/run.py)", "Single procedural engine powering batch evaluation, video tracking, and automated LLM recipe generation."),
    ]
    for idx, (a_t, a_d) in enumerate(apps):
        y = 130 + idx * 124
        add_rect(s7, 598, y, 306, 108, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
        add_rect(s7, 598, y, 4, 108, ACCENT)
        add_text(s7, 616, y + 12, 276, 24, a_t, size=13.5, bold=True, color=INK_DARK)
        add_text(s7, 616, y + 38, 276, 60, a_d, size=11, bold=False, color=INK_MUTED, line_spacing=1.22)

    s8 = prs.Slides.Add(8, 12)
    add_minimal_header(s8, "07 / Evaluation", "Held-Out Test Results (104 Images, 261 Boxes)", 8, total_slides)
    add_kpi_card(s8, 56, 126, 198, 110, "0.819", "Normal mAP@0.50", "F1 = 0.854  |  P = 0.866")
    add_kpi_card(s8, 272, 126, 198, 110, "0.867", "Cluttered mAP@0.50", ">= 4 items  |  F1 = 0.898")
    add_kpi_card(s8, 488, 126, 198, 110, "23.7 FPS", "CPU Throughput", "42.3 ms / frame (no GPU)")
    add_kpi_card(s8, 704, 126, 200, 110, "0.87%", "False-Meat Rate", "1 / 115 pantry negatives")

    prelim = {(r["split"], r["detector"]): r for r in summary["preliminary_results"]}
    split_labels = [
        ("normal", "Normal Daylight"),
        ("cluttered_shelf", "Cluttered (>=4 items)"),
        ("shadowed_sub100lux", "Shadow (<100 lux)"),
        ("angled_perspective", "Angled (+/-16 deg)"),
        ("motion_blur", "Motion Blur (k=11)"),
        ("occluded_20_50pct", "Occluded (35%)"),
    ]
    tbl_rows = []
    for skey, slabel in split_labels:
        orb = prelim[(skey, "orb_baseline")]
        sift = prelim[(skey, "sift_baseline")]
        yolo = prelim[(skey, "yolov8n_pipeline")]
        tbl_rows.append([
            slabel,
            f"{orb['mAP50']:.3f}",
            f"{sift['mAP50']:.3f}",
            f"{yolo['mAP50']:.3f}",
            f"{yolo['f1_score']:.3f}",
            f"{yolo['fps']:.1f} FPS",
        ])
    add_minimal_table(
        s8, 56, 256, 848, 212,
        ["Test Condition", "ORB mAP", "SIFT mAP", "YOLOv8n mAP", "YOLOv8n F1", "CPU Speed"],
        tbl_rows,
        [248, 100, 100, 140, 130, 130],
        font_size=11,
        row_h=28,
    )

    s9 = prs.Slides.Add(9, 12)
    add_minimal_header(s9, "08 / Progress", "Measurable Gains Since Checkpoint 3", 9, total_slides)
    comp_img = os.path.join(M4_DIR, "cp3_vs_cp4_comparison.png")
    add_rect(s9, 56, 128, 848, 248, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
    if os.path.exists(comp_img):
        add_fitted_image(s9, comp_img, 70, 136, 820, 232)

    gains = [
        ("+0.232 mAP", "Shadow Recovery", "Luminance-gated CLAHE lifted <100 lux mAP from 0.443 to 0.675 (no-CLAHE ablation: 0.301)."),
        ("+0.070 mAP", "Normal & Angled", "Normal mAP rose 0.749 -> 0.819; angled +/-16 deg rose +0.105 (0.483 -> 0.588)."),
        ("+4.8 FPS", "Faster CPU Speed", "Single cached model raised normal CPU throughput from 18.9 to 23.7 FPS."),
    ]
    for idx, (g_num, g_head, g_desc) in enumerate(gains):
        x = 56 + idx * 292
        add_rect(s9, x, 392, 264, 98, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
        add_text(s9, x + 16, 402, 232, 24, f"{g_num}  —  {g_head}", size=11.5, bold=True, color=ACCENT)
        add_text(s9, x + 16, 430, 232, 54, g_desc, size=10.5, bold=False, color=INK_MUTED, line_spacing=1.2)

    s10 = prs.Slides.Add(10, 12)
    add_minimal_header(s10, "09 / Error Analysis", "Operating Envelope & Known Failure Modes", 10, total_slides)
    fail_img = os.path.join(M4_DIR, "failure_gallery.png")
    add_rect(s10, 56, 128, 356, 360, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
    if os.path.exists(fail_img):
        add_fitted_image(s10, fail_img, 66, 138, 336, 340)

    add_rect(s10, 434, 128, 470, 168, ACCENT_SOFT, ACCENT, 1.0, rounded=True)
    add_text(s10, 456, 144, 426, 26, "Reliable Operating Envelope", size=15.5, bold=True, color=ACCENT)
    add_bullet_list(
        s10, 456, 178, 426, 106,
        [
            "Camera distance: 0.3 to 1.2 meters, steady pan.",
            "Lighting: >= 30 lux (gated CLAHE handles dim shelves).",
            "Clutter: <= 20% item overlap (0.867 mAP on cluttered shelves).",
        ],
        size=12.5, space_after=6
    )

    add_rect(s10, 434, 314, 470, 174, DANGER_SOFT, (252, 165, 165), 1.0, rounded=True)
    add_text(s10, 456, 330, 426, 26, "Primary Failure Causes (295 Attributed)", size=15.5, bold=True, color=DANGER_INK)
    add_bullet_list(
        s10, 456, 364, 426, 114,
        [
            "Occlusion & clutter (123): >= 35% central blockage drops recall.",
            "Motion blur (70) & viewpoint (52): fast k=11 pan smears labels.",
            "Look-alikes (12) & domain gap (20): vinegar/cereal & synthetic video.",
        ],
        size=12.5, space_after=6
    )

    s11 = prs.Slides.Add(11, 12)
    add_minimal_header(s11, "10 / Demo & Roadmap", "Live Demo Plan & Path to 100%", 11, total_slides)
    add_rect(s11, 56, 134, 412, 348, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
    add_text(s11, 80, 156, 364, 28, "Live Defense Demo Sequence", size=16.5, bold=True, color=ACCENT)
    add_bullet_list(
        s11, 80, 196, 364, 264,
        [
            "Launch FastAPI dashboard (http://127.0.0.1:8000/).",
            "Run pre-demo suite (TC01–TC16: 12/16 PASS, 9 hard cases).",
            "Switch detectors live (YOLOv8n vs. ORB/SIFT vs. Gemini).",
            "Stream native Android CameraX overlay over WebSocket.",
            "Test unseen panelist images & generate 3 recipes.",
        ],
        size=13, space_after=11
    )

    add_rect(s11, 492, 134, 412, 348, CARD_BG, BORDER_LIGHT, 1.0, rounded=True)
    add_text(s11, 516, 156, 364, 28, "Remaining Roadmap to Final Release", size=16.5, bold=True, color=INK_DARK)
    add_bullet_list(
        s11, 516, 196, 364, 264,
        [
            "R1: Record 3 real kitchen MP4 pans (+/-1 count target).",
            "R2: Stage-2 hardening + 60 look-alike negatives (mAP >= 0.85).",
            "R3: Motion-blur (k=7..13) & +/-20 deg angle augmentation.",
            "R4: Android release APK & WebSocket reconnect stress test.",
            "R5: Final technical report & freeze cp5-final release tag.",
        ],
        size=13, space_after=11
    )

    s12 = prs.Slides.Add(12, 12)
    set_slide_bg(s12, BG_DARK)
    add_rect(s12, 453, 158, 54, 4, ACCENT)
    add_text(s12, 80, 184, 800, 64, "Thank You", size=44, bold=True, color=INK_LIGHT, align=2)
    add_text(s12, 80, 254, 800, 38, "Live Demonstration  &  Panel Q&A", size=20, bold=False, color=ACCENT, align=2)
    add_text(
        s12, 80, 330, 800, 56,
        "Stock-to-Stove  |  Team Kaggleing\nOgayon, Aron Gabriel L.  •  Kumar, Allain A.  •  Sarmiento, Janvher Lucas J.",
        size=13.5, bold=False, color=INK_LIGHT_MUTED, align=2, line_spacing=1.45
    )
    add_text(
        s12, 80, 444, 800, 24,
        "python src/server/app.py    |    python src/evaluate.py --split test --weights models/best.pt",
        size=10.5, bold=False, color=(100, 116, 139), align=2
    )

    prs.SaveAs(os.path.abspath(pptx_path))
    prs.SaveAs(os.path.abspath(pdf_path), 32)
    prs.Close()
    pp.Quit()
    return pptx_path, pdf_path


def set_cell_bg_docx(cell, fill_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_pr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))


def set_cell_margins_docx(cell, top=60, bottom=60, left=100, right=100):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_pr.append(parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    ))


def set_table_borders_docx(table, color="E2E8F0"):
    tbl_pr = table._tbl.tblPr
    tbl_pr.append(parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    ))


def add_section_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = FONT_MAIN
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = RGBColor(13, 148, 136)
    return p


def add_slide_script_block(doc, slide_no, title, speaker, duration, visual_cue, script_text):
    tbl = doc.add_table(rows=2, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    set_table_borders_docx(tbl, "CBD5E1")

    c0 = tbl.cell(0, 0)
    c0.width = Inches(6.7)
    set_cell_bg_docx(c0, "F1F5F9")
    set_cell_margins_docx(c0, top=65, bottom=65, left=110, right=110)
    p0 = c0.paragraphs[0]
    p0.paragraph_format.space_before = Pt(0)
    p0.paragraph_format.space_after = Pt(0)
    p0.paragraph_format.keep_with_next = True
    r0a = p0.add_run(f"SLIDE {slide_no:02d}  —  {title}")
    r0a.font.name = FONT_MAIN
    r0a.font.size = Pt(10.5)
    r0a.font.bold = True
    r0a.font.color.rgb = RGBColor(17, 24, 39)
    r0b = p0.add_run(f"    |    Speaker: {speaker}    |    Target: {duration}")
    r0b.font.name = FONT_MAIN
    r0b.font.size = Pt(9.0)
    r0b.font.bold = False
    r0b.font.color.rgb = RGBColor(13, 148, 136)

    c1 = tbl.cell(1, 0)
    c1.width = Inches(6.7)
    set_cell_bg_docx(c1, "FFFFFF")
    set_cell_margins_docx(c1, top=80, bottom=80, left=110, right=110)

    p_cue = c1.paragraphs[0]
    p_cue.paragraph_format.space_before = Pt(0)
    p_cue.paragraph_format.space_after = Pt(5)
    rc1 = p_cue.add_run("Visual Cue: ")
    rc1.font.name = FONT_MAIN
    rc1.font.size = Pt(8.8)
    rc1.font.bold = True
    rc1.font.color.rgb = RGBColor(100, 116, 139)
    rc2 = p_cue.add_run(visual_cue)
    rc2.font.name = FONT_MAIN
    rc2.font.size = Pt(8.8)
    rc2.font.italic = True
    rc2.font.color.rgb = RGBColor(100, 116, 139)

    p_txt = c1.add_paragraph()
    p_txt.paragraph_format.space_before = Pt(0)
    p_txt.paragraph_format.space_after = Pt(0)
    p_txt.paragraph_format.line_spacing = 1.24
    rt = p_txt.add_run(script_text)
    rt.font.name = FONT_MAIN
    rt.font.size = Pt(9.8)
    rt.font.color.rgb = RGBColor(24, 24, 27)

    tr0_pr = tbl.rows[0]._tr.get_or_add_trPr()
    tr0_pr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    tr1_pr = tbl.rows[1]._tr.get_or_add_trPr()
    tr1_pr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(6)


def build_script_pdf():
    docx_path = os.path.join(DEFENSE_DIR, "Stock2Stove_Defense_Script.docx")
    pdf_path = os.path.join(DEFENSE_DIR, "Stock2Stove_Defense_Script.pdf")

    doc = docx.Document()
    for sec in doc.sections:
        sec.top_margin = Inches(0.7)
        sec.bottom_margin = Inches(0.7)
        sec.left_margin = Inches(0.9)
        sec.right_margin = Inches(0.9)

    p_kicker = doc.add_paragraph()
    p_kicker.paragraph_format.space_after = Pt(2)
    rk = p_kicker.add_run("CSST 106  •  COMPUTER VISION FINAL PROJECT DEFENSE")
    rk.font.name = FONT_MAIN
    rk.font.size = Pt(9)
    rk.font.bold = True
    rk.font.color.rgb = RGBColor(13, 148, 136)

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(3)
    rt = p_title.add_run("Stock-to-Stove — Defense Presentation Script & Q&A Guide")
    rt.font.name = FONT_MAIN
    rt.font.size = Pt(18)
    rt.font.bold = True
    rt.font.color.rgb = RGBColor(17, 24, 39)

    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(10)
    rm = p_meta.add_run(
        "Team Kaggleing: Aron Gabriel L. Ogayon (Lead)  •  Allain A. Kumar (Documentation)  •  Janvher Lucas J. Sarmiento (Developer)\n"
        "Companion Deck: Stock2Stove_Defense_Slides.pptx / .pdf  (12 Slides)   |   Estimated Spoken Runtime: ~8.5 Minutes + Live Demo"
    )
    rm.font.name = FONT_MAIN
    rm.font.size = Pt(9.2)
    rm.font.color.rgb = RGBColor(100, 116, 139)

    add_section_heading(doc, "1. Speaker Allocation & Pacing Overview")
    pace_tbl = doc.add_table(rows=4, cols=4)
    pace_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    pace_tbl.autofit = False
    set_table_borders_docx(pace_tbl, "CBD5E1")
    headers = ["Speaker", "Role", "Assigned Slides", "Est. Time"]
    widths = [Inches(2.1), Inches(1.2), Inches(2.3), Inches(1.1)]
    rows_data = [
        ["Aron Gabriel L. Ogayon", "Team Lead", "Slides 1, 2, 3, 11, 12 + Live Demo", "~3:00"],
        ["Allain A. Kumar", "Documentation", "Slides 4, 8, 10", "~2:45"],
        ["Janvher Lucas J. Sarmiento", "Developer", "Slides 5, 6, 7, 9", "~2:45"],
    ]
    for ci, h in enumerate(headers):
        cell = pace_tbl.cell(0, ci)
        cell.width = widths[ci]
        set_cell_bg_docx(cell, "111827")
        set_cell_margins_docx(cell, top=55, bottom=55, left=80, right=80)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.font.name = FONT_MAIN
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
    for ri, rvals in enumerate(rows_data, start=1):
        bg = "FFFFFF" if ri % 2 == 1 else "F8FAFC"
        for ci, val in enumerate(rvals):
            cell = pace_tbl.cell(ri, ci)
            cell.width = widths[ci]
            set_cell_bg_docx(cell, bg)
            set_cell_margins_docx(cell, top=50, bottom=50, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.name = FONT_MAIN
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(24, 24, 27)

    add_section_heading(doc, "2. Slide-by-Slide Verbatim Defense Script")

    slides_script = [
        (
            1, "Title — Stock-to-Stove", "Ogayon", "0:25",
            "Title slide on screen. Smile, make eye contact with the panel.",
            "\"Good afternoon po sa lahat. Kami po ang Team Kaggleing—ako po si Aron Ogayon, kasama sina Allain Kumar at Janvher Sarmiento. Today, ipe-present po namin ang Stock-to-Stove: isang automated pantry at refrigerator inventory detection system na naka-connect sa zero-waste recipe generator. Simple lang po ang goal namin: i-pan lang ang phone camera sa shelf, ma-detect at mabilang ang mga ingredients, at makapag-generate agad ng practical recipes—all running in real time sa standard laptop CPU.\""
        ),
        (
            2, "The Problem", "Ogayon", "0:40",
            "Gesture to the three problem cards: Hidden Food Waste, Decision Fatigue, and Barcodes Fail.",
            "\"Sa halos lahat ng bahay, pareho ang problema: puno ang cupboard o ref, pero hindi mo alam kung ano ang lulutuin. Yung mga items na natatabunan sa likod ng shelf, nakakalimutan—kaya nagkakaroon ng duplicate purchases at nasasayang na pagkain. Yung mga existing inventory apps naman, naka-rely sa manual barcode scanning, na hindi gumagana sa loose produce, unlabeled bags, at masikip na shelves. Bukod pa rito, madilim talaga ang mga cupboard—madalas under 100 lux—at naka-angle ang kuha ng camera, kaya computer vision talaga ang kailangan dito.\""
        ),
        (
            3, "Objectives & Scope", "Ogayon", "0:40",
            "Walk through the four numbered engineering targets.",
            "\"Para ma-solve ito nang maayos, nag-set po kami ng apat na engineering objectives. Una, ma-detect ang walong unified grocery classes sa pantry at refrigerator kahit madilim o cluttered ang shelf. Pangalawa, i-benchmark ang classical baselines—ORB at SIFT with RANSAC—laban sa fine-tuned YOLOv8-nano sa isang locked held-out test set. Pangatlo, ma-achieve ang at least 15 FPS sa laptop CPU gamit ang isang cached model na nagsi-serve sa Web UI, Android app, at CLI. At pang-apat, ipasa ang confirmed counts sa LLM para mag-suggest ng tatlong zero-waste recipes. Si Allain po ang magpapaliwanag ng aming dataset.\""
        ),
        (
            4, "Dataset & 8-Class Taxonomy", "Kumar", "0:50",
            "Point to the four top KPI cards, then the Pantry and Fridge class lists.",
            "\"Thank you, Aron. Ang unified dataset po namin ay may 813 annotated images at 2,480 bounding boxes—naka-split sa 605 training, 104 validation, at 104 completely locked held-out test images na may 261 ground-truth boxes. Meron din kaming dalawang safety sets: 115 non-meat pantry negatives para masukat ang false meat detections, at isang 60-frame video pan clip. Sakop ng taxonomy namin ang limang pantry staples—pasta, canned tomato, canned tuna o meat, cereal at grains, at coffee o condiments—plus tatlong refrigerator items: milk at dairy, eggs, at fresh produce. I-eexplain naman ni Janvher ang aming methodology.\""
        ),
        (
            5, "Classical Baselines vs. YOLOv8n", "Sarmiento", "0:55",
            "Compare the left card (ORB/SIFT) against the highlighted right card (YOLOv8n + Gated CLAHE).",
            "\"Para mapatunayan kung bakit kailangan ng deep learning, gumawa muna kami ng classical ORB at SIFT baselines gamit ang 1,000 keypoints, Lowe's ratio test na 0.75, at RANSAC homography. Dahil planar ang assumption ng homography, bumabagsak ito sa kurbadong lata, shadow gradients, at motion blur—0.011 hanggang 0.025 mAP lang at wala pang 2 FPS. Kaya pinalitan namin ito ng fine-tuned 6.2-megabyte YOLOv8-nano na may Checkpoint 4 upgrade: luminance-gated gamma at CLAHE preprocessing. Nagti-trigger lang ito kapag bumaba sa 45 ang mean LAB lightness, kaya nakakabawi sa madilim na shelf nang hindi nasisira ang well-lit frames. Sa post-processing, may class-wise NMS kami, 0.76 meat-safety gate, at 5-frame temporal voter na kailangan ng at least 3 hits.\""
        ),
        (
            6, "System Architecture & Pipeline", "Sarmiento", "0:40",
            "Trace the workflow diagram left-to-right across the four stages.",
            "\"Ito po ang end-to-end pipeline namin sa src/run.py. Bawat frame ay dumadaan sa apat na malinis na stages: una, preprocess para sa 640-by-640 letterboxing at gated CLAHE; pangalawa, infer gamit ang isang cached model instance sa MODEL_REGISTRY—pwedeng YOLOv8n, ORB, SIFT, o Gemini Vision; pangatlo, postprocess para sa NMS, meat-safety gate, at temporal voting; at pang-apat, render na naglalabas ng bounding boxes, latency HUD, at JSON counts para sa recipe engine.\""
        ),
        (
            7, "Integrated Multi-Platform Application", "Sarmiento", "0:40",
            "Point to the application interface overview on the left and the three entrypoints on the right.",
            "\"Hindi lang po ito nagtapos sa notebook—naka-expose ang parehong engine sa tatlong interfaces. Una, ang FastAPI browser dashboard sa localhost port 8000 para sa image upload, video scan, live webcam, live detector switching, at recipe cards. Pangalawa, ang native Kotlin at Jetpack Compose Android app na gumagamit ng CameraX at nag-stream over WebSocket para sa real-time box overlays. At pangatlo, ang batch CLI para sa reproducible testing. Si Allain naman po ang magshe-share ng aming evaluation results.\""
        ),
        (
            8, "Held-Out Test Results", "Kumar", "1:00",
            "Highlight the four hero KPI cards first, then scan down the comparison table.",
            "\"Lahat po ng numbers dito ay 100 percent reproducible gamit ang python src/evaluate.py sa locked 104-image test set. Sa normal daylight, nakakuha ang YOLOv8n ng 0.819 mAP@0.50 at 0.854 F1 sa bilis na 23.7 FPS sa CPU—malayong-malayo sa ORB na 0.011 at SIFT na 0.025. Sa cluttered shelves na may apat o higit pang items, umabot pa ito ng 0.867 mAP@0.50 at 0.898 F1. Importante rin po ang safety check namin: sa 115 non-meat pantry negatives, isa lang ang nag-false meat, or 0.87 percent false-meat rate. Kahit sa stress tests—shadow, 16-degree tilt, motion blur, at 35 percent occlusion—nanatiling 17.7 hanggang 23.7 FPS ang CPU speed habang nag-zero na ang classical baselines.\""
        ),
        (
            9, "Progress Since Checkpoint 3", "Sarmiento", "0:45",
            "Point to the CP3 vs. CP4 bar chart, especially the Shadow bar jump.",
            "\"Kung ikukumpara po ang Checkpoint 3 sa Checkpoint 4, malinaw ang improvement sa lahat ng test conditions. Pinakamalaking panalo namin ang low-light shadow recovery: tumaas ng plus 0.232 ang sub-100-lux mAP—mula 0.443 naging 0.675—kumpara sa 0.301 lang kapag walang CLAHE sa ablation test namin. Umangat din ng plus 0.070 ang normal mAP to 0.819, plus 0.105 sa angled perspective, at tumaas ng 4.8 FPS ang CPU speed dahil sa model caching.\""
        ),
        (
            10, "Error Analysis & Operating Limits", "Kumar", "0:55",
            "Point to the failure gallery on the left, then contrast Reliable Envelope vs. Failure Causes.",
            "\"Inaudit din po namin ang 295 attributed failures para maging transparent sa operating limits ng system. Reliable ang Stock-to-Stove sa 0.3 hanggang 1.2 meters na distansya, at least 30 lux na ilaw, at hanggang 20 percent overlap. Pinakamaraming error—123 cases—ay galing sa mabigat na central occlusion na 35 percent pataas, kung saan natatakpan ang brand label. Sunod ang mabilis na handheld motion blur sa kernel size 11 na may 70 cases, at steep angles na may 52. Yung natitirang errors ay look-alike containers—gaya ng dark vinegar bottle na napagkamalang canned meat sa TC08—at domain gap sa synthetic demo video. Ipe-present na ni Aron ang live demo plan at roadmap namin.\""
        ),
        (
            11, "Live Demo Plan & Roadmap to 100%", "Ogayon", "0:45",
            "Point left to Live Demo Sequence, then right to R1–R5 Roadmap.",
            "\"Para sa live demo po natin ngayon, ila-launch namin ang FastAPI server, ipapakita ang aming 16 pre-demo test cases—kasama ang 9 hard cases—magsu-switch kami ng detector live mula YOLOv8n papuntang ORB o SIFT, ipapakita ang Android CameraX WebSocket overlay, at ite-test ang kahit anong unseen image mula sa panel hanggang sa recipe generation. Para naman sa final 100 percent release, nakatutok ang limang natitirang tasks sa mismong failure modes na binanggit ni Allain: pag-record ng tatlong totoong kitchen MP4 pans, Stage-2 fine-tuning na may 60 look-alike negatives, motion-blur at 20-degree angle augmentation, Android release APK polish, at final report bago i-freeze ang cp5-final tag.\""
        ),
        (
            12, "Thank You & Q&A", "Ogayon", "0:15",
            "Keep Slide 12 visible while transitioning to the live FastAPI browser window.",
            "\"Maraming salamat po sa pakikinig. Ready na po kaming mag-proceed sa live demonstration at sumagot sa mga tanong ng panel.\""
        ),
    ]

    for s_no, s_title, spk, dur, cue, txt in slides_script:
        add_slide_script_block(doc, s_no, s_title, spk, dur, cue, txt)

    add_section_heading(doc, "3. Live Demo Execution Cue Sheet (Ogayon & Sarmiento)")
    demo_steps = [
        ("Step 1 — Start Server:", "Run .\\run_server.ps1 (or python src/server/app.py) and open http://127.0.0.1:8000/."),
        ("Step 2 — Standard & Hard Image Tests:", "Upload a normal shelf image (e.g., TC01/TC09) to show fast ~42 ms detection, then upload a shadowed or occluded image to show gated CLAHE recovery."),
        ("Step 3 — Live Detector Comparison:", "Toggle detector mode from yolov8 to orb_baseline or sift_baseline on the same shelf image to visually demonstrate classical baseline fragility."),
        ("Step 4 — Video Pan & Android CameraX:", "Run demo_pantry_pan.mp4 (or live webcam / Android USB-reversed CameraX feed) showing temporal voting (K >= 3/5) and real-time FPS HUD."),
        ("Step 5 — Unseen Image & Recipes:", "Accept an unseen grocery photo from the professor, verify detected counts, and click Generate Recipes to show 3 zero-waste meal suggestions."),
        ("Fallback Plan:", "If Wi-Fi/Gemini API is unavailable during demo, YOLOv8n runs 100% locally on CPU and src/llm_recipe.py falls back gracefully to deterministic local recipe templates."),
    ]
    for prefix, desc in demo_steps:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.18
        r1 = p.add_run(prefix + " ")
        r1.font.name = FONT_MAIN
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = RGBColor(13, 148, 136)
        r2 = p.add_run(desc)
        r2.font.name = FONT_MAIN
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = RGBColor(24, 24, 27)

    add_section_heading(doc, "4. Anticipated Panel Q&A Defense Bank")
    qa_items = [
        (
            "Q1: Bakit halos 0.00 ang score ng ORB at SIFT sa test set niyo?",
            "Naka-rely po ang ORB at SIFT sa 2D planar homography laban sa flat reference crops. Sa totoong pantry shelf, nawa-warp ang descriptors sa kurbadong lata, nawawala ang keypoint contrast sa shadow, at kulang ang inliers sa RANSAC kapag may occlusion. Ang fine-tuned YOLOv8n ay natututo ng multi-scale shape, color, at texture features kaya naka-0.819 mAP@0.50 siya sa parehong test split."
        ),
        (
            "Q2: Paano niyo mapapatunayan na nakatulong talaga ang luminance-gated CLAHE?",
            "Nag-run po kami ng controlled ablation sa eksaktong 104-image sub-100-lux split. Kapag walang CLAHE, bumabagsak ang YOLOv8n sa 0.301 mAP@0.50 at 0.368 recall. Kapag naka-on ang luminance-gated Gamma + CLAHE (nagti-trigger lang kapag mean LAB L < 45.0), tumatalon ang mAP@0.50 sa 0.675 (+0.374) habang hindi nagagalaw ang normal-light score na 0.819."
        ),
        (
            "Q3: Bakit may hiwalay na meat-safety confidence gate (>= 0.76)?",
            "High-severity error po kung mag-suggest ng meat o fish recipe gayong vegetarian items lang ang nasa pantry ng user. Dahil sa 0.76 threshold sa canned_tuna_meat, bumaba ang false meat detections sa 1 out of 115 non-meat pantry negatives (0.87% false-meat rate)."
        ),
        (
            "Q4: Bakit 7 out of 9 items lang ang na-confirm sa video clip?",
            "Synthetic cartoon-style shelf graphics po kasi ang ginamit sa demo_pantry_pan.mp4, kaya may domain gap laban sa real-photo weights namin. Sa Task R1, papalitan namin ito ng tatlong totoong smartphone pantry at fridge video pans."
        ),
        (
            "Q5: Bakit bumaba sa 0.392 at 0.533 ang mAP sa 35% occlusion at motion blur (k=11)?",
            "Limitado pa po ang heavy central cutout at walang directional motion-blur kernels sa current training set. Target mismo ng Tasks R2 at R3 ang mas malakas na Cutout/Mixup, 60 look-alike negatives, at synthetic blur (k=7..13)."
        ),
        (
            "Q6: Paano niyo na-achieve ang 23.7 FPS kahit walang GPU?",
            "6.2 MB lang po ang YOLOv8-nano (12.19 MB sa ONNX), at naka-cache lang ang isang loaded model instance sa MODEL_REGISTRY imbes na mag-reload kada request. Nasa 42.28 ms/frame lang ang average inference sa standard laptop CPU."
        ),
        (
            "Q7: Bakit kailangan ng 5-frame temporal voter (K >= 3) sa video?",
            "Sa single-frame video inference, nagkakaroon ng one-frame false positives at count flickering habang gumagalaw ang camera. Kapag kailangan ng at least 3 hits sa loob ng 5-frame sliding window, nagiging stable ang bilang bago ipasa sa recipe generator."
        ),
        (
            "Q8: Paano kung offline o walang internet ang Gemini LLM API habang demo?",
            "100% local po ang computer vision detection gamit ang YOLOv8n. Kung walang internet o ma-rate-limit ang Gemini API, automatic na nagfa-fallback ang src/llm_recipe.py sa local deterministic recipe templates base sa confirmed inventory."
        ),
    ]
    for q, a in qa_items:
        pq = doc.add_paragraph()
        pq.paragraph_format.space_before = Pt(6)
        pq.paragraph_format.space_after = Pt(2)
        pq.paragraph_format.keep_with_next = True
        rq = pq.add_run(q)
        rq.font.name = FONT_MAIN
        rq.font.size = Pt(9.8)
        rq.font.bold = True
        rq.font.color.rgb = RGBColor(17, 24, 39)

        pa = doc.add_paragraph()
        pa.paragraph_format.space_before = Pt(0)
        pa.paragraph_format.space_after = Pt(4)
        pa.paragraph_format.line_spacing = 1.18
        ra = pa.add_run(a)
        ra.font.name = FONT_MAIN
        ra.font.size = Pt(9.3)
        ra.font.color.rgb = RGBColor(51, 65, 85)

    doc.save(docx_path)

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    wdoc = word.Documents.Open(os.path.abspath(docx_path))
    wdoc.SaveAs(os.path.abspath(pdf_path), FileFormat=17)
    wdoc.Close(False)
    word.Quit()
    return docx_path, pdf_path


def main():
    os.makedirs(DEFENSE_DIR, exist_ok=True)
    summary = load_summary()
    pptx_path, slides_pdf = build_slides(summary)
    script_docx, script_pdf = build_script_pdf()
    print(json.dumps({
        "slides_pptx": pptx_path,
        "slides_pdf": slides_pdf,
        "script_docx": script_docx,
        "script_pdf": script_pdf,
    }, indent=2))


if __name__ == "__main__":
    main()
