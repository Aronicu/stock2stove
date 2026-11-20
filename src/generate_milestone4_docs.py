import csv
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm_recipe import generate_structured_recipes
from src.run import run_single_frame
from src.server.config import SAMPLES_DIR


def set_cell_background(cell, fill_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)


def set_cell_margins(cell, top=32, bottom=32, left=65, right=65):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)


def set_table_borders(table, border_color="CBD5E1"):
    tbl_pr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tbl_pr.append(borders)


def remove_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="none"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tbl_pr.append(borders)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))


def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.size = Pt(11.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(24, 68, 114)
    return p


def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 58, 95)
    return p


def add_body(doc, text, bold_prefix=None, size_pt=8.5, space_after=3.0):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.12
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.bold = True
        rb.font.size = Pt(size_pt)
        rb.font.color.rgb = RGBColor(25, 35, 48)
    r = p.add_run(text)
    r.font.size = Pt(size_pt)
    r.font.color.rgb = RGBColor(40, 44, 52)
    return p


def add_styled_table(doc, headers, rows, col_widths_in, font_size=7.4, cell_pad_v=24):
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)

    hdr_row = table.rows[0]
    prevent_row_split(hdr_row)
    for c_idx, h_text in enumerate(headers):
        cell = hdr_row.cells[c_idx]
        cell.width = Inches(col_widths_in[c_idx])
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_background(cell, "1E3A5F")
        set_cell_margins(cell, top=cell_pad_v + 6, bottom=cell_pad_v + 6, left=55, right=55)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(font_size)
        r.font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row_vals in enumerate(rows, start=1):
        row = table.rows[r_idx]
        prevent_row_split(row)
        bg = "FFFFFF" if r_idx % 2 == 1 else "F4F7FA"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = Inches(col_widths_in[c_idx])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=cell_pad_v, bottom=cell_pad_v, left=55, right=55)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.04
            r = p.add_run(str(val))
            r.font.size = Pt(font_size)
            if c_idx == 0:
                r.font.bold = True
            r.font.color.rgb = RGBColor(30, 35, 42)

    return table


def format_split_label(split_key):
    mapping = {
        "normal": "Normal",
        "shadowed_sub100lux": "Shadow (<100 lux)",
        "motion_blur": "Motion Blur (k=11)",
        "angled_perspective": "Angled (±16°)",
        "occluded_20_50pct": "Occluded (35%)",
        "cluttered_shelf": "Cluttered (>=4 items)",
    }
    return mapping.get(split_key, split_key)


def format_det_label(det_key):
    mapping = {
        "orb_baseline": "ORB Baseline",
        "sift_baseline": "SIFT Baseline",
        "yolov8n_pipeline": "YOLOv8n (Main)",
        "yolov8n_no_clahe": "YOLOv8n (--no-clahe)",
    }
    return mapping.get(det_key, det_key)


def load_milestone4_artifacts(milestone_dir):
    summary_path = os.path.join(milestone_dir, "milestone4_evaluation_summary.json")
    prelim_path = os.path.join(milestone_dir, "preliminary_results.csv")
    cp3_comp_path = os.path.join(milestone_dir, "cp3_vs_cp4_comparison.csv")
    test_cases_path = os.path.join(milestone_dir, "test_cases.csv")
    cls_report_path = os.path.join(milestone_dir, "classification_report.txt")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    prelim_rows = []
    with open(prelim_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            prelim_rows.append(row)

    cp3_comp_rows = []
    with open(cp3_comp_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cp3_comp_rows.append(row)

    test_cases_rows = []
    with open(test_cases_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            test_cases_rows.append(row)

    with open(cls_report_path, "r", encoding="utf-8") as f:
        cls_report_txt = f.read()

    return summary, prelim_rows, cp3_comp_rows, test_cases_rows, cls_report_txt


def render_workflow_diagram(output_path):
    fig, ax = plt.subplots(figsize=(13.0, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 32)
    ax.axis("off")

    boxes = [
        (2, 9, 16, 15, "#1E3A5F", "1. Input Capture", "Image / MP4 Video /\nWebcam / Android\nCameraX (640x640)"),
        (21, 9, 18, 15, "#0F766E", "2. preprocess()", "Luminance-Gated\nGamma+CLAHE (L<45)\n+ 640x640 Letterbox"),
        (42, 9, 19, 15, "#1D4ED8", "3. infer() Engine", "YOLOv8n (Main 6.2MB)\nHybrid Gemini Vision\nORB / SIFT Baselines"),
        (64, 9, 17, 15, "#7C3AED", "4. postprocess()", "Class-Wise NMS (0.45)\nClass Conf Gate (0.76)\n5-Frame Voter (K>=3)"),
        (84, 9, 14, 15, "#B45309", "5. App & LLM", "Overlay Boxes + Counts\nLatency/FPS HUD +\n3 Zero-Waste Recipes"),
    ]

    for x, y, w, h, color, title, body in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=color,
            edgecolor="#0F172A",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2.0, y + h - 3.0, title, color="white", fontsize=9.5, fontweight="bold", ha="center", va="center")
        ax.text(x + w / 2.0, y + (h - 4.0) / 2.0, body, color="#F8FAFC", fontsize=7.8, ha="center", va="center", linespacing=1.35)

    arrows = [(18.6, 20.4), (39.6, 41.4), (61.6, 63.4), (81.6, 83.4)]
    for x1, x2 in arrows:
        ax.annotate("", xy=(x2, 16.5), xytext=(x1, 16.5), arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", lw=2.0, color="#334155"))

    ax.text(50, 29.0, "Stock-to-Stove End-to-End Computer Vision & Application Workflow (cp4-80)", fontsize=12, fontweight="bold", color="#0F172A", ha="center")
    ax.text(50, 2.5, "Single Cached Model Instance in FastAPI / CLI Pipeline -> Web Dashboard (GET /) + Android CameraX + Offline/Cloud Gemini Recipe Engine", fontsize=8.5, color="#475569", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_application_interface_overview(output_path):
    sample_path = os.path.join(SAMPLES_DIR, "pantry_images_cv_01.jpg")
    test_img_dir = os.path.join(PROJECT_ROOT, "data", "dataset_yolo", "test", "images")
    multi_img_path = sample_path
    if os.path.isdir(test_img_dir):
        for fn in sorted(os.listdir(test_img_dir)):
            if "fridge_0084" in fn or "fridge_0081" in fn:
                multi_img_path = os.path.join(test_img_dir, fn)
                break

    bgr_main = cv2.imread(multi_img_path)
    if bgr_main is None:
        bgr_main = cv2.imread(sample_path)
    bgr_sample = cv2.imread(sample_path)

    run_single_frame(bgr_main, mode="yolov8", enable_clahe=True, conf_threshold=0.45, source_name="warmup")
    yolo_res = run_single_frame(bgr_main, mode="yolov8", enable_clahe=True, conf_threshold=0.45, source_name=os.path.basename(multi_img_path))
    orb_res = run_single_frame(bgr_sample, mode="orb_baseline", enable_clahe=True, conf_threshold=0.45, source_name="pantry_images_cv_01.jpg")

    yolo_rgb = cv2.cvtColor(yolo_res["rendered"]["annotated_frame"], cv2.COLOR_BGR2RGB)
    orb_rgb = cv2.cvtColor(orb_res["rendered"]["annotated_frame"], cv2.COLOR_BGR2RGB)

    counts = yolo_res["postprocessed"]["counts_by_class"]
    items_for_recipes = [{"class_id": k, "count": int(v)} for k, v in sorted(counts.items())]
    if not items_for_recipes:
        items_for_recipes = [{"class_id": "pasta_noodles", "count": 1}, {"class_id": "canned_tomato", "count": 1}]
    recipes_payload = generate_structured_recipes(items_for_recipes, api_key=None, timeout_sec=2.0)

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.95], hspace=0.25, wspace=0.14)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(yolo_rgb)
    ax1.set_title(
        f"A. Web UI / Android Overlay: YOLOv8n ({yolo_res['elapsed_ms']:.1f} ms | {1000.0 / max(0.1, yolo_res['elapsed_ms']):.1f} FPS)",
        fontsize=10,
        fontweight="bold",
        color="#1E3A5F",
    )
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(orb_rgb)
    ax2.set_title(
        f"B. Baseline Switcher: ORB + RANSAC Quads ({orb_res['elapsed_ms']:.1f} ms)",
        fontsize=10,
        fontweight="bold",
        color="#1E3A5F",
    )
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.add_patch(patches.Rectangle((0.1, 0.2), 9.8, 9.5, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2))
    ax3.text(0.4, 8.9, "C. Live Detection Table & Confirmed Inventory Counts", fontsize=10, fontweight="bold", color="#1E3A5F")
    y_pos = 7.9
    ax3.text(0.4, y_pos, f"Total Count: {yolo_res['postprocessed']['total_count']} items  |  Counts: {counts}", fontsize=8.5, fontweight="bold", color="#0F766E")
    y_pos -= 0.9
    labels = yolo_res["rendered"]["labels"]
    for idx, lbl in enumerate(labels[:6]):
        box_s = ", ".join([f"{v:.2f}" for v in lbl["bbox_norm"]])
        ax3.text(
            0.4,
            y_pos,
            f"{idx + 1}. {lbl['display_name']} ({lbl['class_id']})  |  conf={lbl['confidence']:.2f}  |  box=[{box_s}]",
            fontsize=8.0,
            color="#1E293B",
            family="monospace",
        )
        y_pos -= 0.95

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.add_patch(patches.Rectangle((0.1, 0.2), 9.8, 9.5, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2))
    ax4.text(0.4, 8.9, "D. Downstream Zero-Waste Recipe Output (LLM + Cache)", fontsize=10, fontweight="bold", color="#1E3A5F")
    y_r = 7.8
    for r_idx, rcard in enumerate(recipes_payload.get("recipes", [])[:3], start=1):
        ax4.text(0.4, y_r, f"Recipe {r_idx}: {rcard['title']} ({rcard['prep_time_mins']} mins)", fontsize=8.8, fontweight="bold", color="#1D4ED8")
        y_r -= 0.75
        step_txt = rcard["steps"][0] if rcard.get("steps") else ""
        if len(step_txt) > 74:
            step_txt = step_txt[:71] + "..."
        ax4.text(0.6, y_r, f"Step 1: {step_txt}", fontsize=7.8, color="#334155")
        y_r -= 0.70
        tip_txt = rcard.get("zero_waste_tip", "")
        if len(tip_txt) > 74:
            tip_txt = tip_txt[:71] + "..."
        ax4.text(0.6, y_r, f"Tip: {tip_txt}", fontsize=7.6, color="#0F766E")
        y_r -= 1.1

    fig.suptitle("Stock-to-Stove Integrated Application Showcase (Web Dashboard, Android CameraX & CLI)", fontsize=12.5, fontweight="bold", color="#0F172A")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_remaining_md(milestone_dir):
    content = """# Remaining Tasks to 100% (Milestone 5 Roadmap)

| ID | Task | Details & Verification Target | Owner | Target Date | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | Real Kitchen Video Test Suite | Capture and annotate 3 real smartphone pantry/fridge MP4 pans (daylight, dim cupboard, cluttered shelf) to replace the synthetic `demo_pantry_pan.mp4` cartoon clip and raise video tracking count accuracy within `+/-1` item. | Ogayon | 2026-11-24 | Planned |
| **R2** | Occlusion & Look-Alike Hardening | Run Stage-2 fine-tuning (`50 epochs`, `close_mosaic=10`, stronger Cutout/Mixup) plus 60 domestic hard-negative crops for `cereal_grains`, `pasta_noodles`, and `coffee_condiment` to push `occluded_20_50pct` mAP@0.50 `>= 0.55` and normal mAP@0.50 `>= 0.85`. | Sarmiento | 2026-11-26 | Planned |
| **R3** | Motion-Blur & Extreme-Angle Augmentation | Add directional motion blur (`k=7..13`) and `+/-20 deg` perspective warps during training and evaluate `models/best.pt` via `src/evaluate.py` to lift `motion_blur` mAP@0.50 `>= 0.65`. | Sarmiento | 2026-11-27 | Planned |
| **R4** | Android + Web UI Polish & E2E Stress Test | Finalize APK release build, verify WebSocket reconnect recovery, verify Web UI video upload/webcam live stream FPS display, and run `tests/run_e2e_tests.ps1`. | Ogayon | 2026-11-29 | Planned |
| **R5** | Final Milestone 5 Report & Defense Deck | Expand Draft Technical Report into final Results, Discussion, Conclusion, and polished 12-slide final defense presentation; freeze `cp5-final` tag. | Kumar & All | 2026-12-03 | Planned |
"""
    root_rem = os.path.join(PROJECT_ROOT, "docs", "remaining.md")
    m4_rem = os.path.join(milestone_dir, "remaining.md")
    with open(root_rem, "w", encoding="utf-8") as f:
        f.write(content)
    with open(m4_rem, "w", encoding="utf-8") as f:
        f.write(content)
    return root_rem, m4_rem


def build_milestone4_docx(milestone_dir, workflow_png, app_ui_png):
    summary, prelim_rows, cp3_comp_rows, test_cases_rows, _ = load_milestone4_artifacts(milestone_dir)

    docx_path = os.path.join(milestone_dir, "CSST106_Milestone4_80PercentCheck_Stock2Stove.docx")
    pdf_path = os.path.join(milestone_dir, "CSST106_Milestone4_80PercentCheck_Stock2Stove.pdf")

    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(0.50)
        section.bottom_margin = Inches(0.50)
        section.left_margin = Inches(0.58)
        section.right_margin = Inches(0.58)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(8.8)
    normal_style.font.color.rgb = RGBColor(40, 44, 52)

    p_hdr = doc.add_paragraph()
    p_hdr.paragraph_format.space_before = Pt(0)
    p_hdr.paragraph_format.space_after = Pt(2)
    r_hdr = p_hdr.add_run("CSST 106 — Computer Vision Final Project  |  Milestone 4 — 80% Progress Check & Report Draft")
    r_hdr.font.size = Pt(8.8)
    r_hdr.font.bold = True
    r_hdr.font.color.rgb = RGBColor(90, 100, 115)

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_t = p_title.add_run("Stock-to-Stove: Integrated Pantry Inventory Vision Application & Evaluation")
    r_t.font.size = Pt(15.5)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(24, 68, 114)

    hw = summary["hardware"]
    w_info = summary["weights"]
    meta_tbl = doc.add_table(rows=2, cols=3)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_tbl.autofit = False
    set_table_borders(meta_tbl, "D6E0EB")
    meta_widths = [2.45, 2.50, 2.39]
    meta_entries = [
        [
            ("Team Kaggleing: ", "Ogayon, Kumar, Sarmiento"),
            ("Git Tag: ", "cp4-80  (repo: stock2stove)"),
            ("Weights: ", f"{w_info['weights_path']} ({w_info['model_size_mb']} MB PT / {w_info['onnx_size_mb']} MB ONNX)"),
        ],
        [
            ("Laptop CPU: ", f"AMD Ryzen 7 5700G ({hw['cpu_threads']}T, {hw['os']})"),
            ("Eval Config: ", f"seed={summary['seed']}, conf={summary['conf_threshold']}, NMS={summary['nms_iou_threshold']}"),
            ("Held-Out Set: ", f"{summary['num_test_images']} test imgs ({summary['num_gt_boxes']} GT) + 115 negs + 1 clip"),
        ],
    ]
    for r_idx, row_data in enumerate(meta_entries):
        prevent_row_split(meta_tbl.rows[r_idx])
        for c_idx, (lbl, val) in enumerate(row_data):
            cell = meta_tbl.rows[r_idx].cells[c_idx]
            cell.width = Inches(meta_widths[c_idx])
            set_cell_background(cell, "F1F5F9")
            set_cell_margins(cell, top=24, bottom=24, left=55, right=55)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            rl = p.add_run(lbl)
            rl.font.bold = True
            rl.font.size = Pt(7.6)
            rl.font.color.rgb = RGBColor(50, 60, 75)
            rv = p.add_run(val)
            rv.font.size = Pt(7.6)
            rv.font.color.rgb = RGBColor(24, 68, 114)

    add_heading_1(doc, "1. Introduction, Problem Statement & Objectives (Report Draft)")
    add_body(
        doc,
        "Households waste substantial groceries annually due to forgotten rear-shelf pantry items, duplicate purchases, and meal-planning decision fatigue. "
        "Manual inventory spreadsheets are tedious to maintain, while standard barcode scanners fail on loose produce, unlabeled bags, and partially occluded shelf stacks. "
        "Stock-to-Stove addresses this problem by converting a short smartphone or webcam scan of kitchen shelves and refrigerators into a verified ingredient inventory and three zero-waste cooking recipes.",
        bold_prefix="Problem Context: ",
    )
    add_body(
        doc,
        "(1) Detect and classify 8 unified pantry and refrigerator food categories under realistic lighting, camera tilt, and shelf clutter; "
        "(2) Quantify improvement from classical feature matching (ORB/SIFT + RANSAC) to fine-tuned YOLOv8n (models/best.pt, 6.20 MB); "
        "(3) Maintain interactive CPU inference speed (>=15 FPS on the AMD Ryzen 7 5700G demo laptop) with a single cached model instance; "
        "(4) Expose the full pipeline through an intuitive browser Web Dashboard (GET /), native Android CameraX client, and CLI (src/run.py), linked to downstream recipe generation.",
        bold_prefix="Project Objectives: ",
    )

    add_heading_1(doc, "2. Dataset & Preprocessing Pipeline (Report Draft)")
    add_body(
        doc,
        "The locked dataset (data/dataset_yolo/data.yaml) merges multi-source pantry and refrigerator imagery across 8 unified classes: "
        "pasta_noodles (0), canned_tomato (1), canned_tuna_meat (2), cereal_grains (3), coffee_condiment (4), milk_dairy (5), eggs (6), and fresh_produce (7). "
        "Total annotated imagery comprises 813 images (2,480 ground-truth boxes) plus 115 non-meat pantry_negatives safety scenes.",
    )
    ds_rows = [
        ("Train Split", "605", "1,937", "74 empty", "Freiburg (225), Kitchen-YOLO (146), Fridge (76), Composite (65), Negatives (74), Pantry (19)"),
        ("Validation Split", "104", "282", "13 empty", "Freiburg (57), Kitchen-YOLO (18), Fridge (12), Pantry (4), Negatives (13)"),
        ("Held-Out Test Split (Locked)", "104", "261", "13 empty", "Freiburg (57), Kitchen-YOLO (18), Fridge (12), Pantry (4), Negatives (13)"),
        ("Pantry Negatives (Safety)", "115", "0 (non-meat)", "115 negs", "False-meat safety benchmark verifying canned_tuna_meat precision on non-meat shelves"),
        ("Video Pan Clip", "60 frames", "9 shelf items", "Synthetic", "data/raw_home/demo_pantry_pan.mp4 (640x480 @ 15 FPS) for W=5, K=3 temporal tracking"),
    ]
    add_styled_table(
        doc,
        ["Dataset Partition", "Images", "GT Boxes", "Negatives", "Source Composition"],
        ds_rows,
        [1.55, 0.72, 0.90, 0.80, 3.37],
        font_size=7.3,
        cell_pad_v=18,
    )
    add_body(
        doc,
        "In Checkpoint 3, unconditional LAB CLAHE boosted dark <100 lux recall but caused contrast drift on normal images (dropping mAP@0.50 from 0.8190 to 0.7494). "
        "For Milestone 4 (src/cv/preprocess.py), apply_clahe_lab() was upgraded to a luminance-gated restoration stage: when mean LAB lightness L < 45.0, it applies inverse gamma restoration (img / 0.26)^(1/2.4) followed by L-channel CLAHE (clipLimit=1.8, 8x8 grid); otherwise (L >= 45.0) it preserves original pixel contrast before 640x640 aspect-preserving letterboxing.",
        bold_prefix="Luminance-Gated Preprocessing Fix: ",
    )

    add_heading_1(doc, "3. Methodology, Workflow Diagram & Integrated Application (Report Draft)")
    add_body(
        doc,
        "Every input image, video frame, WebSocket packet, and HTTP upload executes the unified procedural pipeline in src/run.py: "
        "preprocess() -> infer() -> postprocess() -> render(). Models (yolov8n_stock2stove.pt, ORB/SIFT templates, Gemini Vision client) are cached once in MODEL_REGISTRY rather than reloaded per click or frame. "
        "Post-processing enforces class-wise NMS (IoU=0.45), a class-specific confidence gate (canned_tuna_meat >= 0.76 to prevent false meat triggers), and a 5-frame sliding-window temporal voter (K >= 3 hits for video/WebSocket confirmation).",
    )
    if os.path.exists(workflow_png):
        p_wf = doc.add_paragraph()
        p_wf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_wf.paragraph_format.space_before = Pt(2)
        p_wf.paragraph_format.space_after = Pt(1)
        p_wf.add_run().add_picture(workflow_png, width=Inches(6.9))
        add_body(doc, "Figure 1: Stock-to-Stove 5-stage computer vision pipeline and application integration architecture.", size_pt=7.3, space_after=3)

    if os.path.exists(app_ui_png):
        p_ui = doc.add_paragraph()
        p_ui.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_ui.paragraph_format.space_before = Pt(2)
        p_ui.paragraph_format.space_after = Pt(1)
        p_ui.add_run().add_picture(app_ui_png, width=Inches(6.8))
        add_body(doc, "Figure 2: Integrated application interface outputs showing YOLOv8n boxes, ORB+RANSAC baseline quads, detection counts, latency HUD, and zero-waste recipes.", size_pt=7.3, space_after=3)

    add_heading_1(doc, "4. Reproducible Experiments & Checkpoint 3 -> Checkpoint 4 Progress")
    add_body(
        doc,
        f"Command: python src/evaluate.py --split test --weights {w_info['weights_path']} (seed=42, conf=0.45, NMS IoU=0.45, eval IoU=0.50). "
        f"Model footprint: {w_info['model_size_mb']} MB PyTorch (.pt) / {w_info['onnx_size_mb']} MB ONNX.",
        size_pt=8.2,
    )
    res_table_rows = []
    for row in prelim_rows:
        res_table_rows.append((
            format_split_label(row["split"]),
            format_det_label(row["detector"]),
            f"{float(row['precision']):.4f}",
            f"{float(row['recall']):.4f}",
            f"{float(row['f1_score']):.4f}",
            f"{float(row['mAP50']):.4f}",
            f"{float(row['mAP50_95']):.4f}",
            f"{float(row['mean_inference_ms']):.1f} ms ({float(row['fps']):.1f})",
        ))
    add_styled_table(
        doc,
        ["Condition Split", "Method", "Prec", "Rec", "F1", "mAP@.5", "mAP@.5:.95", "Laptop Speed (FPS)"],
        res_table_rows,
        [1.30, 1.45, 0.58, 0.58, 0.58, 0.65, 0.76, 1.44],
        font_size=7.0,
        cell_pad_v=14,
    )

    add_heading_2(doc, "4.1 Checkpoint 3 (cp3-core) vs. Checkpoint 4 (cp4-80) Delta Analysis")
    cp_rows = []
    for r in cp3_comp_rows:
        cp_rows.append((
            r["condition"],
            f"{float(r['cp3_mAP50']):.4f}",
            f"{float(r['cp4_mAP50']):.4f}",
            f"{float(r['delta_mAP50']):+.4f}",
            f"{float(r['cp3_f1']):.4f}",
            f"{float(r['cp4_f1']):.4f}",
            f"{float(r['delta_f1']):+.4f}",
            f"{float(r['cp4_fps']):.1f} FPS",
        ))
    add_styled_table(
        doc,
        ["Condition", "CP3 mAP@.5", "CP4 mAP@.5", "Delta mAP", "CP3 F1", "CP4 F1", "Delta F1", "CP4 Speed"],
        cp_rows,
        [1.42, 0.78, 0.78, 0.78, 0.72, 0.72, 0.78, 0.86],
        font_size=7.1,
        cell_pad_v=14,
    )
    add_body(
        doc,
        "Why Numbers Changed Since Checkpoint 3: Replacing unconditional LAB CLAHE with luminance-gated Gamma + CLAHE (activated only when mean L < 45.0) eliminated contrast distortion on bright/normal shelves (+0.0696 mAP@0.50 on Normal, +0.1049 on Angled, +0.0972 on Motion Blur) while rescuing extreme <100 lux shadowed scenes (+0.2323 mAP@0.50, +0.2452 Recall). Meanwhile, --no-clahe achieves only 0.3010 mAP@0.50 under <100 lux shadow vs. 0.6750 with the gated pipeline.",
        size_pt=8.0,
        space_after=2.5,
    )

    cp3_chart = os.path.join(milestone_dir, "cp3_vs_cp4_comparison.png")
    if os.path.exists(cp3_chart):
        p_cp = doc.add_paragraph()
        p_cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cp.paragraph_format.space_before = Pt(2)
        p_cp.paragraph_format.space_after = Pt(1)
        p_cp.add_run().add_picture(cp3_chart, width=Inches(6.2))
        add_body(doc, "Figure 3: Checkpoint 3 vs. Checkpoint 4 mAP@0.50 and F1-Score comparison across all 6 test conditions.", size_pt=7.3, space_after=3)

    add_heading_2(doc, "4.2 Per-Class Held-Out Normal Test Performance (YOLOv8n)")
    pc_rows = []
    for crow in summary["per_class_normal_yolov8n"]:
        pc_rows.append((
            crow["class_id"],
            f"{crow['ap50']:.4f}",
            f"{crow['precision']:.4f}",
            f"{crow['recall']:.4f}",
        ))
    add_styled_table(
        doc,
        ["Unified Class ID", "AP@0.50", "Precision", "Recall"],
        pc_rows,
        [2.50, 1.60, 1.60, 1.64],
        font_size=7.3,
        cell_pad_v=14,
    )

    add_heading_1(doc, "5. Performance Analysis, Operating Conditions & Limitations")
    fa = summary["failure_analysis"]
    cc = fa["cause_counts"]
    vid = summary["video_evaluation"]
    fm = summary["false_meat_evaluation"]

    add_body(
        doc,
        "The system is reliable on well-lit and moderately cluttered domestic shelves (Normal mAP@0.50 = 0.8190, Cluttered >=4 items mAP@0.50 = 0.8665, eggs AP = 1.000, canned_tomato AP = 0.9478, canned_tuna_meat AP = 0.9218, milk_dairy AP = 0.8987, fresh_produce AP = 0.8752) at 17–21 FPS on CPU, with only "
        f"{fm['false_meat_count']}/{fm['total_pantry_negatives']} ({fm['false_meat_rate']:.4f}) false-meat triggers.",
        bold_prefix="Where It Works: ",
    )
    add_body(
        doc,
        "Designed for handheld smartphone/webcam scans at 0.3–1.2 m distance, >=30 lux ambient or cabinet lighting (assisted by luminance-gated CLAHE), frontal to +/-15 deg shelf angles, and <=20% front-item overlap on real grocery packaging. "
        f"NOT designed for: (a) synthetic/cartoon graphics (demo_pantry_pan.mp4 achieves {vid['predicted_confirmed_items']}/{vid['expected_unique_items']} confirmed items with {vid['frame_overcounts']} per-frame overcounts due to synthetic domain gap), (b) severe >=35% central box occlusion, (c) rapid pan motion blur (k>=11), or (d) opaque pantry containers lacking visible labels.",
        bold_prefix="Operating Conditions & Out-of-Scope Regimes: ",
    )

    by_cause = {}
    for gcase in fa.get("gallery_cases", []):
        by_cause.setdefault(gcase["cause"], []).append(gcase)

    def fmt_examples(cause_name, extra=""):
        items = by_cause.get(cause_name, [])
        if not items:
            return extra or "Held-out test split"
        parts = []
        for it in items[:3]:
            short_fn = it["filename"]
            if len(short_fn) > 24:
                short_fn = short_fn[:18] + "..."
            parts.append(f"{it['id']} ({short_fn})")
        joined = ", ".join(parts)
        return f"{joined}; {extra}" if extra else joined

    lim_rows = [
        ("Occlusion & Clutter", str(cc.get("Occlusion and clutter", 0)), fmt_examples("Occlusion and clutter"), "Front items covering >=35% of rear packaging hide brand cues; NMS suppresses tightly overlapping stacked cartons."),
        ("Image Quality / Motion Blur", str(cc.get("Image quality (blur/low-res)", 0)), fmt_examples("Image quality (blur/low-res)"), "Horizontal camera pan smears vertical label edges and barcodes, dropping Recall from 0.8429 to 0.6015."),
        ("Viewpoint & Scale", str(cc.get("Viewpoint and scale", 0)), fmt_examples("Viewpoint and scale"), "Perspective warp (+/-16 deg) shrinks axis-aligned IoU overlap and distorts tall narrow bags (coffee_condiment AP = 0.6390)."),
        ("Label / Domain Gap", str(cc.get("Label errors / domain gap", 0)), fmt_examples("Label errors / domain gap", f"demo_pantry_pan.mp4 ({vid['predicted_confirmed_items']}/{vid['expected_unique_items']})"), "Synthetic rendered shelf clip and rare label discrepancies differ from natural real-world training photos."),
        ("Lighting & Shadows", str(cc.get("Lighting (shadows/glare)", 0)), fmt_examples("Lighting (shadows/glare)"), "Extreme <100 lux cast shadows lose chromatic contrast; luminance-gated Gamma+CLAHE cuts shadow FNs from 123 to 59 but unlit dark bags remain vulnerable."),
        ("Look-Alike Packaging", str(cc.get("Look-alike classes", 0)), fmt_examples("Look-alike classes", "TC08 vinegar"), "Rectangular cartons and glass bottles cause confusion among cereal_grains (0.6241), pasta_noodles (0.6457), coffee_condiment (0.6390), and vinegar."),
    ]
    add_styled_table(
        doc,
        ["Limitation Category", "Events", "Example Image / Clip", "Root Cause & Failure Mechanism"],
        lim_rows,
        [1.35, 0.55, 2.05, 3.39],
        font_size=7.0,
        cell_pad_v=16,
    )

    side_tbl = doc.add_table(rows=1, cols=2)
    side_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    side_tbl.autofit = False
    remove_table_borders(side_tbl)
    prevent_row_split(side_tbl.rows[0])
    cm_png = os.path.join(milestone_dir, "test_confusion_matrix.png")
    gal_png = os.path.join(milestone_dir, "failure_gallery.png")
    c0 = side_tbl.rows[0].cells[0]
    c1 = side_tbl.rows[0].cells[1]
    c0.width = Inches(3.45)
    c1.width = Inches(3.89)
    set_cell_margins(c0, top=10, bottom=10, left=10, right=40)
    set_cell_margins(c1, top=10, bottom=10, left=20, right=10)
    if os.path.exists(cm_png):
        p0 = c0.paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p0.add_run().add_picture(cm_png, width=Inches(3.25))
    if os.path.exists(gal_png):
        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p1.add_run().add_picture(gal_png, width=Inches(3.75))
    add_body(doc, "Figure 4: Held-out test confusion matrix (left) and 12 annotated failure gallery cases (right; Green=GT, Red=Prediction).", size_pt=7.3, space_after=3)

    tc_summary = summary["test_cases_summary"]
    add_heading_1(doc, f"6. Pre-Demo Test Cases (TC01–TC{tc_summary['total_cases']:02d})")
    add_body(
        doc,
        f"Executed {tc_summary['total_cases']} pre-demo test cases ({tc_summary['hard_cases']} hard/edge cases): {tc_summary['passed']} PASS, {tc_summary['failed']} FAIL (demonstrating both working and known failure boundaries).",
        size_pt=8.1,
    )
    tc_rows = []
    for row in test_cases_rows:
        inp = row["Input"]
        if len(inp) > 26:
            inp = inp[:13] + "..." + inp[-10:]
        tc_rows.append((
            row["ID"],
            inp,
            row["Condition"][:28],
            row["CaseType"],
            row["Expected"][:32],
            row["Actual"][:34],
            row["Pass"],
        ))
    add_styled_table(
        doc,
        ["ID", "Input", "Condition", "Type", "Expected", "Actual", "Pass"],
        tc_rows,
        [0.42, 1.32, 1.25, 0.92, 1.45, 1.50, 0.48],
        font_size=6.6,
        cell_pad_v=14,
    )

    add_heading_1(doc, "7. To-100% Roadmap (docs/remaining.md)")
    rem_rows = [
        ("R1", "Real Kitchen Video Test Suite", "Replace synthetic demo_pantry_pan.mp4 with 3 real smartphone shelf/fridge MP4 pans", "Ogayon", "2026-11-24"),
        ("R2", "Occlusion & Look-Alike Hardening", "50-epoch Stage-2 fine-tune + 60 domestic crops (cereal/pasta/coffee)", "Sarmiento", "2026-11-26"),
        ("R3", "Motion-Blur & Angle Augmentation", "Add k=7..13 motion blur and +/-20 deg perspective warps to training", "Sarmiento", "2026-11-27"),
        ("R4", "Android & Web UI E2E Polish", "Release APK, WebSocket reconnect test, and tests/run_e2e_tests.ps1", "Ogayon", "2026-11-29"),
        ("R5", "Final Milestone 5 Report & Slides", "Complete Discussion/Conclusion, 12-slide defense deck, tag cp5-final", "Kumar & All", "2026-12-03"),
    ]
    add_styled_table(
        doc,
        ["ID", "Task", "Target Scope", "Owner", "Target Date"],
        rem_rows,
        [0.45, 1.65, 3.24, 1.05, 0.95],
        font_size=7.2,
        cell_pad_v=16,
    )

    doc.save(docx_path)

    page_count = None
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        wdoc = word.Documents.Open(os.path.abspath(docx_path))
        wdoc.Repaginate()
        page_count = int(wdoc.ComputeStatistics(2))
        wdoc.SaveAs(os.path.abspath(pdf_path), FileFormat=17)
        wdoc.Close()
        word.Quit()
    except Exception as exc:
        print(f"DOCX->PDF export warning: {exc}")

    return docx_path, pdf_path, page_count


def ole_rgb(r, g, b):
    return int(r) + (int(g) << 8) + (int(b) << 16)


def add_slide_header(slide, title_text, subtitle_text, slide_num, total_slides=10):
    bar = slide.Shapes.AddShape(1, 0, 0, 960, 68)
    bar.Fill.ForeColor.RGB = ole_rgb(24, 58, 95)
    bar.Line.Visible = 0

    accent = slide.Shapes.AddShape(1, 0, 68, 960, 4)
    accent.Fill.ForeColor.RGB = ole_rgb(2, 132, 199)
    accent.Line.Visible = 0

    tb = slide.Shapes.AddTextbox(1, 28, 10, 820, 52)
    tr = tb.TextFrame.TextRange
    tr.Text = f"{title_text}\n{subtitle_text}"
    tr.Paragraphs(1).Font.Size = 20
    tr.Paragraphs(1).Font.Bold = -1
    tr.Paragraphs(1).Font.Color.RGB = ole_rgb(255, 255, 255)
    tr.Paragraphs(2).Font.Size = 11
    tr.Paragraphs(2).Font.Color.RGB = ole_rgb(186, 230, 253)

    ftr = slide.Shapes.AddTextbox(1, 28, 514, 904, 20)
    ftr_tr = ftr.TextFrame.TextRange
    ftr_tr.Text = f"CSST 106 Stock-to-Stove | Team Kaggleing | Milestone 4 (cp4-80)                                                                               Slide {slide_num} / {total_slides}"
    ftr_tr.Font.Size = 9
    ftr_tr.Font.Color.RGB = ole_rgb(100, 116, 139)


def add_slide_bullets(slide, left, top, width, height, heading, bullets, font_size=13):
    box = slide.Shapes.AddShape(5, left, top, width, height)
    box.Fill.ForeColor.RGB = ole_rgb(248, 250, 252)
    box.Line.ForeColor.RGB = ole_rgb(203, 213, 225)
    box.Line.Weight = 1.0

    tb = slide.Shapes.AddTextbox(1, left + 12, top + 10, width - 24, height - 20)
    tf = tb.TextFrame
    tf.WordWrap = -1
    lines = [heading] + [f"• {b}" for b in bullets]
    tf.TextRange.Text = "\n".join(lines)
    tf.TextRange.Paragraphs(1).Font.Size = font_size + 2
    tf.TextRange.Paragraphs(1).Font.Bold = -1
    tf.TextRange.Paragraphs(1).Font.Color.RGB = ole_rgb(24, 58, 95)
    for idx in range(2, len(lines) + 1):
        p = tf.TextRange.Paragraphs(idx)
        p.Font.Size = font_size
        p.Font.Color.RGB = ole_rgb(30, 41, 59)
        p.ParagraphFormat.SpaceBefore = 5


def add_slide_image(slide, img_path, left, top, width):
    if not os.path.exists(img_path):
        return None
    im = cv2.imread(img_path)
    if im is None or im.shape[1] <= 0:
        return None
    height = float(width) * (float(im.shape[0]) / float(im.shape[1]))
    return slide.Shapes.AddPicture(os.path.abspath(img_path), 0, -1, left, top, float(width), height)


def add_slide_table(slide, left, top, width, height, headers, rows, font_size=10, row_height=22, col_widths=None):
    tbl_shape = slide.Shapes.AddTable(len(rows) + 1, len(headers), left, top, width, height)
    tbl = tbl_shape.Table
    if col_widths:
        for c_idx, cw in enumerate(col_widths):
            tbl.Columns(c_idx + 1).Width = cw

    tbl.Rows(1).Height = row_height
    for c_idx, h in enumerate(headers):
        cell = tbl.Cell(1, c_idx + 1)
        cell.Shape.Fill.Solid()
        cell.Shape.Fill.ForeColor.RGB = ole_rgb(30, 58, 95)
        tf = cell.Shape.TextFrame
        tf.MarginTop = 1
        tf.MarginBottom = 1
        tf.MarginLeft = 4
        tf.MarginRight = 4
        tr = tf.TextRange
        tr.Text = str(h)
        tr.Font.Size = font_size
        tr.Font.Bold = -1
        tr.Font.Color.RGB = ole_rgb(255, 255, 255)
    for r_idx, row_vals in enumerate(rows, start=2):
        tbl.Rows(r_idx).Height = row_height
        bg = ole_rgb(255, 255, 255) if r_idx % 2 == 0 else ole_rgb(241, 245, 249)
        for c_idx, val in enumerate(row_vals):
            cell = tbl.Cell(r_idx, c_idx + 1)
            cell.Shape.Fill.Solid()
            cell.Shape.Fill.ForeColor.RGB = bg
            tf = cell.Shape.TextFrame
            tf.MarginTop = 1
            tf.MarginBottom = 1
            tf.MarginLeft = 4
            tf.MarginRight = 4
            tr = tf.TextRange
            tr.Text = str(val)
            tr.Font.Size = font_size - 0.5 if len(rows) > 8 else font_size
            tr.Font.Color.RGB = ole_rgb(30, 41, 59)
            if c_idx == 0:
                tr.Font.Bold = -1
    return tbl_shape


def build_milestone4_slides(milestone_dir, workflow_png, app_ui_png):
    summary, prelim_rows, cp3_comp_rows, test_cases_rows, _ = load_milestone4_artifacts(milestone_dir)
    pptx_path = os.path.join(milestone_dir, "CSST106_Milestone4_Slides_Stock2Stove.pptx")
    pdf_path = os.path.join(milestone_dir, "CSST106_Milestone4_Slides_Stock2Stove.pdf")

    import win32com.client
    pp = win32com.client.Dispatch("PowerPoint.Application")
    pp.Visible = 1
    prs = pp.Presentations.Add()
    prs.PageSetup.SlideWidth = 960
    prs.PageSetup.SlideHeight = 540

    normal_yolo = {}
    for r in prelim_rows:
        if r["split"] == "normal" and r["detector"] == "yolov8n_pipeline":
            normal_yolo = r
            break

    w_info = summary["weights"]
    vid = summary["video_evaluation"]
    fm = summary["false_meat_evaluation"]
    tc_sum = summary["test_cases_summary"]

    s1 = prs.Slides.Add(1, 12)
    bg1 = s1.Shapes.AddShape(1, 0, 0, 960, 540)
    bg1.Fill.ForeColor.RGB = ole_rgb(15, 23, 42)
    bg1.Line.Visible = 0
    bar1 = s1.Shapes.AddShape(1, 0, 380, 960, 6)
    bar1.Fill.ForeColor.RGB = ole_rgb(56, 189, 248)
    bar1.Line.Visible = 0
    tb1 = s1.Shapes.AddTextbox(1, 60, 120, 840, 240)
    tr1 = tb1.TextFrame.TextRange
    tr1.Text = (
        "Stock-to-Stove\n"
        "Automated Pantry Inventory Detection & Zero-Waste Recipe Application\n"
        "CSST 106 Computer Vision Final Project  |  Milestone 4 (80% Progress Check)"
    )
    tr1.Paragraphs(1).Font.Size = 40
    tr1.Paragraphs(1).Font.Bold = -1
    tr1.Paragraphs(1).Font.Color.RGB = ole_rgb(56, 189, 248)
    tr1.Paragraphs(2).Font.Size = 20
    tr1.Paragraphs(2).Font.Color.RGB = ole_rgb(241, 245, 249)
    tr1.Paragraphs(3).Font.Size = 14
    tr1.Paragraphs(3).Font.Color.RGB = ole_rgb(148, 163, 184)

    tb1b = s1.Shapes.AddTextbox(1, 60, 410, 840, 90)
    tr1b = tb1b.TextFrame.TextRange
    tr1b.Text = (
        f"Team Kaggleing: Aron Gabriel L. Ogayon (Lead) | Allain A. Kumar (Docs) | Janvher Lucas J. Sarmiento (Dev)\n"
        f"Tag: cp4-80  |  Weights: {w_info['weights_path']} ({w_info['model_size_mb']} MB)  |  Normal mAP@0.50: {float(normal_yolo.get('mAP50', 0)):.4f} ({float(normal_yolo.get('fps', 0)):.1f} FPS CPU)"
    )
    tr1b.Paragraphs(1).Font.Size = 13
    tr1b.Paragraphs(1).Font.Color.RGB = ole_rgb(226, 232, 240)
    tr1b.Paragraphs(2).Font.Size = 12
    tr1b.Paragraphs(2).Font.Color.RGB = ole_rgb(56, 189, 248)

    s2 = prs.Slides.Add(2, 12)
    add_slide_header(s2, "1. Problem Statement & Project Objectives", "Why Pantry Inventory Needs Computer Vision Beyond Barcode Scanning", 2)
    add_slide_bullets(
        s2, 32, 92, 435, 395,
        "The Household Pantry & Kitchen Problem",
        [
            "Forgotten rear-shelf groceries expire, causing duplicate purchases and avoidable food waste.",
            "Decision fatigue when looking at mixed shelves leads to takeout instead of cooking.",
            "Manual barcode scanning fails on loose produce, unlabeled bags, and stacked shelves.",
            "Real domestic cupboards suffer from <100 lux shadows, camera tilt, and partial occlusion.",
        ],
        font_size=13,
    )
    add_slide_bullets(
        s2, 492, 92, 435, 395,
        "Milestone 4 Engineering Objectives",
        [
            "Detect & count 8 unified pantry + refrigerator categories in one integrated pipeline.",
            "Compare classical SIFT/ORB + RANSAC baselines against fine-tuned YOLOv8n.",
            "Run real-time CPU inference (>=15 FPS on AMD Ryzen 7 5700G) with cached weights.",
            "Provide browser Web UI (GET /), Android CameraX app, CLI, and zero-waste LLM recipes.",
        ],
        font_size=13,
    )

    s3 = prs.Slides.Add(3, 12)
    add_slide_header(s3, "2. Dataset & 8-Class Pantry/Fridge Schema", "Multi-Source Merged Dataset (813 Labeled Images + 115 Negatives + Video Pan)", 3)
    ds_slide_rows = [
        ("0: pasta_noodles", "Pantry", "205", "24", "26", "Spaghetti boxes, macaroni, instant noodles"),
        ("1: canned_tomato", "Pantry", "149", "27", "21", "Tomato sauce, paste, crushed tomato cans"),
        ("2: canned_tuna_meat", "Pantry", "162", "26", "16", "Tuna, corned beef, sardines (conf>=0.76)"),
        ("3: cereal_grains", "Pantry", "117", "25", "19", "Cereal boxes, oatmeal cartons, grain bags"),
        ("4: coffee_condiment", "Pantry", "168", "28", "32", "Coffee bags/jars, cooking oil, soy sauce"),
        ("5: milk_dairy", "Fridge", "283", "51", "40", "Milk cartons, cheese blocks, butter, yogurt"),
        ("6: eggs", "Fridge", "55", "6", "5", "Egg cartons, egg trays, loose fridge eggs"),
        ("7: fresh_produce", "Fridge", "798", "95", "102", "Tomatoes, onions, garlic, bell peppers, greens"),
    ]
    add_slide_table(
        s3, 32, 86, 896, 250,
        ["Class ID", "Zone", "Train", "Valid", "Test", "Representative Items"],
        ds_slide_rows,
        font_size=10.5,
        row_height=26,
        col_widths=[165, 80, 75, 75, 75, 426],
    )
    add_slide_bullets(
        s3, 32, 390, 896, 98,
        "Frozen Split & Safety Sets",
        [
            "Train: 605 imgs (1,937 boxes) | Valid: 104 imgs (282 boxes) | Held-Out Test: 104 imgs (261 boxes) | Pantry Negatives: 115 non-meat imgs | Video: 60 frames.",
        ],
        font_size=11,
    )

    s4 = prs.Slides.Add(4, 12)
    add_slide_header(s4, "3. Technical Approach: Classical Baselines vs. Fine-Tuned YOLOv8n", "Addressing Planar Feature Fragility with YOLOv8n & Luminance-Gated CLAHE", 4)
    add_slide_bullets(
        s4, 32, 92, 435, 395,
        "Classical Baselines (ORB & SIFT + RANSAC)",
        [
            "1,000 keypoints per frame matched against multi-angle reference templates.",
            "Lowe's ratio test (0.75) + RANSAC homography (min 6 inliers) outputs polygon quads.",
            "Fails on cylindrical cans, occluded logos, motion blur, and shadow gradients (mAP <= 0.036).",
            "Slow on CPU: 0.44-0.55 FPS (ORB) and 1.60-2.32 FPS (SIFT).",
        ],
        font_size=12.5,
    )
    add_slide_bullets(
        s4, 492, 92, 435, 395,
        "Main Method: YOLOv8n + Luminance-Gated CLAHE",
        [
            "Fine-tuned YOLOv8n (CSPDarknet + PAN-FPN, 6.20 MB PT / 12.19 MB ONNX).",
            "CP4 Upgrade: Luminance-gated Gamma + CLAHE activates only when LAB mean L < 45.0.",
            "Eliminates normal-light contrast drift while lifting <100 lux shadow recall +24.5 pts.",
            "Class-wise NMS (0.45), meat gate (>=0.76), and 5-frame temporal voter (K>=3/5).",
        ],
        font_size=12.5,
    )

    s5 = prs.Slides.Add(5, 12)
    add_slide_header(s5, "4. End-to-End Pipeline & Integrated Application Interface", "Unified src/run.py Engine Exposed via Browser Web Dashboard, Android CameraX & CLI", 5)
    add_slide_image(s5, workflow_png, 28, 86, 444)
    add_slide_bullets(
        s5, 28, 244, 444, 248,
        "Pipeline Architecture & Caching",
        [
            "preprocess(): Luminance-gated Gamma+CLAHE (L<45) + 640x640 letterbox.",
            "infer(): Single cached YOLOv8n / Gemini / ORB / SIFT instance in MODEL_REGISTRY.",
            "postprocess(): Class-wise NMS (0.45), meat gate (0.76), 5-frame voter (K>=3).",
            "render(): Boxes/quads, confidence, counts, ms/FPS HUD & JSON recipes.",
        ],
        font_size=11,
    )
    add_slide_image(s5, app_ui_png, 488, 86, 444)
    add_slide_bullets(
        s5, 488, 394, 444, 100,
        "Application Entrypoints",
        [
            "Browser Dashboard (GET /): Image, Video & Webcam detection + recipe cards.",
            "Android CameraX (WS /ws/scan) & Batch CLI (src/run.py).",
        ],
        font_size=10.5,
    )

    s6 = prs.Slides.Add(6, 12)
    add_slide_header(s6, "5. Reproducible Results on Held-Out Test Set (evaluate.py)", f"python src/evaluate.py --split test --weights {w_info['weights_path']} (N=104 Imgs, 261 GT)", 6)
    res_slide_rows = []
    for r in prelim_rows:
        if r["detector"] in ("orb_baseline", "sift_baseline", "yolov8n_pipeline"):
            if r["detector"] == "orb_baseline" and r["split"] not in ("normal", "shadowed_sub100lux"):
                continue
            if r["detector"] == "sift_baseline" and r["split"] not in ("normal", "shadowed_sub100lux"):
                continue
            res_slide_rows.append((
                format_split_label(r["split"]),
                format_det_label(r["detector"]),
                f"{float(r['precision']):.3f}",
                f"{float(r['recall']):.3f}",
                f"{float(r['f1_score']):.3f}",
                f"{float(r['mAP50']):.3f}",
                f"{float(r['mAP50_95']):.3f}",
                f"{float(r['mean_inference_ms']):.0f}ms ({float(r['fps']):.1f})",
            ))
    add_slide_table(
        s6, 26, 86, 560, 290,
        ["Condition", "Method", "P", "R", "F1", "mAP@.5", "mAP.5:.95", "Speed"],
        res_slide_rows,
        font_size=8.8,
        row_height=24,
        col_widths=[102, 102, 48, 48, 48, 56, 62, 94],
    )
    pc_slide_rows = [(c["class_id"], f"{c['ap50']:.3f}", f"{c['precision']:.3f}", f"{c['recall']:.3f}") for c in summary["per_class_normal_yolov8n"]]
    add_slide_table(
        s6, 598, 86, 334, 235,
        ["Class (Normal)", "AP@.50", "Prec", "Rec"],
        pc_slide_rows,
        font_size=8.8,
        row_height=24,
        col_widths=[148, 62, 62, 62],
    )
    add_slide_bullets(
        s6, 598, 360, 334, 130,
        "Safety & Video Checks",
        [
            f"False-Meat Safety: {fm['false_meat_count']}/{fm['total_pantry_negatives']} ({fm['false_meat_rate']:.4f})",
            f"Video Pan: {vid['predicted_confirmed_items']}/{vid['expected_unique_items']} items ({vid['fps']} FPS)",
            f"Model Size: {w_info['model_size_mb']} MB PT / {w_info['onnx_size_mb']} MB ONNX",
        ],
        font_size=10,
    )

    s7 = prs.Slides.Add(7, 12)
    add_slide_header(s7, "6. Progress Since Checkpoint 3 (cp3-core vs. cp4-80)", "Luminance-Gated Restoration Fixes Bright-Light Drift While Boosting <100 Lux Recall", 7)
    cp3_chart = os.path.join(milestone_dir, "cp3_vs_cp4_comparison.png")
    add_slide_image(s7, cp3_chart, 28, 135, 558)
    cp_slide_rows = [
        (r["condition"], f"{float(r['cp3_mAP50']):.3f}", f"{float(r['cp4_mAP50']):.3f}", f"{float(r['delta_mAP50']):+.3f}", f"{float(r['cp4_fps']):.1f}")
        for r in cp3_comp_rows
    ]
    add_slide_table(
        s7, 598, 86, 334, 170,
        ["Condition", "CP3", "CP4", "Delta", "FPS"],
        cp_slide_rows,
        font_size=9.0,
        row_height=22,
        col_widths=[122, 52, 52, 58, 50],
    )
    add_slide_bullets(
        s7, 598, 290, 334, 200,
        "Root Cause of CP3 -> CP4 Gains",
        [
            "CP3 applied CLAHE unconditionally, shifting bright contrast (-0.07 mAP).",
            "CP4 gates Gamma+CLAHE strictly to L<45.0.",
            "Normal mAP@0.50: 0.749 -> 0.819 (+0.070).",
            "Shadow mAP@0.50: 0.443 -> 0.675 (+0.232).",
            f"Laptop speed: ~{float(normal_yolo.get('mean_inference_ms', 0)):.1f} ms ({float(normal_yolo.get('fps', 0)):.1f} FPS).",
        ],
        font_size=10,
    )

    s8 = prs.Slides.Add(8, 12)
    add_slide_header(s8, f"7. Sample Outputs & Pre-Demo Test Cases (TC01–TC{tc_sum['total_cases']:02d})", f"Pre-Demo Verification: {tc_sum['passed']}/{tc_sum['total_cases']} Passed ({tc_sum['hard_cases']} Hard/Edge Cases)", 8)
    tc_slide_rows = []
    for row in test_cases_rows[:16]:
        inp = row["Input"]
        if len(inp) > 20:
            inp = inp[:10] + "..." + inp[-7:]
        tc_slide_rows.append((row["ID"], inp, row["CaseType"], row["Expected"][:22], row["Actual"][:32], row["Pass"]))
    add_slide_table(
        s8, 24, 82, 912, 380,
        ["ID", "Input", "Case Type", "Expected", "Actual Output", "Pass"],
        tc_slide_rows,
        font_size=8.0,
        row_height=21,
        col_widths=[45, 130, 125, 165, 392, 55],
    )

    s9 = prs.Slides.Add(9, 12)
    add_slide_header(s9, "8. Performance Analysis, Operating Envelope & Limitations", "Failure Taxonomy & Concrete Held-Out Error Examples (Green=GT, Red=Prediction)", 9)
    gal_png = os.path.join(milestone_dir, "failure_gallery.png")
    add_slide_image(s9, gal_png, 28, 84, 364)
    cc = summary["failure_analysis"]["cause_counts"]
    add_slide_bullets(
        s9, 414, 84, 518, 210,
        "Measured Limitations & Examples",
        [
            f"Occlusion & Clutter ({cc.get('Occlusion and clutter', 0)}): F01-F03 drop rear items when >=35% blocked.",
            f"Motion Blur ({cc.get('Image quality (blur/low-res)', 0)}): F11 (k=11 pan) smears vertical text edges.",
            f"Low-Light Cast Shadows ({cc.get('Lighting (shadows/glare)', 0)}): F04-F06 unlit corners lose color.",
            f"Angle/Scale ({cc.get('Viewpoint and scale', 0)}) & Look-Alikes ({cc.get('Look-alike classes', 0)}): F07-F10 + TC08 vinegar.",
        ],
        font_size=11,
    )
    add_slide_bullets(
        s9, 414, 308, 518, 190,
        "Defined Operating Conditions",
        [
            "Designed for: 0.3-1.2 m distance, >=30 lux lighting, +/-15 deg angle, <=20% overlap.",
            f"Not designed for: Synthetic graphics (demo_pantry_pan.mp4: {vid['predicted_confirmed_items']}/{vid['expected_unique_items']} items), >=35% central occlusion, heavy motion blur, or opaque unprinted jars.",
        ],
        font_size=11,
    )

    s10 = prs.Slides.Add(10, 12)
    add_slide_header(s10, "9. Live Demo Plan & Remaining Roadmap to 100%", "Classroom Verification Protocol & Task Owners (docs/remaining.md)", 10)
    add_slide_bullets(
        s10, 32, 92, 435, 395,
        "Live Classroom Demo Protocol",
        [
            "1. Launch local FastAPI server (python src/server/app.py) once; model cached in RAM.",
            f"2. Open Web Dashboard (http://127.0.0.1:8000/) & test TC01-TC{tc_sum['total_cases']:02d} live (normal + hard cases).",
            "3. Show Android CameraX live bounding-box overlay & detector switcher (YOLOv8 vs ORB).",
            "4. Run teacher's unseen images and confirm evaluate.py matches slide/report metrics.",
        ],
        font_size=12.5,
    )
    add_slide_bullets(
        s10, 492, 92, 435, 395,
        "Remaining Tasks to 100% (Milestone 5)",
        [
            "R1 (Ogayon, 11/24): Record 3 real kitchen MP4 shelf/fridge pans replacing synthetic clip.",
            "R2 (Sarmiento, 11/26): 50-epoch Stage-2 fine-tune + 60 hard crops for cereal/pasta/coffee.",
            "R3 (Sarmiento, 11/27): Add k=7..13 motion-blur & +/-20 deg perspective augmentations.",
            "R4 (Ogayon, 11/29): Finalize release APK & run full E2E integration tests.",
            "R5 (Kumar & All, 12/03): Final Milestone 5 technical report & 12-slide defense deck.",
        ],
        font_size=12,
    )

    prs.SaveAs(os.path.abspath(pptx_path))
    prs.SaveAs(os.path.abspath(pdf_path), 32)
    prs.Close()
    pp.Quit()
    return pptx_path, pdf_path


def write_draft_technical_report_md(milestone_dir):
    summary, prelim_rows, cp3_comp_rows, test_cases_rows, _ = load_milestone4_artifacts(milestone_dir)
    w_info = summary["weights"]
    hw = summary["hardware"]
    vid = summary["video_evaluation"]
    fm = summary["false_meat_evaluation"]
    fa = summary["failure_analysis"]
    cc = fa["cause_counts"]
    tc_sum = summary["test_cases_summary"]

    lines = [
        "# Stock-to-Stove: Draft Technical Report (Milestone 4 — `cp4-80`)",
        "",
        "**Course:** CSST 106 — Computer Vision Final Project  ",
        "**Team Kaggleing:** Aron Gabriel L. Ogayon (Lead), Allain A. Kumar (Documentation), Janvher Lucas J. Sarmiento (Developer)  ",
        f"**Evaluated Weights:** `{w_info['weights_path']}` (`{w_info['model_size_mb']} MB` PyTorch, `{w_info['onnx_size_mb']} MB` ONNX)  ",
        f"**Demo Hardware:** {hw['processor']} ({hw['cpu_threads']} CPU threads, {hw['os']})  ",
        "",
        "---",
        "",
        "## 1. Introduction & Problem Statement",
        "Managing home pantry and refrigerator inventory is manual, error-prone, and visually cluttered. Forgotten rear-shelf items lead to duplicate grocery purchases and preventable food waste, while decision fatigue drives unnecessary takeout orders. Barcode-only apps fail on loose produce, repackaged grains, and occluded containers.",
        "",
        "**Objectives:**",
        "1. Build a unified 8-class pantry and refrigerator object detection pipeline robust to dark cupboard shadows (`<100 lux`), shelf clutter, and camera pan motion.",
        "2. Benchmark classical planar feature matching (`ORB` and `SIFT` + `RANSAC`) against a fine-tuned single-stage deep detector (`YOLOv8n`) on a locked held-out test set.",
        "3. Achieve real-time CPU throughput (`>=15 FPS`) using a single cached model instance inside an interactive browser web dashboard (`GET /`), native Android CameraX app, and CLI (`src/run.py`).",
        "4. Connect confirmed ingredient counts to a downstream recipe generator producing three zero-waste cooking suggestions.",
        "",
        "---",
        "",
        "## 2. Dataset & Preprocessing",
        "The dataset (`data/dataset_yolo/data.yaml`) unifies 8 domestic grocery categories across Pantry (`pasta_noodles`, `canned_tomato`, `canned_tuna_meat`, `cereal_grains`, `coffee_condiment`) and Refrigerator (`milk_dairy`, `eggs`, `fresh_produce`) zones.",
        "",
        "| Partition | Images | GT Boxes | Negatives | Composition |",
        "| :--- | :---: | :---: | :---: | :--- |",
        "| Train | 605 | 1,937 | 74 empty | Freiburg (225), Kitchen-YOLO (146), Fridge (76), Composite (65), Negatives (74), Pantry (19) |",
        "| Validation | 104 | 282 | 13 empty | Freiburg (57), Kitchen-YOLO (18), Fridge (12), Pantry (4), Negatives (13) |",
        "| Held-Out Test (Locked) | 104 | 261 | 13 empty | Freiburg (57), Kitchen-YOLO (18), Fridge (12), Pantry (4), Negatives (13) |",
        "| Pantry Negatives | 115 | 0 | 115 | Non-meat pantry safety benchmark |",
        "| Video Pan Clip | 60 frames | 9 shelf items | Synthetic | `data/raw_home/demo_pantry_pan.mp4` (`640x480` @ 15 FPS) |",
        "",
        "**Preprocessing (`src/cv/preprocess.py`):** Every frame passes through luminance-gated low-light restoration and aspect-preserving `640x640` letterboxing (padding value `114`). When mean LAB lightness `L < 45.0`, inverse gamma restoration `(img / 0.26)^(1/2.4)` and L-channel CLAHE (`clipLimit=1.8`, `8x8` grid) recover shadow detail; well-lit frames (`L >= 45.0`) bypass CLAHE to prevent contrast drift against training distributions.",
        "",
        "---",
        "",
        "## 3. Methodology & Integrated Application",
        "The unified procedural pipeline (`src/run.py`) executes `preprocess() -> infer() -> postprocess() -> render()`:",
        "- **Workflow Diagram:** ![Workflow Diagram](workflow_diagram.png)",
        "- **Classical Baselines (`src/cv/baseline_detector.py`):** Extracts 1,000 ORB or SIFT keypoints, matches multi-angle templates with Lowe's ratio test (`0.75`), and estimates 4-point homography via RANSAC (`min_inliers=6`).",
        "- **Main Detector (`src/cv/yolo_detector.py`):** Fine-tuned YOLOv8n (`models/best.pt`) cached once in `MODEL_REGISTRY`. Also supports `hybrid_gemini` and `gemini_vision` modes in the server.",
        "- **Post-Processing (`src/cv/temporal_tracker.py`):** Class-wise NMS (`IoU=0.45`), meat-safety confidence gate (`canned_tuna_meat >= 0.76`), and a 5-frame sliding window (`K >= 3` hits for video confirmation).",
        "- **Application Interfaces:** Browser Dashboard (`GET /` in `src/server/app.py`), Native Android CameraX WebSocket App (`android/`), and CLI (`src/run.py`).",
        "",
        "![Application Interface](application_interface_overview.png)",
        "",
        "---",
        "",
        "## 4. Experiments & Reproducible Results (`src/evaluate.py`)",
        f"Reproduced via `python src/evaluate.py --split test --weights {w_info['weights_path']}` (`seed=42`, `conf=0.45`, `NMS IoU=0.45`, `eval IoU=0.50`).",
        "",
        "### 4.1 Held-Out Test Set Metrics Across Conditions",
        "| Condition | Method | Precision | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 | Speed (ms / FPS) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in prelim_rows:
        lines.append(
            f"| {format_split_label(r['split'])} | {format_det_label(r['detector'])} | "
            f"{float(r['precision']):.4f} | {float(r['recall']):.4f} | {float(r['f1_score']):.4f} | "
            f"{float(r['mAP50']):.4f} | {float(r['mAP50_95']):.4f} | "
            f"{float(r['mean_inference_ms']):.2f} ms / {float(r['fps']):.2f} FPS |"
        )

    lines.extend([
        "",
        "### 4.2 Per-Class Normal Held-Out Test Metrics (YOLOv8n)",
        "| Class ID | AP@0.50 | Precision | Recall |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for c in summary["per_class_normal_yolov8n"]:
        lines.append(f"| `{c['class_id']}` | {c['ap50']:.4f} | {c['precision']:.4f} | {c['recall']:.4f} |")

    lines.extend([
        "",
        "### 4.3 Checkpoint 3 (`cp3-core`) vs. Checkpoint 4 (`cp4-80`) Comparison",
        "| Condition | CP3 mAP@0.50 | CP4 mAP@0.50 | Delta mAP | CP3 F1 | CP4 F1 | Delta F1 | CP4 FPS |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in cp3_comp_rows:
        lines.append(
            f"| {r['condition']} | {float(r['cp3_mAP50']):.4f} | {float(r['cp4_mAP50']):.4f} | {float(r['delta_mAP50']):+.4f} | "
            f"{float(r['cp3_f1']):.4f} | {float(r['cp4_f1']):.4f} | {float(r['delta_f1']):+.4f} | {float(r['cp4_fps']):.2f} |"
        )

    lines.extend([
        "",
        "**Cause of Change Since Checkpoint 3:** Unconditional LAB CLAHE in `cp3-core` degraded normal well-lit mAP@0.50 from `0.8190` to `0.7494` and only achieved `0.4427` under `<100 lux` shadows. Switching to luminance-gated Gamma + CLAHE (`mean L < 45.0`) restored normal contrast (`0.8190`) while elevating `<100 lux` shadow mAP@0.50 to `0.6750` (`+0.2323`) and boosting motion blur (`+0.0972`) and angled (`+0.1049`) mAP@0.50.",
        "",
        f"**Video Pan & False-Meat Checks:** `demo_pantry_pan.mp4` achieved `{vid['predicted_confirmed_items']}/{vid['expected_unique_items']}` confirmed items at `{vid['fps']} FPS` (`{vid['mean_inference_ms']} ms/frame`). False-meat triggers on 115 negative scenes: `{fm['false_meat_count']}/{fm['total_pantry_negatives']}` (`{fm['false_meat_rate']:.4f}`).",
        "",
        "---",
        "",
        "## 5. Performance Analysis, Operating Conditions & Limitations",
        "- **Where It Works:** High reliability on well-lit and cluttered real pantry/fridge scenes (`Normal mAP@0.50 = 0.8190`, `Cluttered >=4 items = 0.8665`), especially `eggs` (`1.0000`), `canned_tomato` (`0.9478`), `canned_tuna_meat` (`0.9218`), `milk_dairy` (`0.8987`), and `fresh_produce` (`0.8752`).",
        "- **Operating Conditions:** Handheld distance `0.3–1.2 m`, ambient lighting `>=30 lux` (or shadowed scenes corrected by luminance-gated CLAHE), frontal to `+-15 deg` angle, and `<=20%` item overlap.",
        "- **Not Designed For:** Synthetic/cartoon imagery (`demo_pantry_pan.mp4` missed " + str(vid['expected_unique_items'] - vid['predicted_confirmed_items']) + f"/{vid['expected_unique_items']} blocks with {vid['frame_overcounts']} per-frame overcounts), severe `>=35%` central foreground occlusion, heavy `k>=11` motion blur, or unlabeled opaque containers.",
        f"- **Measured Error Causes (`n={fa['total_attributed_failures']}` events across test runs):** Occlusion and clutter (`{cc.get('Occlusion and clutter', 0)}`), Image quality/blur (`{cc.get('Image quality (blur/low-res)', 0)}`), Lighting/shadows (`{cc.get('Lighting (shadows/glare)', 0)}`), Viewpoint/scale (`{cc.get('Viewpoint and scale', 0)}`), Label/domain gap (`{cc.get('Label errors / domain gap', 0)}`), Look-alike classes (`{cc.get('Look-alike classes', 0)}`). See `failure_gallery.png` (`F01`–`F12`).",
        "",
        "---",
        "",
        f"## 6. Pre-Demo Test Cases (`TC01`–`TC{tc_sum['total_cases']:02d}`)",
        f"Summary: **{tc_sum['passed']}/{tc_sum['total_cases']} PASS** ({tc_sum['hard_cases']} hard/edge cases).",
        "",
        "| ID | Input | Condition | Type | Expected | Actual | Pass |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :---: |",
    ])
    for row in test_cases_rows:
        lines.append(f"| {row['ID']} | `{row['Input']}` | {row['Condition']} | {row['CaseType']} | {row['Expected']} | {row['Actual']} | {row['Pass']} |")

    out_path = os.path.join(milestone_dir, "DRAFT_TECHNICAL_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def write_draft_slides_md(milestone_dir):
    summary, prelim_rows, cp3_comp_rows, test_cases_rows, _ = load_milestone4_artifacts(milestone_dir)
    w_info = summary["weights"]
    vid = summary["video_evaluation"]
    tc_sum = summary["test_cases_summary"]
    lines = [
        "# Stock-to-Stove: Milestone 4 Draft Presentation Slides (10 Slides)",
        "",
        "*(Companion Markdown version of `CSST106_Milestone4_Slides_Stock2Stove.pptx` and `.pdf`)*",
        "",
        "---",
        "## Slide 1 — Title & Project Overview",
        "- **Stock-to-Stove:** Automated Pantry Inventory Detection & Zero-Waste Recipe Application",
        "- **Team Kaggleing:** Ogayon, Kumar, Sarmiento | **Tag:** `cp4-80`",
        f"- **Weights:** `{w_info['weights_path']}` (`{w_info['model_size_mb']} MB` PT / `{w_info['onnx_size_mb']} MB` ONNX)",
        "",
        "---",
        "## Slide 2 — Problem Statement & Objectives",
        "- Household grocery waste caused by hidden rear-shelf items and decision fatigue.",
        "- Barcodes fail on fresh produce, loose bags, and occluded shelves.",
        "- Objectives: 8-class detection, baseline vs. YOLOv8n evaluation, >=15 FPS CPU throughput, Web/Android UI + LLM recipes.",
        "",
        "---",
        "## Slide 3 — Dataset & 8-Class Schema",
        "- 813 annotated images (2,480 boxes) + 115 pantry negatives + 60-frame video clip.",
        "- Train: 605 | Valid: 104 | Held-Out Test (frozen): 104 (261 GT boxes).",
        "- 5 Pantry classes (`pasta_noodles`, `canned_tomato`, `canned_tuna_meat`, `cereal_grains`, `coffee_condiment`) + 3 Fridge classes (`milk_dairy`, `eggs`, `fresh_produce`).",
        "",
        "---",
        "## Slide 4 — Technical Approach (Baselines vs. Main Model)",
        "- Classical Baselines: ORB / SIFT (1,000 keypoints) + Lowe's ratio (0.75) + RANSAC homography -> brittle under curvature, shadows, and blur.",
        "- Main Model: Fine-tuned YOLOv8n + Luminance-Gated Gamma/CLAHE (`mean L < 45.0`) + Class NMS (0.45) + 5-Frame Temporal Voter (`K>=3`).",
        "",
        "---",
        "## Slide 5 — End-to-End Pipeline & Application Interface",
        "- Single cached model instance in `src/run.py` & `src/server/app.py`.",
        "- Browser Dashboard (`GET /`), Native Android CameraX WebSocket client, CLI (`src/run.py`).",
        "- Visual diagrams: `workflow_diagram.png` & `application_interface_overview.png`.",
        "",
        "---",
        "## Slide 6 — Reproducible Evaluation Results (`evaluate.py`)",
        "- Command: `python src/evaluate.py --split test --weights models/best.pt`.",
        "- Normal Held-Out: YOLOv8n mAP@0.50 = `0.8190`, F1 = `0.8544` vs. ORB (`0.0109`) & SIFT (`0.0246`).",
        "- Cluttered (>=4 items): `0.8665` mAP@0.50 | False-Meat Safety: `1/115` (`0.0087`).",
        "",
        "---",
        "## Slide 7 — Progress Since Checkpoint 3 (`cp3-core` -> `cp4-80`)",
        "| Condition | CP3 mAP@0.50 | CP4 mAP@0.50 | Delta | CP4 FPS |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in cp3_comp_rows:
        lines.append(f"| {r['condition']} | {float(r['cp3_mAP50']):.4f} | {float(r['cp4_mAP50']):.4f} | {float(r['delta_mAP50']):+.4f} | {float(r['cp4_fps']):.2f} |")
    lines.extend([
        "",
        "---",
        "## Slide 8 — Sample Outputs & Pre-Demo Test Cases",
        f"- {tc_sum['total_cases']} pre-demo test cases ({tc_sum['hard_cases']} hard/edge): {tc_sum['passed']} PASS, {tc_sum['failed']} FAIL (`test_cases.csv`).",
        "- Includes normal daylight, empty kitchen negatives, look-alike vinegar bottle, cluttered shelves, <100 lux shadows, 35% occlusion, motion blur, +/-16 deg tilt, and video pan.",
        "",
        "---",
        "## Slide 9 — Performance Analysis, Operating Conditions & Limitations",
        "- Works reliably on well-lit/cluttered real shelves (`0.3-1.2 m`, `>=30 lux`, `<=20%` overlap).",
        f"- Fails on synthetic video graphics (`demo_pantry_pan.mp4` {vid['predicted_confirmed_items']}/{vid['expected_unique_items']} items), >=35% central occlusion (`F01-F03`), `k=11` motion blur (`F11`), and cereal/pasta/coffee/vinegar look-alikes (`F07-F08`, `TC08`).",
        "",
        "---",
        "## Slide 10 — Live Demo Plan & To-100% Roadmap",
        f"- Live Demo: Start FastAPI server (`http://127.0.0.1:8000/`), test `TC01`–`TC{tc_sum['total_cases']:02d}`, switch detectors live, show Android CameraX overlay, run unseen teacher images, verify `evaluate.py`.",
        "- Remaining Tasks (`docs/remaining.md`): R1 Real Video Clips (11/24), R2 Stage-2 Hardening (11/26), R3 Blur/Angle Aug (11/27), R4 App Polish (11/29), R5 Final Report & Deck (12/03).",
    ])
    out_path = os.path.join(milestone_dir, "DRAFT_PRESENTATION_SLIDES.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def write_milestone4_readme(milestone_dir):
    summary, prelim_rows, cp3_comp_rows, test_cases_rows, _ = load_milestone4_artifacts(milestone_dir)
    w_info = summary["weights"]
    vid = summary["video_evaluation"]
    fm = summary["false_meat_evaluation"]
    tc_sum = summary["test_cases_summary"]
    cc = summary["failure_analysis"]["cause_counts"]

    lines = [
        "# Milestone 4 — 80% Progress Check (`cp4-80`)",
        "",
        "## 1. Launching the Integrated Application",
        "```powershell",
        "python src/server/app.py",
        "```",
        "- Browser UI: `http://127.0.0.1:8000/`",
        "- CLI pipeline: `python src/run.py --input data/test/ --output outputs/run_latest/`",
        "",
        "## 2. Reproducing All Evaluation Numbers (`src/evaluate.py`)",
        "```powershell",
        "python src/evaluate.py --split test --weights models/best.pt",
        "```",
        f"- **Evaluated Weights:** `{w_info['weights_path']}` (`{w_info['model_size_mb']} MB` PT / `{w_info['onnx_size_mb']} MB` ONNX)",
        "- **Outputs Written:** `outputs/eval/metrics.json`, `outputs/eval/confusion_matrix.png`, and `docs/milestone4/*`.",
        "",
        "## 3. Held-Out Test Results Summary (`docs/milestone4/preliminary_results.csv`)",
        "| Condition | Method | Precision | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 | Speed (ms / FPS) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in prelim_rows:
        lines.append(
            f"| {format_split_label(r['split'])} | {format_det_label(r['detector'])} | "
            f"{float(r['precision']):.4f} | {float(r['recall']):.4f} | {float(r['f1_score']):.4f} | "
            f"{float(r['mAP50']):.4f} | {float(r['mAP50_95']):.4f} | "
            f"{float(r['mean_inference_ms']):.2f} ms / {float(r['fps']):.2f} FPS |"
        )

    lines.extend([
        "",
        "## 4. Progress Since Checkpoint 3 (`docs/milestone4/cp3_vs_cp4_comparison.csv`)",
        "| Condition | CP3 mAP@0.50 | CP4 mAP@0.50 | Delta mAP@0.50 | CP3 F1 | CP4 F1 | Delta F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in cp3_comp_rows:
        lines.append(
            f"| {r['condition']} | {float(r['cp3_mAP50']):.4f} | {float(r['cp4_mAP50']):.4f} | {float(r['delta_mAP50']):+.4f} | "
            f"{float(r['cp3_f1']):.4f} | {float(r['cp4_f1']):.4f} | {float(r['delta_f1']):+.4f} |"
        )

    lines.extend([
        "",
        f"## 5. Pre-Demo Test Cases (`docs/milestone4/test_cases.csv`): {tc_sum['passed']}/{tc_sum['total_cases']} PASS ({tc_sum['hard_cases']} Hard/Edge)",
        f"- **Video Clip (`{vid['clip_name']}`):** {vid['predicted_confirmed_items']}/{vid['expected_unique_items']} confirmed @ {vid['fps']} FPS ({vid['mean_inference_ms']} ms/frame)",
        f"- **False-Meat Safety (`pantry_negatives`):** {fm['false_meat_count']}/{fm['total_pantry_negatives']} ({fm['false_meat_rate']:.4f})",
        "- **Failure Taxonomy:** " + ", ".join([f"{k}: {v}" for k, v in sorted(cc.items(), key=lambda x: x[1], reverse=True)]),
        "",
        "## 6. Deliverables in `docs/milestone4/`",
        "- `CSST106_Milestone4_80PercentCheck_Stock2Stove.docx` & `.pdf`",
        "- `CSST106_Milestone4_Slides_Stock2Stove.pptx` & `.pdf` (10 slides)",
        "- `DRAFT_TECHNICAL_REPORT.md` & `DRAFT_PRESENTATION_SLIDES.md`",
        "- `remaining.md` (mirrored at `docs/remaining.md`)",
        "- `preliminary_results.csv`, `preliminary_results_comparison.png`, `cp3_vs_cp4_comparison.csv`, `cp3_vs_cp4_comparison.png`",
        "- `test_cases.csv`, `test_log.csv`, `classification_report.txt`, `test_confusion_matrix.png`",
        "- `failure_gallery.png`, `failures/`, `pipeline_stages_strip.png`, `workflow_diagram.png`, `application_interface_overview.png`",
        "- `milestone4_evaluation_summary.json`",
    ])
    readme_path = os.path.join(milestone_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return readme_path


def main():
    milestone_dir = os.path.join(PROJECT_ROOT, "docs", "milestone4")
    os.makedirs(milestone_dir, exist_ok=True)
    workflow_png = render_workflow_diagram(os.path.join(milestone_dir, "workflow_diagram.png"))
    app_ui_png = render_application_interface_overview(os.path.join(milestone_dir, "application_interface_overview.png"))
    root_rem, m4_rem = write_remaining_md(milestone_dir)
    docx_path, pdf_path, pages = build_milestone4_docx(milestone_dir, workflow_png, app_ui_png)
    pptx_path, slides_pdf = build_milestone4_slides(milestone_dir, workflow_png, app_ui_png)
    report_md = write_draft_technical_report_md(milestone_dir)
    slides_md = write_draft_slides_md(milestone_dir)
    readme_path = write_milestone4_readme(milestone_dir)
    print(json.dumps({
        "workflow_png": workflow_png,
        "app_ui_png": app_ui_png,
        "docx": docx_path,
        "pdf": pdf_path,
        "pdf_pages": pages,
        "pptx": pptx_path,
        "slides_pdf": slides_pdf,
        "draft_report_md": report_md,
        "draft_slides_md": slides_md,
        "remaining_root": root_rem,
        "remaining_m4": m4_rem,
        "readme": readme_path,
    }, indent=2))


if __name__ == "__main__":
    main()
