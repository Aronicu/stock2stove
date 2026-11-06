"""
Script to generate the CSST 106 Milestone 1 Project Proposal document
in professional Word (.docx) and PDF format.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def create_proposal_docx(output_docx_path="docs/CSST106_Milestone1_Proposal_Stock2Stove.docx"):
    os.makedirs(os.path.dirname(output_docx_path), exist_ok=True)
    doc = docx.Document()

    # Set page margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(40, 40, 40)

    # Header / Meta
    p_meta = doc.add_paragraph()
    r_course = p_meta.add_run("CSST 106 — Computer Vision Final Project  |  Milestone 1 — Project Proposal and Initial Planning\n")
    r_course.font.size = Pt(9.5)
    r_course.font.bold = True
    r_course.font.color.rgb = RGBColor(100, 100, 100)

    # Title
    p_title = doc.add_paragraph()
    r_title = p_title.add_run("Milestone 1: Project Proposal\nStock-to-Stove")
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(24, 76, 120)

    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run("Automated Pantry Shelf Inventory Tracking via Fine-Tuned YOLOv8 Object Detection with Downstream Generative LLM Recipe Suggestion")
    r_sub.font.size = Pt(12)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(80, 80, 80)

    # Meta Table Box
    meta_table = doc.add_table(rows=2, cols=3)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False

    meta_data = [
        [("Project Code:", "stock2stove"), ("Tag:", "cp1-proposal"), ("Target Points:", "10 / 10 pts")],
        [("Submission Date:", "Oct 15, 11:30 PM"), ("In-Class Pitch:", "Oct 16"), ("Safety Net / Fallback:", "Defined & Feasible")]
    ]

    for r_idx, row in enumerate(meta_data):
        for c_idx, (label, val) in enumerate(row):
            cell = meta_table.cell(r_idx, c_idx)
            set_cell_background(cell, "F2F5F8")
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            r_lbl = p.add_run(f"{label} ")
            r_lbl.font.size = Pt(8.5)
            r_lbl.font.bold = True
            r_lbl.font.color.rgb = RGBColor(60, 60, 60)
            r_v = p.add_run(val)
            r_v.font.size = Pt(8.5)
            r_v.font.color.rgb = RGBColor(24, 76, 120)

    doc.add_paragraph()

    # Helper for Headings
    def add_h1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(4)
        run = h.add_run(text)
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(24, 76, 120)
        return h

    def add_h2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(3)
        run = h.add_run(text)
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(45, 110, 160)
        return h

    # 1. System Title
    add_h1("1. Title")
    p = doc.add_paragraph()
    p.add_run("Stock-to-Stove: Shelf Inventory Detection via Fine-Tuned YOLO with Downstream LLM Recipe Generation").bold = True

    # 2. Problem Statement
    add_h1("2. Problem Statement")
    p = doc.add_paragraph()
    p.add_run("Who has the problem? ").bold = True
    p.add_run(
        "Everyday home cooks, busy working households, students, and meal preppers managing residential kitchen pantries and cupboards."
    )
    p2 = doc.add_paragraph()
    p2.add_run("What they do today: ").bold = True
    p2.add_run(
        "Pantry inventory management is overwhelmingly manual, error-prone, and neglected. To check what groceries they have in stock, "
        "users must manually open dark cabinets, shuffle front-row boxes, and guess what remains in the back. Because people forget what is "
        "already in their cupboards, they frequently purchase duplicate items (e.g., buying more canned tomatoes while three jars sit hidden). "
        "Simultaneously, forgotten ingredients expire silently, driving part of the estimated $1,500 in grocery waste generated annually per household. "
        "Furthermore, even with ingredients present, cooks experience severe 'decision fatigue' trying to mentally combine available items into a meal."
    )
    p3 = doc.add_paragraph()
    p3.add_run("Why vision helps: ").bold = True
    p3.add_run(
        "A brief 5-second video pan across shelves captured on a standard smartphone camera replaces tedious barcode scanning or manual logging. "
        "Modern deep object detectors can instantly isolate, classify, and count branded packaging across severe clutter, perspective shifts, and deep "
        "shelf shadows. By turning raw visual pixels into a verified, deduplicated list of detected ingredients, the system effortlessly feeds a lightweight "
        "LLM API to generate instant, zero-waste recipes, bridging computer vision detection directly to tangible kitchen action."
    )

    # 3. Objectives & Expected Outcomes
    add_h1("3. Objectives and Expected Outcomes")
    doc.add_paragraph("The project is governed by strict, measurable engineering benchmarks:")
    
    obj_table = doc.add_table(rows=5, cols=3)
    obj_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Metric / Target Area", "Measurable Engineering Target", "Evaluation Method"]
    for c_idx, h_text in enumerate(headers):
        cell = obj_table.cell(0, c_idx)
        set_cell_background(cell, "2C5282")
        p = cell.paragraphs[0]
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(255, 255, 255)

    obj_rows = [
        ("Detection Accuracy", "mAP@0.5 >= 0.85 and mAP@0.5:0.95 >= 0.60 across 5 target classes", "Evaluated on held-out test split (37 frames, ~60 instances)"),
        ("Inference Speed", ">= 15 FPS on laptop CPU (Intel Core i5/i7) and >= 45 FPS on CUDA GPU", "Timed over 100 consecutive video frames in Python"),
        ("Baseline Outperformance", ">= 30% higher Recall & F1 over SIFT/ORB baseline under occlusion & rotation", "Benchmarked on rotated (>30 deg) and occluded (20-50%) test subset"),
        ("Downstream Latency", "End-to-end recipe response delivered in < 2.5 seconds via LLM API", "Wall-clock API round-trip measurement using Gemini 2.5 Flash")
    ]

    for r_idx, row in enumerate(obj_rows, start=1):
        for c_idx, val in enumerate(row):
            cell = obj_table.cell(r_idx, c_idx)
            set_cell_background(cell, "FFFFFF" if r_idx % 2 == 1 else "F9FBFD")
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)

    doc.add_paragraph()
    p_fallback = doc.add_paragraph()
    p_fallback.add_run("Minimum Deliverable Version (Safety Net): ").bold = True
    p_fallback.add_run(
        "If fine-tuning YOLOv8 fails to converge or exhibits severe overfitting, we will deploy pretrained open-vocabulary YOLO-World / COCO detectors "
        "for generic containers ('bottle', 'box', 'can') coupled with multi-template ORB homography on cropped logos, ensuring an end-to-end live demo from shelf video to recipe suggestion is guaranteed."
    )

    # 4. Proposed Approach and Algorithms
    add_h1("4. Proposed Approach and Algorithms")
    doc.add_paragraph("The pipeline operates across five modular stages, maintaining Computer Vision as the core technical focus:")

    pipe_steps = [
        ("Stage 1 — Acquisition:", "Smartphone video (1080p, 30 FPS) captured panning across pantry shelves, subsampled at 1–2 FPS to extract non-redundant frames. (Why: Provides realistic, high-resolution domestic shelf video with natural motion blur and lighting shifts.)"),
        ("Stage 2 — Preprocessing:", "Letterbox resizing to 640x640 preserving packaging aspect ratios, plus Luminance CLAHE contrast normalization. (Why: CLAHE neutralizes deep cupboard shadows without saturating bright front-row boxes.)"),
        ("Stage 3A — Classical Baseline (SIFT/ORB):", "Cropped reference package templates matched against shelf frames using ORB/SIFT descriptors, Brute-Force/FLANN matcher, and RANSAC homography. (Why: Establishes a fast, interpretable classical baseline that operates without training data and reveals planar matching limitations.)"),
        ("Stage 3B — Main Method (YOLOv8n / YOLO11n):", "Single-stage detector fine-tuned using transfer learning with CSPDarknet backbone, PAN-FPN neck, and anchor-free detection head. (Why: YOLOv8 delivers state-of-the-art CPU inference speed while offering robust multi-scale feature representations that easily overcome packaging deformations and heavy occlusions.)"),
        ("Stage 4 — Post-Processing & Filtering:", "Non-Maximum Suppression (IoU >= 0.45, conf >= 0.50) combined with a multi-frame temporal voting filter requiring presence across >= 3 consecutive frames. (Why: Eliminates transient false positives and bounding box flicker during video panning.)"),
        ("Stage 5 — Downstream LLM Integration:", "Deduplicated detected items are passed into a constrained Python f-string prompt querying Gemini 2.5 Flash / OpenAI API for 3 practical recipes. (Why: Adds compelling user-facing AI utility in ~15 lines of modular code without distracting from the core vision timeline.)")
    ]

    for title, desc in pipe_steps:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.2)
        p.paragraph_format.space_after = Pt(3)
        r_t = p.add_run(f"{title} ")
        r_t.bold = True
        r_t.font.color.rgb = RGBColor(24, 76, 120)
        p.add_run(desc)

    # 5. Data
    add_h1("5. Data")
    p_data = doc.add_paragraph()
    p_data.add_run("Source & Collection: ").bold = True
    p_data.add_run("Primary smartphone video footage captured directly by the team in home kitchen pantries. Sample subsets are committed to data/samples/ and full datasets hosted on Roboflow Universe under the Creative Commons Attribution 4.0 International (CC-BY 4.0) license.\n")
    p_data.add_run("Target Classes (5 Pantry Staples): ").bold = True
    p_data.add_run("1) Boxed Pasta (Barilla Penne), 2) Canned Crushed Tomatoes (Hunt's), 3) Cereal Box (Cheerios), 4) Ground Coffee Bag (Starbucks/Folgers), 5) Canned Tuna/Beans (StarKist).\n")
    p_data.add_run("Volume & Split: ").bold = True
    p_data.add_run("250 labeled frames containing >= 50 labeled instances per class (~400 total annotations). Split 70% Train (175 frames), 15% Validation (38 frames), and 15% Held-out Test (37 frames), stratified across lighting conditions (well-lit, shadowed < 100 lux) and occlusion levels (0%, 20-40%, 40-60%).")

    # 6. Architecture Diagram
    add_h1("6. Architecture and Workflow Diagram")
    doc.add_paragraph(
        "The system architecture partitions input acquisition, baseline vs. deep learning detection, temporal filtering, and the downstream LLM hand-off:"
    )

    flow_table = doc.add_table(rows=5, cols=2)
    flow_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    flow_stages = [
        ("Input Stage", "Pantry Video (1080p @ 30 FPS) -> Frame Extractor (1-2 FPS) -> CLAHE Preprocessing & Letterbox 640x640"),
        ("Baseline Branch", "Cropped Package Templates -> SIFT/ORB Descriptors -> FLANN Matcher + RANSAC -> Homography Box"),
        ("Main Method Branch", "Fine-Tuned YOLOv8n (CSPDarknet + PAN-FPN) -> Anchor-Free Task-Aligned Bounding Box Predictor"),
        ("Post-Processing", "Non-Maximum Suppression (IoU >= 0.45) -> Multi-Frame Temporal Consistency Filter (>= 3 frames)"),
        ("Downstream LLM", "Deduplicated Inventory Array -> Zero-Shot Constrained Prompt -> Gemini Flash API -> 3 Recipe Cards")
    ]
    for r_idx, (stg, det) in enumerate(flow_stages):
        c1 = flow_table.cell(r_idx, 0)
        c2 = flow_table.cell(r_idx, 1)
        set_cell_background(c1, "2C5282")
        set_cell_background(c2, "F7FAFC")
        set_cell_margins(c1, top=60, bottom=60, left=100, right=100)
        set_cell_margins(c2, top=60, bottom=60, left=100, right=100)
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(stg)
        r1.font.bold = True
        r1.font.color.rgb = RGBColor(255, 255, 255)
        p2 = c2.paragraphs[0]
        p2.add_run(det)

    doc.add_paragraph()

    # 7. Task Assignments
    add_h1("7. Task Assignments")
    doc.add_paragraph("Every team member owns substantive code modules and testing pipelines:")

    task_table = doc.add_table(rows=6, cols=3)
    task_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_headers = ["Module & Scope", "Responsible Owner", "Code Deliverables"]
    for c_idx, h in enumerate(t_headers):
        cell = task_table.cell(0, c_idx)
        set_cell_background(cell, "2C5282")
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(255, 255, 255)

    tasks = [
        ("Module 1: Data Acquisition & Preprocessing", "Lead A (Data & Vision)", "src/data_prep.py, src/augment.py, Roboflow dataset"),
        ("Module 2: Classical Baseline (SIFT/ORB)", "Lead B (Classical CV)", "src/baseline_sift.py, src/baseline_orb.py, benchmark notebook"),
        ("Module 3: YOLOv8 Training & Optimization", "Lead A & C (Deep Learning)", "src/train_yolo.py, src/evaluate.py, trained weights"),
        ("Module 4: Downstream LLM & Demo UI", "Lead C (Systems & LLM)", "src/llm_recipe.py, src/app_demo.py, terminal & UI demo"),
        ("Module 5: Benchmarking, Ablation & Reports", "All Members", "outputs/benchmark_results.csv, final report & slides")
    ]
    for r_idx, row in enumerate(tasks, start=1):
        for c_idx, val in enumerate(row):
            cell = task_table.cell(r_idx, c_idx)
            set_cell_background(cell, "FFFFFF" if r_idx % 2 == 1 else "F9FBFD")
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)

    doc.add_paragraph()

    # 8. Timeline
    add_h1("8. Timeline Aligned to CSST 106 Checkpoints")
    time_table = doc.add_table(rows=7, cols=4)
    time_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    time_headers = ["Checkpoint", "Hand-In Deadline", "Checked In Class", "Core Technical Deliverables"]
    for c_idx, h in enumerate(time_headers):
        cell = time_table.cell(0, c_idx)
        set_cell_background(cell, "2C5282")
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(255, 255, 255)

    checkpoints = [
        ("CP1: Proposal & Initial Plan", "Thu Oct 15, 11:30 PM", "Fri Oct 16", "Proposal PDF, repo setup, feasibility spike running, 20-sample contact sheet (tag: cp1-proposal)"),
        ("CP2: 50% Progress Check", "Thu Oct 22, 11:30 PM", "Fri Oct 23", "250 labeled frames complete; SIFT/ORB baseline code operational; baseline precision/recall recorded (tag: cp2-50)"),
        ("CP3: Core Implementation", "Thu Oct 29, 11:30 PM", "Fri Oct 30", "YOLOv8 fine-tuned (50 epochs); validation mAP@0.5 verified; baseline vs YOLO comparison curves (tag: cp3-core)"),
        ("CP4: 80% Progress Check", "Thu Nov 12, 11:30 PM", "Fri Nov 13", "Occlusion stress tests; temporal multi-frame consistency filter; LLM recipe API connected and tested (tag: cp4-80)"),
        ("CP5: Final Presentation & Demo", "Thu Nov 19, 11:30 PM", "From Fri Nov 20", "End-to-end live video pipeline demo; complete README, benchmarks, slide deck (tag: v1.0)"),
        ("CP6: Technical Defense", "Thu Nov 19, 11:30 PM", "Scheduled Slot", "Individual member defense on model architecture, failure cases, loss curves, and code ownership")
    ]

    for r_idx, row in enumerate(checkpoints, start=1):
        for c_idx, val in enumerate(row):
            cell = time_table.cell(r_idx, c_idx)
            set_cell_background(cell, "FFFFFF" if r_idx % 2 == 1 else "F9FBFD")
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)

    doc.add_paragraph()

    # 9. Risks
    add_h1("9. Risks and Fallback Strategies")
    p_r1 = doc.add_paragraph()
    p_r1.add_run("Risk 1: Severe Shelf Occlusion & Dark Shadows (High Likelihood / High Impact)\n").bold = True
    p_r1.add_run(
        "Pantry shelves often have items hidden 30-50% behind front packages, and interior cupboard lighting can be below 50 lux, leading to missed detections. "
        "Fallback: We implement CLAHE preprocessing to equalize shadow gradients, train YOLO with Mosaic and CutMix/Random Erasing to force detection on partial features, and apply multi-frame temporal voting across 5 consecutive video frames."
    )

    p_r2 = doc.add_paragraph()
    p_r2.add_run("Risk 2: Classical Baseline Failure on Cylindrical Cans & Overfitting on Small Dataset (Medium Likelihood / Medium Impact)\n").bold = True
    p_r2.add_run(
        "SIFT/ORB assumes planar surfaces and struggles on curved soup cans and wrinkled coffee pouches. Additionally, 250 frames might cause deep neural networks to overfit. "
        "Fallback: For SIFT/ORB, we store multi-angle reference templates (front, side, top) and relax RANSAC inlier thresholds. For YOLO, we freeze the pretrained CSPDarknet backbone during early epochs with aggressive HSV color-jittering, and retain zero-shot YOLO-World as our emergency fallback."
    )

    # 10. Feasibility Spike
    add_h1("10. Feasibility Spike Evidence")
    doc.add_paragraph(
        "The feasibility spike has been verified and executed via 'python src/spike.py'. It confirmed OpenCV image ingestion, Gaussian blurring, Canny edge detection, ORB keypoint extraction, and generated a 20-sample visual contact sheet in outputs/contact_sheet_20_samples.jpg."
    )

    # Save document
    doc.save(output_docx_path)
    print(f"[DOCX] Proposal successfully saved to: {output_docx_path}")

def convert_docx_to_pdf(docx_path="docs/CSST106_Milestone1_Proposal_Stock2Stove.docx", pdf_path="docs/CSST106_Milestone1_Proposal_Stock2Stove.pdf"):
    print(f"[PDF] Attempting Word automation to export {pdf_path}...")
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(docx_path))
        doc.SaveAs(os.path.abspath(pdf_path), FileFormat=17) # 17 = wdFormatPDF
        doc.Close()
        word.Quit()
        print(f"[PDF] Proposal PDF generated successfully: {pdf_path}")
    except Exception as e:
        print(f"[PDF NOTICE] Word automation failed: {e}. DOCX is fully saved.")

if __name__ == "__main__":
    create_proposal_docx()
    convert_docx_to_pdf()
