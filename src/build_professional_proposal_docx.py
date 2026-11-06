"""
Generate professional executive-grade DOCX and PDF proposal for Stock-to-Stove
CSST 106 Computer Vision Final Project - Milestone 1
Team: Kaggleing
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

NAVY = RGBColor(30, 58, 138)       # #1E3A8A
SLATE = RGBColor(31, 41, 55)       # #1F2937
STEEL = RGBColor(37, 99, 235)      # #2563EB
MUTED = RGBColor(100, 116, 139)    # #64748B
WHITE = RGBColor(255, 255, 255)

HEX_NAVY = "1E3A8A"
HEX_LIGHT_BG = "F8FAFC"
HEX_CALLOUT_BG = "F1F5F9"
HEX_BORDER = "E2E8F0"
HEX_ACCENT = "2563EB"

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

def set_cell_borders(cell, top="none", bottom="none", left="none", right="none", 
                     color="E2E8F0", sz="4", left_color="1E3A8A", left_sz="24"):
    tcPr = cell._tc.get_or_add_tcPr()
    left_str = f'<w:left w:val="{left}" w:sz="{left_sz}" w:space="0" w:color="{left_color}"/>' if left != "none" else '<w:left w:val="none"/>'
    top_str = f'<w:top w:val="{top}" w:sz="{sz}" w:space="0" w:color="{color}"/>' if top != "none" else '<w:top w:val="none"/>'
    bottom_str = f'<w:bottom w:val="{bottom}" w:sz="{sz}" w:space="0" w:color="{color}"/>' if bottom != "none" else '<w:bottom w:val="none"/>'
    right_str = f'<w:right w:val="{right}" w:sz="{sz}" w:space="0" w:color="{color}"/>' if right != "none" else '<w:right w:val="none"/>'
    
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'{top_str}{left_str}{bottom_str}{right_str}'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

def add_callout(doc, quote_text, header="THE CORE DILEMMA: \"WHAT DO I COOK TONIGHT?\""):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    set_cell_background(cell, HEX_CALLOUT_BG)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    set_cell_borders(cell, left="single", left_color=HEX_NAVY, left_sz="24", color="none")
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    
    r_hdr = p.add_run(header + "\n")
    r_hdr.font.size = Pt(9.5)
    r_hdr.font.bold = True
    r_hdr.font.color.rgb = NAVY
    
    r_body = p.add_run(quote_text)
    r_body.font.size = Pt(10)
    r_body.font.italic = True
    r_body.font.color.rgb = SLATE
    
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

def style_table(table, col_widths, headers, data_rows, header_bg=HEX_NAVY):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Header Row
    hdr_cells = table.rows[0].cells
    for idx, (title, w) in enumerate(zip(headers, col_widths)):
        hdr_cells[idx].width = Inches(w)
        set_cell_background(hdr_cells[idx], header_bg)
        set_cell_margins(hdr_cells[idx], top=90, bottom=90, left=110, right=110)
        set_cell_borders(hdr_cells[idx], top="single", bottom="single", color="1E3A8A", sz="8")
        
        p = hdr_cells[idx].paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(title)
        r.font.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = WHITE
        
    # Data Rows
    for r_idx, row_values in enumerate(data_rows, start=1):
        row_cells = table.rows[r_idx].cells
        bg_color = WHITE if r_idx % 2 == 1 else HEX_LIGHT_BG
        for c_idx, (val, w) in enumerate(zip(row_values, col_widths)):
            row_cells[c_idx].width = Inches(w)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=70, bottom=70, left=110, right=110)
            set_cell_borders(row_cells[c_idx], bottom="single", color=HEX_BORDER, sz="4")
            
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            r.font.color.rgb = SLATE

def generate_docx(output_path="docs/CSST106_Milestone1_Proposal_Stock2Stove.docx"):
    doc = docx.Document()
    
    # 0.75-inch margins for professional academic layout
    for s in doc.sections:
        s.top_margin = Inches(0.75)
        s.bottom_margin = Inches(0.75)
        s.left_margin = Inches(0.75)
        s.right_margin = Inches(0.75)
        
    # Default Normal Style
    normal = doc.styles['Normal']
    normal.font.name = 'Calibri'
    normal.font.size = Pt(10)
    normal.font.color.rgb = SLATE
    
    # ------------------ TOP HEADER BANNER ------------------
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(0)
    p_meta.paragraph_format.space_after = Pt(2)
    r_course = p_meta.add_run("CSST 106 — COMPUTER VISION FINAL PROJECT  |  MILESTONE 1 — PROJECT PROPOSAL & INITIAL PLANNING")
    r_course.font.size = Pt(8.5)
    r_course.font.bold = True
    r_course.font.color.rgb = MUTED
    
    # Project Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(2)
    r_t = p_title.add_run("Stock-to-Stove: Automated Pantry Shelf Tracking & Recipe Generation Pipeline")
    r_t.font.size = Pt(20)
    r_t.font.bold = True
    r_t.font.color.rgb = NAVY
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(8)
    r_sub = p_sub.add_run("Fine-Tuned YOLOv8 Object Detection on Shelf Video with Downstream Generative LLM Culinary Synthesis")
    r_sub.font.size = Pt(11)
    r_sub.font.italic = True
    r_sub.font.color.rgb = STEEL
    
    # ------------------ TEAM & PROJECT METADATA CARD ------------------
    # Clean 2-column card containing Project Details and Team: Kaggleing with Members
    meta_table = doc.add_table(rows=1, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    col_w = [3.2, 3.8]
    c_left = meta_table.cell(0, 0)
    c_right = meta_table.cell(0, 1)
    
    c_left.width = Inches(col_w[0])
    c_right.width = Inches(col_w[1])
    
    for c in [c_left, c_right]:
        set_cell_background(c, HEX_LIGHT_BG)
        set_cell_margins(c, top=80, bottom=80, left=130, right=130)
        set_cell_borders(c, top="single", bottom="single", left="single", right="single", color=HEX_BORDER, sz="6")
        
    # Left Column: Project Metadata
    p_left = c_left.paragraphs[0]
    p_left.paragraph_format.space_before = Pt(0)
    p_left.paragraph_format.space_after = Pt(2)
    p_left.paragraph_format.line_spacing = 1.15
    
    r_l1_lbl = p_left.add_run("Project Code: ")
    r_l1_lbl.font.bold = True
    r_l1_lbl.font.size = Pt(8.5)
    r_l1_lbl.font.color.rgb = MUTED
    r_l1_v = p_left.add_run("stock2stove\n")
    r_l1_v.font.bold = True
    r_l1_v.font.size = Pt(8.5)
    r_l1_v.font.color.rgb = NAVY
    
    r_l2_lbl = p_left.add_run("Course: ")
    r_l2_lbl.font.bold = True
    r_l2_lbl.font.size = Pt(8.5)
    r_l2_lbl.font.color.rgb = MUTED
    r_l2_v = p_left.add_run("CSST 106 — Computer Vision\n")
    r_l2_v.font.size = Pt(8.5)
    r_l2_v.font.color.rgb = SLATE
    
    r_l3_lbl = p_left.add_run("Milestone: ")
    r_l3_lbl.font.bold = True
    r_l3_lbl.font.size = Pt(8.5)
    r_l3_lbl.font.color.rgb = MUTED
    r_l3_v = p_left.add_run("Milestone 1 — Project Proposal")
    r_l3_v.font.size = Pt(8.5)
    r_l3_v.font.color.rgb = SLATE

    # Right Column: Team: Kaggleing & Members
    p_right = c_right.paragraphs[0]
    p_right.paragraph_format.space_before = Pt(0)
    p_right.paragraph_format.space_after = Pt(2)
    p_right.paragraph_format.line_spacing = 1.15
    
    r_team_lbl = p_right.add_run("Team: ")
    r_team_lbl.font.bold = True
    r_team_lbl.font.size = Pt(9)
    r_team_lbl.font.color.rgb = MUTED
    r_team_val = p_right.add_run("Kaggleing\n")
    r_team_val.font.bold = True
    r_team_val.font.size = Pt(9.5)
    r_team_val.font.color.rgb = NAVY
    
    r_mem_hdr = p_right.add_run("Members:\n")
    r_mem_hdr.font.bold = True
    r_mem_hdr.font.size = Pt(8.5)
    r_mem_hdr.font.color.rgb = MUTED
    
    members = [
        ("• Ogayon, Aron Gabriel L.", " — Team Lead"),
        ("• Kumar, Allain A.", " — Documentation"),
        ("• Sarmiento, Janvher Lucas J", " — Programmer")
    ]
    for idx, (m_name, m_role) in enumerate(members):
        newline = "\n" if idx < len(members) - 1 else ""
        r_name = p_right.add_run(m_name)
        r_name.font.size = Pt(8.5)
        r_name.font.bold = True
        r_name.font.color.rgb = SLATE
        r_role = p_right.add_run(m_role + newline)
        r_role.font.size = Pt(8.5)
        r_role.font.color.rgb = STEEL

    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    
    # Helper functions for clean section headers
    def add_sec_h1(num_str, title_str):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(3)
        h.paragraph_format.keep_with_next = True
        
        r_n = h.add_run(f"{num_str}. ")
        r_n.font.size = Pt(13)
        r_n.font.bold = True
        r_n.font.color.rgb = STEEL
        
        r_t = h.add_run(title_str)
        r_t.font.size = Pt(13)
        r_t.font.bold = True
        r_t.font.color.rgb = NAVY
        return h

    def add_sec_h2(subtitle_str):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(2)
        h.paragraph_format.keep_with_next = True
        
        r = h.add_run(subtitle_str)
        r.font.size = Pt(10.5)
        r.font.bold = True
        r.font.color.rgb = STEEL
        return h

    def add_p(text, bold_prefix=None, space_after=3):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.bold = True
            r_pre.font.color.rgb = NAVY
        p.add_run(text)
        return p

    # ------------------ 1. TITLE ------------------
    add_sec_h1("1", "Title")
    add_p(
        "\"Stock-to-Stove: Automated Pantry Shelf Inventory Tracking via Fine-Tuned YOLOv8 Object Detection with Downstream Generative LLM Recipe Suggestion\"",
        bold_prefix="System Name & Formal Title: "
    )

    # ------------------ 2. PROBLEM STATEMENT ------------------
    add_sec_h1("2", "Problem Statement")
    
    add_callout(
        doc,
        "\"It is 6:30 PM on a Tuesday evening. You open the kitchen cupboard, tired from work, and stare blankly "
        "at three rows of cans, half-empty pasta cartons, and forgotten jars. You have a cupboard full of food, "
        "yet you have absolutely no idea what to cook. You pull out your phone, spend $35 on food delivery, "
        "and those shelf ingredients sit in the dark for another month until they quietly expire.\"",
        header="THE REAL KITCHEN DILEMMA: \"A CUPBOARD FULL OF FOOD, BUT NOTHING TO COOK\""
    )

    add_sec_h2("Who Has the Problem?")
    add_p(
        "This dilemma affects millions of everyday home cooks, busy working professionals, dual-income households, "
        "and college students. Managing domestic kitchen inventory is notoriously chaotic, fragmented, and universally neglected."
    )

    add_sec_h2("What They Do Today (and Why Current Solutions Fail)")
    add_p(
        "1. Decision Fatigue and the 'Pantry Stare': People rarely fail to cook because they lack ingredients; they fail "
        "because they cannot mentally synthesize disparate raw items into a cohesive dinner. Staring into a crowded cabinet "
        "creates immediate cognitive overload.",
        bold_prefix="• "
    )
    add_p(
        "2. Manual Audits and Inventory Friction: Dedicated inventory apps require users to manually type item names or scan "
        "individual barcodes every time they buy or use an item. Within three days, 95% of users abandon the chore due to friction.",
        bold_prefix="• "
    )
    add_p(
        "3. Accidental Duplication and Massive Food Waste: Items pushed to the rear shadows are forgotten. Consumers buy "
        "duplicate items (e.g., purchasing a fourth can of crushed tomatoes while three already sit hidden). Simultaneously, "
        "forgotten items expire silently, driving part of the estimated $1,500 in grocery waste generated annually per household.",
        bold_prefix="• "
    )

    add_sec_h2("Why Computer Vision Helps")
    add_p(
        "Computer vision bridges the physical shelf directly to actionable meal inspiration with zero manual typing:",
        bold_prefix="The Solution: "
    )
    add_p(
        "• Frictionless 5-Second Video Capture: Instead of barcode scanning, a user simply records a continuous phone video "
        "pan across their pantry shelves. Vision extracts multi-object detections simultaneously."
    )
    add_p(
        "• Robustness to Packaging Chaos: Modern deep object detectors recognize branded items despite viewpoint angles (30°-60°), "
        "severe shelf shadows (<100 lux), and partial occlusions where items sit in front of one another."
    )
    add_p(
        "• Direct Bridge from Pixels to the Stove: Vision converts raw pixels into a clean, verified inventory list "
        "(['pasta', 'canned tomatoes', 'garlic', 'tuna']). This array immediately feeds a downstream Large Language Model (LLM) "
        "that instantly answers the burning question: 'Here are 3 delicious, 15-minute recipes you can make right now with what is on your shelf.'"
    )

    # ------------------ 3. OBJECTIVES & EXPECTED OUTCOMES ------------------
    add_sec_h1("3", "Objectives and Expected Outcomes")
    add_p("The project's success is governed by concrete, measurable quantitative engineering targets evaluated on held-out test data:")

    obj_headers = ["Metric / Benchmark Area", "Measurable Engineering Target", "Verification Protocol"]
    obj_widths = [1.8, 2.7, 2.5]
    obj_rows = [
        ["Detection Accuracy", "mAP@0.5 >= 0.85 and mAP@0.5:0.95 >= 0.60 across all 5 target pantry classes", "Evaluated on held-out test split (37 unseen shelf frames, ~60 instances)"],
        ["Inference Throughput", ">= 15 FPS on modern laptop CPU (Intel Core i5/i7) and >= 45 FPS on CUDA GPU", "Timed over 100 consecutive video frames in Python runtime"],
        ["Occlusion / Rotation Delta", ">= +30% higher Recall and macro-F1 over classical SIFT/ORB baseline", "Tested on challenging subset (items rotated >30 deg or occluded 20-50%)"],
        ["Downstream LLM Latency", "End-to-end recipe response delivered in < 2.5 seconds total latency", "Wall-clock API round-trip measurement using Gemini 2.5 Flash / OpenAI"]
    ]
    t_obj = doc.add_table(rows=len(obj_rows)+1, cols=3)
    style_table(t_obj, obj_widths, obj_headers, obj_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    add_p(
        "If custom deep learning fine-tuning experiences unexpected divergence or severe annotation overfitting, we will "
        "deploy an ensemble combining pretrained zero-shot YOLO-World / YOLOv8-COCO (for general grocery containers: 'can', 'box', 'bottle') "
        "alongside multi-template ORB homography on cropped brand packaging. The downstream deduplication and LLM recipe engine "
        "will remain 100% operational, guaranteeing a functional live demo from video scan to recipe suggestion.",
        bold_prefix="Minimum Deliverable Version (Safety Net / Fallback): "
    )

    # ------------------ 4. PROPOSED APPROACH & ALGORITHMS ------------------
    add_sec_h1("4", "Proposed Approach and Algorithms")
    add_p(
        "The architecture enforces strict Separation of Concerns. Computer Vision performs all heavy perception tasks, "
        "while the downstream LLM functions as a modular 15-line text-in, text-out utility:"
    )

    app_headers = ["Stage", "Algorithm & Implementation", "Technical Rationale (Why)"]
    app_widths = [1.4, 2.8, 2.8]
    app_rows = [
        ["1. Acquisition", "Smartphone video (1080p, 30 FPS) panning pantry shelves; subsampled at 1-2 FPS.", "Captures realistic real-world pantry scenes with natural motion blur, lighting shifts, and shelf depth."],
        ["2. Preprocessing", "Luminance CLAHE normalization + letterbox resizing to 640x640.", "CLAHE equalizes dark interior shelf shadows (<100 lux) without blowing out bright front packaging."],
        ["3. Baseline (Days 4-7)", "Classical SIFT / ORB feature matching + FLANN/BFMatcher + RANSAC homography.", "Establishes a classical comparison point in days without neural training data, isolating planar limits."],
        ["4. Main Method (Days 8-12)", "Fine-tuned YOLOv8n / YOLO11n single-stage detector (CSPDarknet + PAN-FPN).", "Offers top-tier CPU inference speed with superior multi-scale feature representations that resist occlusion."],
        ["5. Post-Processing", "Non-Maximum Suppression (IoU >= 0.45, conf >= 0.50) + multi-frame temporal voting.", "Eliminates false positives and bounding box flicker across sequential panning frames."],
        ["6. Downstream LLM (Day 13)", "Deduplicated class array injected into constrained prompt -> Gemini Flash API.", "Bridges vision to immediate user utility (<2.5s) without consuming CV development time."]
    ]
    t_app = doc.add_table(rows=len(app_rows)+1, cols=3)
    style_table(t_app, app_widths, app_headers, app_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ------------------ 5. DATA SPECIFICATION ------------------
    add_sec_h1("5", "Data")
    add_p(
        "Primary smartphone video captured by team members in private domestic pantries under varied kitchen conditions. "
        "Sample frames are stored in data/samples/ and the full dataset is hosted on Roboflow Universe under the Creative Commons "
        "Attribution 4.0 International (CC-BY 4.0) academic license.",
        bold_prefix="Source & Repository: "
    )

    data_headers = ["Class Identifier", "Target Grocery Item", "Key Visual & Packaging Features", "Occlusion Profile"]
    data_widths = [1.4, 1.8, 2.3, 1.5]
    data_rows = [
        ["pasta_box", "Barilla Penne (16 oz)", "Dark blue rectangular cardboard box, white text, clear cellophane window", "Front or rear row, 0-40% occluded"],
        ["tomato_can", "Hunt's Crushed Tomatoes", "Cylindrical reflective tin, bright red label, tomato graphic", "Curved surface, 0-50% occluded"],
        ["cereal_box", "Cheerios / Honey Nut", "Bright yellow tall rectangular carton, large bowl illustration", "High visibility, corner occlusions"],
        ["coffee_bag", "Starbucks / Folgers Ground", "Dark brown/green deformable foil pouch, specular reflections", "Wrinkled surface, 0-30% occluded"],
        ["tuna_can", "StarKist Chunk Light", "Short squat silver cylinder, blue/white label, reflective top lid", "Often stacked or behind cans"]
    ]
    t_data = doc.add_table(rows=len(data_rows)+1, cols=4)
    style_table(t_data, data_widths, data_headers, data_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    add_p(
        "• Dataset Size: 250 annotated frames (>= 50 labeled instances per class, ~400 total annotations).\n"
        "• Variation Conditions: Direct kitchen overhead lighting, deep interior cupboard shadows (<100 lux), angled sweeps (30°-60°), and partial occlusions (20%-50%).\n"
        "• Split Strategy: 70% Training (175 frames), 15% Validation (38 frames), 15% Held-Out Test (37 frames), stratified across lighting and occlusion levels."
    )

    # ------------------ 6. ARCHITECTURE & WORKFLOW DIAGRAM ------------------
    add_sec_h1("6", "Architecture and Workflow Diagram")
    add_p(
        "The architecture outlines the complete progression from raw shelf video to zero-waste recipe output:"
    )

    flow_headers = ["Pipeline Stage", "Module Inputs", "Processing Engine", "Stage Outputs"]
    flow_widths = [1.3, 1.8, 2.2, 1.7]
    flow_rows = [
        ["1. Ingestion", "Pantry Video (1080p, 30 FPS)", "Frame Subsampler (1-2 FPS) + CLAHE", "640x640 Normalized Frames"],
        ["2. Baseline", "Reference Template Crops + Frame", "ORB/SIFT + FLANN + RANSAC Homography", "Bounding Quads (Planar)"],
        ["3. Main Method", "Normalized Shelf Frames", "Fine-Tuned YOLOv8n (CSPDarknet + PAN-FPN)", "Anchor-Free Multi-Class Boxes"],
        ["4. Post-Filter", "Raw Detections across Video", "NMS (IoU >= 0.45) + Multi-Frame Voting", "Deduplicated Inventory Array"],
        ["5. LLM Synthesis", "['pasta', 'tomatoes', 'tuna']", "Prompt Builder + Gemini 2.5 Flash API", "3 Zero-Waste Recipe Cards"]
    ]
    t_flow = doc.add_table(rows=len(flow_rows)+1, cols=4)
    style_table(t_flow, flow_widths, flow_headers, flow_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ------------------ 7. TASK ASSIGNMENTS ------------------
    add_sec_h1("7", "Task Assignments")
    add_p("To comply with CSST 106 requirements, every team member owns substantive code modules and testing suites:")

    task_headers = ["Module & Scope", "Responsible Owner", "Code Deliverables & Artifacts"]
    task_widths = [2.2, 2.0, 2.8]
    task_rows = [
        ["Module 1: Data Acquisition & Preprocessing", "Kumar, Allain A.\n(Documentation & Data Lead)", "src/data_prep.py, src/augment.py, Roboflow dataset config & validation splits"],
        ["Module 2: Classical Baseline (SIFT/ORB)", "Sarmiento, Janvher Lucas J\n(Programmer)", "src/baseline_sift.py, src/baseline_orb.py, feature evaluation notebook"],
        ["Module 3: YOLOv8 Training & Optimization", "Ogayon, Aron Gabriel L.\n(Team Lead)", "src/train_yolo.py, src/evaluate.py, fine-tuned weights & benchmark curves"],
        ["Module 4: Downstream LLM & Demo Application", "Sarmiento, Janvher Lucas J\n(Programmer)", "src/llm_recipe.py, src/app_demo.py, terminal & live video stream UI"],
        ["Module 5: Benchmarking, Documentation & Reports", "Kumar, Allain A. & All Members", "outputs/benchmark_results.csv, technical report, pitch & defense deck"]
    ]
    t_task = doc.add_table(rows=len(task_rows)+1, cols=3)
    style_table(t_task, task_widths, task_headers, task_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ------------------ 8. TIMELINE ALIGNED TO CHECKPOINTS ------------------
    add_sec_h1("8", "Timeline Aligned to Course Checkpoints")
    add_p("The project timeline is strictly aligned with the five CSST 106 syllabus checkpoints:")

    time_headers = ["Checkpoint", "Hand-In Deadline", "In-Class Check", "Technical Deliverables & Focus"]
    time_widths = [1.6, 1.5, 1.2, 2.7]
    time_rows = [
        ["CP1: Proposal & Plan", "Thu Oct 15, 11:30 PM", "Fri Oct 16", "Proposal PDF/DOCX, repo scaffold, working spike, 20-sample contact sheet (tag: cp1-proposal)"],
        ["CP2: 50% Progress Check", "Thu Oct 22, 11:30 PM", "Fri Oct 23", "250 labeled frames complete on Roboflow; SIFT/ORB baseline running with initial precision/recall (tag: cp2-50)"],
        ["CP3: Core Implementation", "Thu Oct 29, 11:30 PM", "Fri Oct 30", "YOLOv8 fine-tuned (50 epochs); validation mAP@0.5 >= 0.80; baseline vs YOLO comparison curves (tag: cp3-core)"],
        ["CP4: 80% Progress Check", "Thu Nov 12, 11:30 PM", "Fri Nov 13", "Occlusion stress tests completed; temporal consistency voting filter active; LLM recipe API connected (tag: cp4-80)"],
        ["CP5: Final Presentation", "Thu Nov 19, 11:30 PM", "From Fri Nov 20", "End-to-end live video pipeline demo; code cleanup, performance charts, final report (tag: v1.0)"],
        ["CP6: Technical Defense", "Thu Nov 19, 11:30 PM", "Scheduled Slot", "Individual member defense on model architecture, failure cases, loss curves, and code ownership"]
    ]
    t_time = doc.add_table(rows=len(time_rows)+1, cols=4)
    style_table(t_time, time_widths, time_headers, time_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ------------------ 9. RISKS & FALLBACK STRATEGIES ------------------
    add_sec_h1("9", "Risks and Fallback Strategies")
    
    add_p(
        "Pantry shelves often have rear items hidden 30-50% behind front cans, and interior cupboard lighting can drop below 50 lux, causing false negatives.\n"
        "Fallback: 1) Apply CLAHE luminance equalization to illuminate shadow gradients. 2) Train YOLO with Mosaic and CutMix/Random Erasing to force detection on partial packaging cues. 3) Enforce 5-frame temporal voting in video pans so transient occlusions do not drop the item.",
        bold_prefix="Risk 1: Severe Shelf Occlusion & Cupboard Shadows (High Likelihood / High Impact) — "
    )

    add_p(
        "SIFT/ORB relies on 2D planar homography, failing on curved cans and wrinkled coffee pouches. Additionally, 250 frames might induce overfitting in deep networks.\n"
        "Fallback: For SIFT/ORB, store multi-angle reference templates (front, side, top) and relax RANSAC inlier thresholds. For YOLO, freeze the CSPDarknet backbone during initial training with heavy HSV augmentations. If custom fine-tuning stalls, fall back to pretrained open-vocabulary YOLO-World.",
        bold_prefix="Risk 2: Classical Baseline Failure on Cylindrical Cans & Neural Overfitting (Medium Likelihood / Medium Impact) — "
    )

    add_p(
        "Downstream API request fails during in-class live presentation due to classroom Wi-Fi latency, quota limit, or missing keys.\n"
        "Fallback: Implement strict local offline caching in src/llm_recipe.py. If API call exceeds 2.5s or fails, a deterministic local recipe lookup dictionary automatically outputs matched recipes without breaking the demo.",
        bold_prefix="Risk 3: Downstream LLM API Rate Limits or Presentation Network Failure (Low Likelihood / High Impact) — "
    )

    # ------------------ 10. FEASIBILITY SPIKE EVIDENCE ------------------
    add_sec_h1("10", "Feasibility Spike Evidence")
    add_p(
        "The feasibility spike mandated in Section 3 of the syllabus was executed successfully via 'python src/spike.py'. "
        "It verified OpenCV image ingestion, Gaussian blurring, Canny edge detection, ORB keypoint extraction, and rendered the "
        "20-sample visual contact sheet in outputs/contact_sheet_20_samples.jpg.",
        bold_prefix="Verified Implementation: "
    )

    doc.save(output_path)
    print(f"[SUCCESS] Professional Proposal saved to: {output_path}")

def convert_to_pdf(docx_path="docs/CSST106_Milestone1_Proposal_Stock2Stove.docx", pdf_path="docs/CSST106_Milestone1_Proposal_Stock2Stove.pdf"):
    print(f"[PDF] Converting {docx_path} to {pdf_path} via Word automation...")
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(docx_path))
        doc.SaveAs(os.path.abspath(pdf_path), FileFormat=17) # 17 = wdFormatPDF
        doc.Close()
        word.Quit()
        print(f"[PDF] Flawless PDF generated: {pdf_path}")
    except Exception as e:
        print(f"[PDF ERROR] {e}")

if __name__ == "__main__":
    generate_docx()
    convert_to_pdf()
