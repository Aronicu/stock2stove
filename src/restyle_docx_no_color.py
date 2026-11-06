"""
Script to edit the existing DOCX in place, removing all blue/colored styling
and replacing it with clean, professional monochrome/academic styling (black, white, gray).
The only colors will be in actual graphics/images.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

BLACK = RGBColor(0, 0, 0)
DARK_GRAY = RGBColor(50, 50, 50)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    # Remove existing shd if any
    for child in list(tcPr):
        if child.tag.endswith('shd'):
            tcPr.remove(child)
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_borders(cell, top="none", bottom="none", left="none", right="none", 
                     color="CCCCCC", sz="4", left_color="333333", left_sz="18"):
    tcPr = cell._tc.get_or_add_tcPr()
    for child in list(tcPr):
        if child.tag.endswith('tcBorders'):
            tcPr.remove(child)
            
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

def restyle_docx(docx_path="docs/CSST106_Milestone1_Proposal_Stock2Stove.docx"):
    print(f"[EDIT] Loading existing DOCX directly from {docx_path}...")
    doc = docx.Document(docx_path)
    
    # 1. Reset all paragraph text colors to pure black
    for p in doc.paragraphs:
        for r in p.runs:
            r.font.color.rgb = BLACK
            
    # 2. Restyle all tables to clean academic monochrome
    print(f"[EDIT] Restyling {len(doc.tables)} tables to clean monochrome...")
    for t_idx, table in enumerate(doc.tables):
        if t_idx == 0:
            # Metadata Table (Project info & Team: Kaggleing)
            for row in table.rows:
                for cell in row.cells:
                    set_cell_background(cell, "F9F9F9")
                    set_cell_borders(cell, top="single", bottom="single", left="single", right="single", color="CCCCCC", sz="6")
                    for p in cell.paragraphs:
                        for r in p.runs:
                            r.font.color.rgb = BLACK
                            
        elif t_idx == 1:
            # Kitchen Dilemma Callout Box
            cell = table.cell(0, 0)
            set_cell_background(cell, "F5F5F5")
            set_cell_borders(cell, left="single", left_color="222222", left_sz="20", top="none", bottom="none", right="none")
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.color.rgb = BLACK
                    
        else:
            # Standard Content Tables (Objectives, Approach, Data, Flow, Tasks, Timeline)
            # Header Row
            for cell in table.rows[0].cells:
                set_cell_background(cell, "EAEAEA")
                set_cell_borders(cell, top="single", bottom="single", left="none", right="none", color="333333", sz="8")
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.bold = True
                        r.font.color.rgb = BLACK
                        
            # Data Rows
            for r_idx in range(1, len(table.rows)):
                row = table.rows[r_idx]
                bg = "FFFFFF" if r_idx % 2 == 1 else "FAFAFA"
                for cell in row.cells:
                    set_cell_background(cell, bg)
                    set_cell_borders(cell, top="none", bottom="single", left="none", right="none", color="E0E0E0", sz="4")
                    for p in cell.paragraphs:
                        for r in p.runs:
                            r.font.color.rgb = BLACK

    # 3. Add Contact Sheet Graphic if not already present
    contact_sheet_path = os.path.abspath("outputs/contact_sheet_20_samples.jpg")
    if os.path.exists(contact_sheet_path):
        # Find section 10 paragraph
        sec10_found = False
        for p in doc.paragraphs:
            if "10. Feasibility Spike Evidence" in p.text:
                sec10_found = True
                break
                
        # Check if an inline shape/image already exists
        has_image = len(doc.inline_shapes) > 0
        if not has_image:
            print("[EDIT] Adding contact sheet graphic to Section 10...")
            p_img = doc.add_paragraph()
            p_img.paragraph_format.space_before = Pt(8)
            p_img.paragraph_format.space_after = Pt(4)
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run_img = p_img.add_run()
            run_img.add_picture(contact_sheet_path, width=Inches(6.0))
            
            p_caption = doc.add_paragraph()
            p_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_caption.paragraph_format.space_before = Pt(2)
            p_caption.paragraph_format.space_after = Pt(8)
            r_cap = p_caption.add_run("Figure 1: 20-Sample Pantry Shelf Contact Sheet (data/samples/) covering 5 classes under varied lighting, angles, and occlusions.")
            r_cap.font.size = Pt(8.5)
            r_cap.font.italic = True
            r_cap.font.color.rgb = DARK_GRAY

    # Save modified DOCX directly in place
    doc.save(docx_path)
    print(f"[SUCCESS] Directly modified {docx_path} in place with clean monochrome styling.")

def update_pdf(docx_path="docs/CSST106_Milestone1_Proposal_Stock2Stove.docx", pdf_path="docs/CSST106_Milestone1_Proposal_Stock2Stove.pdf"):
    print(f"[PDF] Exporting clean monochrome PDF to {pdf_path}...")
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(docx_path))
        doc.SaveAs(os.path.abspath(pdf_path), FileFormat=17) # 17 = wdFormatPDF
        doc.Close()
        word.Quit()
        print(f"[PDF SUCCESS] Flawless PDF updated: {pdf_path}")
    except Exception as e:
        print(f"[PDF ERROR] {e}")

if __name__ == "__main__":
    restyle_docx()
    update_pdf()
