# Stock-to-Stove: Automated Pantry Inventory Detection & Recipe Suggestion

> **CSST 106 — Computer Vision Final Project**  
> **Milestone 1 — Project Proposal and Initial Planning**  
> **Team:** Kaggleing  
> **Members:**  
> - **Ogayon, Aron Gabriel L.** — Team Lead  
> - **Kumar, Allain A.** — Documentation  
> - **Sarmiento, Janvher Lucas J** — Programmer  

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-black.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-gray.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-lightgray.svg)](https://github.com/ultralytics/ultralytics)

---

## 📌 Project Overview
**Stock-to-Stove** solves the common kitchen dilemma: *"A cupboard full of food, but you don't know what to cook."* Staring blankly at pantry shelves leads to decision fatigue, costly takeout delivery, forgotten rear-shelf items, and substantial household grocery waste.

By simply recording a 5-second smartphone video pan across kitchen pantry shelves, computer vision isolates, classifies, and tallies in-stock grocery items—even across dark cupboard shadows and severe occlusions. The verified inventory list is passed directly to a downstream Large Language Model (Gemini / OpenAI API) to immediately suggest 3 actionable, zero-waste recipes using what is currently on the shelf.

- **Primary Dataset:** [images.cv Pantry Image Classification Dataset](https://images.cv/dataset/pantry-image-classification-dataset) (384+ images) combined with custom smartphone video shelf pans.
- **Classical Baseline (Days 4–7):** SIFT / ORB keypoints and descriptors with FLANN / BFMatcher and RANSAC homography.
- **Main Method (Days 8–12):** Fine-tuned single-stage deep detector (**YOLOv8-nano / YOLO11n**) with CSPDarknet backbone and PAN-FPN neck.
- **Downstream LLM Module (Day 13):** Modular zero-shot recipe generator providing user-facing culinary utility from detected bounding boxes.

---

## 📂 Repository Structure
```text
stock2stove/
├── README.md               # Project overview and run guide
├── requirements.txt        # Python dependencies
├── data/
│   └── samples/            # Real benchmark pantry samples (pantry_images_cv_*.jpg)
├── notebooks/              # Jupyter exploration notebooks (baseline SIFT & YOLO EDA)
├── src/
│   ├── spike.py            # Feasibility spike on images.cv dataset (Canny edge, ORB, contact sheet)
│   ├── download_images_cv_pantry.py # Automated scraper/downloader for images.cv dataset
│   └── llm_recipe.py       # Modular LLM recipe generator downstream module
├── outputs/                # Contact sheet (contact_sheet_20_samples.jpg) & spike outputs
└── docs/
    ├── CSST106_Milestone1_Proposal_Stock2Stove.docx # Official Proposal DOCX (Monochrome academic)
    ├── CSST106_Milestone1_Proposal_Stock2Stove.pdf  # Generated PDF submission
    └── PITCH_DEFENSE_CHEATSHEET.md                  # 3-minute in-class pitch prep
```

---

## 🚀 Quickstart & Feasibility Spike

### 1. Installation
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Feasibility Spike on images.cv Dataset
Verifies OpenCV image ingestion, Gaussian blurring, Canny edge detection, ORB keypoint extraction, and generates the 20-sample contact sheet using real images from images.cv:
```powershell
python src/spike.py
```
Outputs saved in `outputs/`:
- `outputs/contact_sheet_20_samples.jpg`
- `outputs/spike_edge_pantry_images_cv_*.jpg`
- `outputs/spike_orb_pantry_images_cv_*.jpg`

### 3. Test Downstream LLM Recipe Integration
```powershell
python src/llm_recipe.py
```

---

## 🏷️ Milestone 1 Tagging
```powershell
git add .
git commit -m "Milestone 1: Incorporate images.cv pantry dataset into feasibility spike and proposal DOCX"
git tag -f cp1-proposal
git push origin cp1-proposal
```
