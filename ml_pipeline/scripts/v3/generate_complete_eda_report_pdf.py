import sys
import os
from pathlib import Path
from fpdf import FPDF

GD = (15, 81, 50)      # Forest Green
GM = (5, 150, 105)     # Emerald Accent
GL = (236, 253, 245)   # Light Mint
BK = (30, 41, 59)      # Charcoal
GY = (100, 116, 139)   # Slate Gray

def S(txt):
    if not isinstance(txt, str):
        txt = str(txt)
    txt = txt.replace("—", "--").replace("–", "-").replace("“", '"').replace("”", '"').replace("’", "'").replace("→", "->")
    return txt.encode("latin-1", "replace").decode("latin-1")

class CompleteEDAReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*GY)
        self.cell(0, 8, S("ZARI.ai -- Comprehensive Exploratory Data Analysis (EDA) Report: New Dataset & 3-Crop Target Dataset"), border=0, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(226, 232, 240)
        self.line(10, 15, 200, 15)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GY)
        self.cell(0, 10, S(f"Page {self.page_no()} of {{nb}}  |  ZARI.ai New Dataset Intelligence & EDA Documentation"), align="C")

    def title_box(self, title, subtitle):
        self.set_fill_color(*GD)
        self.rect(10, 10, 190, 36, "F")
        self.set_y(14)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(255, 255, 255)
        self.cell(190, 8, S(title), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("Helvetica", "B", 9.0)
        self.set_text_color(200, 240, 220)
        self.cell(190, 6, S(subtitle), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_y(52)

    def sec(self, t):
        self.ln(3)
        self.set_font("Helvetica", "B", 11.5)
        self.set_text_color(*GD)
        self.set_x(10)
        self.cell(190, 6, S(t), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*GM)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def sub(self, t):
        self.ln(2)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*GM)
        self.set_x(10)
        self.cell(190, 5, S(t), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, t):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*BK)
        for para in t.strip().split("\n\n"):
            for line in para.strip().splitlines():
                if line.strip():
                    self.set_x(10)
                    self.multi_cell(190, 4.5, S(line.strip()))
            self.ln(1.5)

    def fig(self, img_path, caption, w=170):
        if not os.path.exists(img_path):
            print(f"Warning: Figure missing: {img_path}")
            return
        self.ln(2)
        self.set_x((210 - w) / 2)
        self.image(str(img_path), w=w)
        self.ln(1.5)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(*GY)
        self.cell(190, 4, S(f"Figure: {caption}"), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

def build_pdf():
    pdf = CompleteEDAReportPDF("P", "mm", "A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=15)
    
    fig_dir = Path("/home/hammad/Desktop/project zari - experimental/ml_pipeline/reports/figures")

    # Cover Page
    pdf.add_page()
    pdf.title_box("ZARI.ai New Dataset & 3-Crop EDA Report", "Complete Analysis: New Dataset (v3) Breakdown & 3-Crop Target Production Distributions")

    pdf.sec("1. Executive Summary: New Dataset & Scope Refinement")
    pdf.body("""This technical report presents the complete Exploratory Data Analysis (EDA) graphs and statistical distributions for the NEW dataset (v3) and production 3-crop target dataset in ZARI.ai:

1. New Dataset Analysis (Phase 3 Audit): Comprehensive breakdown of the newly integrated dataset across target nightshade crops.
2. Production 3-Crop Target Dataset: Refined production scope covering Tomato (13 classes), Potato (3 classes), and Pepper (6 classes) for 22 total canonical disease classes (31,071 processed leaf images).""")

    pdf.sec("2. New Dataset EDA (Class Breakdown & Augmentation)")
    pdf.body("Exploratory analysis of the newly ingested dataset (v3):")

    pdf.fig(fig_dir / "new_dataset_class_distribution.png", "New Dataset Class Distribution Across Tomato, Potato, and Pepper Disease Classes", w=165)
    
    pdf.add_page()
    pdf.sec("2. New Dataset EDA (Continued)")
    pdf.fig(fig_dir / "new_dataset_crop_breakdown.png", "New Dataset Volume Share by Target Crop Species (Tomato, Potato, Pepper)", w=165)
    pdf.fig(fig_dir / "new_dataset_augmented_vs_raw.png", "New Dataset Raw Field Samples vs Synthetic Augmentation Ratios", w=165)

    # Genuine 3-Crop Target Dataset EDA Section
    pdf.add_page()
    pdf.sec("3. Production 3-Crop Target Dataset EDA (Tomato, Potato, Pepper)")
    pdf.body("""To maximize diagnostic precision for Pakistan's primary nightshade crops, ZARI.ai refined the target dataset scope exclusively to Tomato (13 classes), Potato (3 classes), and Pepper (6 classes) for a total of 31,071 verified images across 22 classes:""")

    pdf.fig(fig_dir / "3crop_actual_01_class_distribution.png", "Production 3-Crop Target Dataset: 22 Disease Classes (31,071 Total Leaf Images)", w=165)
    pdf.fig(fig_dir / "3crop_actual_02_crop_breakdown.png", "3-Crop Target Volume Breakdown (Tomato: 72.8%, Pepper: 20.0%, Potato: 7.2%)", w=165)

    pdf.add_page()
    pdf.fig(fig_dir / "3crop_actual_03_imbalance_ratios.png", "Class Imbalance Ratio Across 22 Target Classes (Tomato, Potato, Pepper)", w=165)
    pdf.fig(fig_dir / "3crop_actual_04_split_verification.png", "GroupKFold Stratified Split Verification for 22 Target Production Classes (80% / 10% / 10%)", w=165)

    # Save PDF
    out_dir = Path("/home/hammad/Desktop/project zari - experimental/ml_pipeline/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ZARI_FULL_VS_3CROP_EDA_COMPLETE_REPORT.pdf"
    pdf.output(str(out_path))
    print(f"✓ Re-generated ZARI_FULL_VS_3CROP_EDA_COMPLETE_REPORT.pdf with NEW DATASET FIGURES at: {out_path}")

if __name__ == "__main__":
    build_pdf()
