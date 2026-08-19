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
    txt = txt.replace("—", "--").replace("–", "-").replace("“", '"').replace("”", '"').replace("’", "'").replace("→", "->").replace("≥", ">=").replace("≤", "<=").replace("α", "alpha").replace("τ", "tau")
    return txt.encode("latin-1", "replace").decode("latin-1")

class MasterReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*GY)
        self.cell(0, 8, S("ZARI.ai -- Master Final Technical Defense & Architecture Report"), border=0, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(226, 232, 240)
        self.line(10, 15, 200, 15)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GY)
        self.cell(0, 10, S(f"Page {self.page_no()} of {{nb}}  |  ZARI.ai Official Defense Master Documentation"), align="C")

    def title_box(self, title, subtitle):
        self.set_fill_color(*GD)
        self.rect(10, 10, 190, 38, "F")
        self.set_y(14)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.cell(190, 8, S(title), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(200, 240, 220)
        self.cell(190, 6, S(subtitle), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_y(54)

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

    def bul(self, t):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*BK)
        self.set_x(14)
        self.multi_cell(182, 4.5, S(f"-  {t}"))

    def kv(self, k, v):
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*GD)
        self.set_x(10)
        self.cell(55, 5, S(k) + ":")
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*BK)
        self.multi_cell(135, 5, S(v))

    def fig(self, img_path, caption, w=165):
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

    def th(self, cols, widths):
        self.set_fill_color(*GD)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8)
        self.set_x(10)
        for c, w in zip(cols, widths):
            self.cell(w, 6, S(str(c)), border=1, fill=True, align="C")
        self.ln()

    def tr(self, row, widths, fill=False):
        self.set_fill_color(*GL if fill else (255, 255, 255))
        self.set_text_color(*BK)
        self.set_font("Helvetica", "", 7.5)
        self.set_x(10)
        for c, w in zip(row, widths):
            self.cell(w, 5.5, S(str(c)), border=1, fill=fill, align="L")
        self.ln()

def build_pdf():
    pdf = MasterReportPDF("P", "mm", "A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=15)
    
    fig_dir = Path("/home/hammad/Desktop/project zari - experimental/ml_pipeline/reports/figures")

    # Cover Page
    pdf.add_page()
    pdf.title_box("ZARI.ai Master Final Technical Defense Report", "End-to-End Multilingual Agricultural Advisory System: Vision AI, EDL, SCRC, RAG & LLM Engine")

    pdf.sec("1. Executive System Overview")
    pdf.body("""ZARI.ai is an integrated, evidence-grounded AI agricultural advisory system engineered specifically for Pakistani smallholder farming. The system operates on a single unified web server (http://127.0.0.1:8000), serving both static web interface components and real-time AI diagnostic microservices.""")

    pdf.sub("Master System Specifications")
    pdf.kv("Unified Single Server Port", "http://127.0.0.1:8000")
    pdf.kv("Vision AI Backbone", "2-Stage Hierarchical EfficientNetV2-B2 Suite")
    pdf.kv("Total Model Suite Parameters", "31,109,749 parameters (~31.11 Million Params)")
    pdf.kv("Total Checkpoints Weight Size", "355.73 MB Total File Size on Disk")
    pdf.kv("Input Image Tensor Shape", "RGB (384, 384, 3)")
    pdf.kv("Uncertainty Estimation", "Evidential Deep Learning (Dirichlet Softplus)")
    pdf.kv("Safety Risk Control", "SCRC Threshold tau_SCRC = 0.3175 (Dual Safety Gates)")
    pdf.kv("Multilingual Vector RAG", "ChromaDB (208 Chunks) + MiniLM-L12-v2 (384d)")
    pdf.kv("LLM Advisory Generator", "Meta Llama 3.1 8B Instant on Groq LPUs (~750 tok/s)")
    pdf.kv("Offline Fallback Synthesizer", "Deterministic IPM Synthesizer (100% Offline Edge Mode)")
    pdf.kv("Offline End-to-End Latency", "9.40 ms (3.24ms Vision + 5.32ms RAG + 0.86ms Fallback)")
    pdf.kv("Online End-to-End Latency", "389 ms (3.24ms Vision + 5.32ms RAG + 380ms Groq API)")

    pdf.sec("2. Exploratory Data Analysis (EDA): New Dataset & 3-Crop Scope")
    pdf.body("""Below are the verified Exploratory Data Analysis (EDA) figures for the newly ingested dataset (v3) and production 3-crop target dataset:""")

    pdf.fig(fig_dir / "new_dataset_class_distribution.png", "New Dataset (v3) Class Volume Distribution Across Target Nightshade Diseases", w=165)
    
    pdf.add_page()
    pdf.sec("2. Exploratory Data Analysis (Continued)")
    pdf.fig(fig_dir / "new_dataset_crop_breakdown.png", "New Dataset Volume Share by Target Crop Species (Tomato, Potato, Pepper)", w=165)
    pdf.fig(fig_dir / "new_dataset_augmented_vs_raw.png", "New Dataset Raw Field Samples vs Synthetic Augmentation Ratios", w=165)

    pdf.add_page()
    pdf.sub("Production 3-Crop Target Dataset Distributions (31,071 Processed Images)")
    pdf.fig(fig_dir / "3crop_actual_01_class_distribution.png", "Production 3-Crop Target Dataset: 22 Disease Classes (31,071 Total Leaf Images)", w=165)
    pdf.fig(fig_dir / "3crop_actual_02_crop_breakdown.png", "Production Crop Share (Tomato: 72.8%, Pepper: 20.0%, Potato: 7.2%)", w=165)

    pdf.add_page()
    pdf.fig(fig_dir / "3crop_actual_03_imbalance_ratios.png", "Class Imbalance Multipliers Across 22 Production Disease Classes", w=165)
    pdf.fig(fig_dir / "3crop_actual_04_split_verification.png", "GroupKFold Stratified Train / Val / Test Split Breakdown (80% / 10% / 10%)", w=165)

    pdf.add_page()
    pdf.sec("3. Vision AI Model Suite (EfficientNetV2-B2)")
    pdf.body("""ZARI.ai utilizes 4 distinct EfficientNetV2-B2 PyTorch models in a 2-stage hierarchical routing architecture. Below is the exact empirical checkpoint registry:""")

    pdf.th(["Model Component", "Architecture", "Output Head", "Exact Params", "Checkpoint Path", "Metric / Accuracy"], [35, 30, 35, 25, 40, 25])
    pdf.tr(["Model A (Router)", "EfficientNetV2-B2", "3 Crop Classes", "7,772,858", "best_model_a_efficientnetv2_b2.pth", "99.48% Accuracy"], [35, 30, 35, 25, 40, 25], fill=True)
    pdf.tr(["Model B (Tomato)", "EDLEfficientNetB2", "13 Diseases", "7,786,948", "best_model_b_tomato.pth", "0.9787 Macro F1"], [35, 30, 35, 25, 40, 25])
    pdf.tr(["Model B (Potato)", "EDLEfficientNetB2", "3 Diseases", "7,772,858", "best_model_b_potato.pth", "0.9718 Macro F1"], [35, 30, 35, 25, 40, 25], fill=True)
    pdf.tr(["Model B (Pepper)", "EDLEfficientNetB2", "6 Diseases", "7,777,085", "best_model_b_pepper.pth", "0.9963 Macro F1"], [35, 30, 35, 25, 40, 25])
    pdf.tr(["TOTAL SYSTEM", "4-Model Suite", "22 Classes Total", "31,109,749", "355.73 MB Total Weights", "99.1% Reliability"], [35, 30, 35, 25, 40, 25], fill=True)

    pdf.sec("4. Evidential Deep Learning (EDL) & SCRC Safety Gates")
    pdf.body("""Evidential Deep Learning replaces standard Softmax with Dirichlet parameters to compute evidential uncertainty u = K / S. The system enforces statistical risk control to limit false acceptances to <= 5.0%:""")
    pdf.kv("Calibrated Safety Threshold", "tau_SCRC = 0.3175 (scrc_threshold.json)")
    pdf.kv("Model A Router Confidence Gate", "p_crop >= 0.80 (Rejects non-target crops like Corn, Wheat, Cotton)")
    pdf.kv("Model B Classifier Confidence Gate", "p_disease >= 0.70")
    pdf.kv("Dual Safety Gate Rule", "Accept if (u <= 0.3175 AND p_crop >= 0.80 AND p_disease >= 0.70) else REJECT")

    pdf.sec("5. Multilingual Vector RAG System Architecture")
    pdf.body("""Ground-truth IPM advisory chunks are stored locally in ChromaDB (ml_pipeline/rag/chroma_db/) indexed via HNSW graphs using sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384d):""")
    pdf.kv("Total RAG Knowledge Chunks", "208 Chunks (26 Knowledge Records x 8 IPM Sections)")
    pdf.kv("Target Taxonomy Scope", "22 Production Disease Classes + 4 Regional Quarantine Records")
    pdf.kv("IPM Hierarchy", "Cultural Control -> Biological Control -> Chemical Active Ingredients")

    pdf.sec("6. LLM Advisory Engine & Groq Acceleration")
    pdf.body("""Advisory text generation is powered by Meta Llama 3.1 8B Instant (8 billion params) deployed on Groq LPUs (~750 tok/s), with prompt rules strictly prohibiting invented dosages or PHI days, plus an offline fallback synthesizer.""")

    # Save PDF
    out_dir = Path("/home/hammad/Desktop/project zari - experimental/ml_pipeline/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ZARI_MASTER_FINAL_TECHNICAL_DEFENSE_REPORT.pdf"
    pdf.output(str(out_path))
    print(f"✓ Created ZARI_MASTER_FINAL_TECHNICAL_DEFENSE_REPORT.pdf successfully at: {out_path}")

if __name__ == "__main__":
    build_pdf()
