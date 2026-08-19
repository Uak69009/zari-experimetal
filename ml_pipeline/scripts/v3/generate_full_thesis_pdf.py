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

class ThesisReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*GY)
        self.cell(0, 8, S("ZARI.ai -- Master Thesis & Technical Defense Technical Specification"), border=0, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(226, 232, 240)
        self.line(10, 15, 200, 15)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GY)
        self.cell(0, 10, S(f"Page {self.page_no()} of {{nb}}  |  ZARI.ai Master Thesis Technical Defense"), align="C")

    def title_box(self, title, subtitle):
        self.set_fill_color(*GD)
        self.rect(10, 10, 190, 40, "F")
        self.set_y(14)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.cell(190, 8, S(title), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(200, 240, 220)
        self.cell(190, 6, S(subtitle), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_y(56)

    def sec(self, t):
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*GD)
        self.set_x(10)
        self.cell(190, 6, S(t), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*GM)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def sub(self, t):
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
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
    pdf = ThesisReportPDF("P", "mm", "A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=15)
    
    fig_dir = Path("/home/hammad/Desktop/project zari - experimental/ml_pipeline/reports/figures")

    # Cover Page
    pdf.add_page()
    pdf.title_box("ZARI.ai Master Thesis & Technical Defense Report", "End-to-End Multilingual Agricultural Disease Advisory System with Evidential Deep Learning, SCRC Safety Control & Vector RAG Engine")

    pdf.sec("1. Abstract & System Specification")
    pdf.body("""ZARI.ai is a production-grade, uncertainty-aware agricultural disease advisory platform engineered for Pakistan's nightshade crops (Tomato, Potato, Pepper). The system combines a 2-stage hierarchical EfficientNetV2-B2 vision suite, Dirichlet Evidential Deep Learning (EDL), Selective Classification Risk Control (SCRC), a multilingual ChromaDB vector RAG engine, and a Meta Llama 3.1 8B LLM generator.""")

    pdf.sub("Master Architectural Registry")
    pdf.kv("Single Server Deployment Port", "http://127.0.0.1:8000 (Unified Web Interface & FastAPI Backend)")
    pdf.kv("Vision Backbone", "2-Stage Hierarchical EfficientNetV2-B2 (4 PyTorch Models)")
    pdf.kv("Total Vision Parameters", "31,109,749 parameters (~31.11 Million Params)")
    pdf.kv("Total Model Suite Weight File Size", "355.73 MB Total Weights File Size on Disk")
    pdf.kv("Input Image Tensor Resolution", "RGB (384, 384, 3)")
    pdf.kv("Uncertainty Formulation", "Dirichlet Softplus Evidential Deep Learning (u = K / S)")
    pdf.kv("Calibrated Safety Threshold", "tau_SCRC = 0.3175 (Dual Safety Gates: p_crop >= 0.80, p_disease >= 0.70)")
    pdf.kv("Multilingual RAG Store", "ChromaDB (208 Chunks) + MiniLM-L12-v2 Embedder (384d)")
    pdf.kv("Primary LLM Advisory Generator", "Meta Llama 3.1 8B Instant on Groq LPUs (~750 tok/s)")
    pdf.kv("Offline Fallback Synthesizer", "Deterministic Evidence Synthesizer (100% Edge Availability)")
    pdf.kv("Offline Latency", "9.40 ms (3.24ms Vision + 5.32ms RAG + 0.86ms Fallback)")
    pdf.kv("Online Latency", "389 ms (3.24ms Vision + 5.32ms RAG + 380ms Groq API)")

    # Section 2: EDA
    pdf.add_page()
    pdf.sec("2. Exploratory Data Analysis: New Dataset (v3) & 3-Crop Target Distributions")
    pdf.body("""The figures below detail the Exploratory Data Analysis (EDA) for the newly ingested dataset (v3) and the production 3-crop target dataset (31,071 leaf images across 22 classes):""")

    pdf.fig(fig_dir / "new_dataset_class_distribution.png", "New Dataset (v3) Class Distribution Across Target Nightshade Crop Diseases", w=165)
    
    pdf.add_page()
    pdf.fig(fig_dir / "new_dataset_crop_breakdown.png", "New Dataset Volume Share by Target Crop Species (Tomato, Potato, Pepper)", w=165)
    pdf.fig(fig_dir / "new_dataset_augmented_vs_raw.png", "New Dataset Raw Field Samples vs Synthetic Augmentation Ratios", w=165)

    pdf.add_page()
    pdf.sub("Production 3-Crop Target Dataset Distributions (31,071 Processed Images)")
    pdf.fig(fig_dir / "3crop_actual_01_class_distribution.png", "Production 3-Crop Target Dataset: 22 Disease Classes (31,071 Total Leaf Images)", w=165)
    pdf.fig(fig_dir / "3crop_actual_02_crop_breakdown.png", "Production Crop Volume Share (Tomato: 72.8%, Pepper: 20.0%, Potato: 7.2%)", w=165)

    pdf.add_page()
    pdf.fig(fig_dir / "3crop_actual_03_imbalance_ratios.png", "Class Imbalance Multipliers Across 22 Production Target Disease Classes", w=165)
    pdf.fig(fig_dir / "3crop_actual_04_split_verification.png", "GroupKFold Stratified Train / Val / Test Split Breakdown (80% / 10% / 10%)", w=165)

    # Section 3: Vision Suite
    pdf.add_page()
    pdf.sec("3. Vision AI Model Suite (EfficientNetV2-B2)")
    pdf.body("""ZARI.ai utilizes 4 distinct EfficientNetV2-B2 PyTorch models in a 2-stage hierarchical routing architecture. Below is the exact empirical checkpoint parameter registry:""")

    pdf.th(["Model Component", "Architecture", "Output Head", "Exact Params", "Checkpoint Path", "Metric / Accuracy"], [35, 30, 35, 25, 40, 25])
    pdf.tr(["Model A (Router)", "EfficientNetV2-B2", "3 Crop Classes", "7,772,858", "best_model_a_efficientnetv2_b2.pth", "99.48% Accuracy"], [35, 30, 35, 25, 40, 25], fill=True)
    pdf.tr(["Model B (Tomato)", "EDLEfficientNetB2", "13 Diseases", "7,786,948", "best_model_b_tomato.pth", "0.9787 Macro F1"], [35, 30, 35, 25, 40, 25])
    pdf.tr(["Model B (Potato)", "EDLEfficientNetB2", "3 Diseases", "7,772,858", "best_model_b_potato.pth", "0.9718 Macro F1"], [35, 30, 35, 25, 40, 25], fill=True)
    pdf.tr(["Model B (Pepper)", "EDLEfficientNetB2", "6 Diseases", "7,777,085", "best_model_b_pepper.pth", "0.9963 Macro F1"], [35, 30, 35, 25, 40, 25])
    pdf.tr(["TOTAL SYSTEM", "4-Model Suite", "22 Classes Total", "31,109,749", "355.73 MB Total Weights", "99.1% Reliability"], [35, 30, 35, 25, 40, 25], fill=True)

    pdf.ln(3)
    pdf.fig(fig_dir / "01_model_a_crop_router_curves.png", "Stage 1 Model A Crop Router Loss & Accuracy Trajectories (99.48% Accuracy)", w=160)

    pdf.add_page()
    pdf.fig(fig_dir / "02_model_b_disease_classifiers_curves.png", "Stage 2 Model B Disease Classifiers Macro F1 Training Trajectories (Tomato, Potato, Pepper)", w=160)
    pdf.fig(fig_dir / "03_swin_vs_efficientnet_f1_trajectories.png", "Swin Transformer vs EfficientNetV2-B2 Convergence Speed & F1 Comparison", w=160)

    # Section 4: EDL & SCRC
    pdf.add_page()
    pdf.sec("4. Evidential Deep Learning (EDL) & SCRC Safety Gates")
    pdf.body("""Evidential Deep Learning replaces standard Softmax with Dirichlet parameters to compute evidential uncertainty u = K / S:
1. Softplus Evidence Transformation: e_k = Softplus(z_k) = ln(1 + exp(z_k))
2. Dirichlet Parameters: alpha_k = e_k + 1.0, S = sum(alpha_k)
3. Evidential Uncertainty: u = K / S
4. Calibrated Safety Threshold: tau_SCRC = 0.3175 (scrc_threshold.json)
5. Dual Safety Gate Action Rule: Accept if (u <= 0.3175 AND p_crop >= 0.80 AND p_disease >= 0.70) else REJECT""")

    pdf.fig(fig_dir / "04_knowledge_distillation_curves.png", "Knowledge Distillation Loss Trajectories & Evidential Calibration Curves", w=160)

    # Section 5: Vector RAG & LLM
    pdf.add_page()
    pdf.sec("5. Multilingual Vector RAG & LLM Advisory Engine")
    pdf.body("""Ground-truth Integrated Pest Management (IPM) advisories are retrieved from ChromaDB (ml_pipeline/rag/chroma_db/) using sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384d):
- Total Knowledge Chunks: 208 Chunks (26 Disease Records x 8 IPM Sections)
- Primary LLM: Meta Llama 3.1 8B Instant on Groq LPUs (~750 tok/s)
- Offline Fallback: Deterministic Evidence Synthesizer (100% Edge Availability, 0.86ms Execution Time)""")

    # Appendix: Presentation Defense Slides
    pdf.add_page()
    pdf.sec("6. Appendix: Defense Presentation Slides & Architecture Diagrams")
    
    pdf.sub("Slide 1: System Workflow Architecture Block Diagram")
    pdf.th(["Step #", "Pipeline Module", "Input / Tech", "Parameter Count", "Latency"], [20, 50, 50, 40, 30])
    pdf.tr(["Step 1", "Client Upload", "Next.js 14 / HTML5", "0 Params", "< 1.0 ms"], [20, 50, 50, 40, 30], fill=True)
    pdf.tr(["Step 2", "Tensor Preprocessing", "PyTorch Torchvision", "0 Params", "0.45 ms"], [20, 50, 50, 40, 30])
    pdf.tr(["Step 3", "Model A Crop Router", "EfficientNetV2-B2", "7,772,858 Params", "1.25 ms"], [20, 50, 50, 40, 30], fill=True)
    pdf.tr(["Step 4", "Model B Classifier", "EDLEfficientNetB2", "7,786,948 Params", "1.54 ms"], [20, 50, 50, 40, 30])
    pdf.tr(["Step 5", "SCRC Risk Gate", "Dirichlet Uncertainty", "0 Params (tau=0.3175)", "0.45 ms"], [20, 50, 50, 40, 30], fill=True)
    pdf.tr(["Step 6", "Vector RAG Search", "MiniLM + ChromaDB", "22,713,216 Params", "5.32 ms"], [20, 50, 50, 40, 30])
    pdf.tr(["Step 7a", "Offline Advisory", "Deterministic Synthesis", "0 Params", "0.86 ms"], [20, 50, 50, 40, 30], fill=True)
    pdf.tr(["Step 7b", "Online Cloud LLM", "Llama 3.1 8B (Groq)", "8,030,261,248 Params", "380.00 ms"], [20, 50, 50, 40, 30])

    pdf.ln(4)
    pdf.sub("Slide 2: Active Pass vs Total System Parameter Comparison")
    pdf.th(["Execution Mode", "Active Models", "Active Parameter Count", "Weights Memory", "End-to-End Latency"], [45, 55, 40, 30, 20])
    pdf.tr(["Offline Edge Mode", "Model A + Model B + MiniLM", "38,273,022 Params", "177.93 MB", "9.40 ms"], [45, 55, 40, 30, 20], fill=True)
    pdf.tr(["Online Cloud Mode", "Model A + Model B + MiniLM + Llama", "8,068,534,270 Params", "Groq Cloud", "388.56 ms"], [45, 55, 40, 30, 20])

    # Save PDF
    out_dir = Path("/home/hammad/Desktop/project zari - experimental/ml_pipeline/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ZARI_FULL_THESIS_REPORT.pdf"
    pdf.output(str(out_path))
    print(f"✓ Created ZARI_FULL_THESIS_REPORT.pdf successfully at: {out_path}")

if __name__ == "__main__":
    build_pdf()
