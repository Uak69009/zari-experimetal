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

class EfficientNetReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*GY)
        self.cell(0, 8, S("ZARI.ai -- EfficientNetV2-B2, EDL & SCRC Vision AI System Technical Report"), border=0, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(226, 232, 240)
        self.line(10, 15, 200, 15)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GY)
        self.cell(0, 10, S(f"Page {self.page_no()} of {{nb}}  |  ZARI.ai Vision AI Production Architecture Documentation"), align="C")

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
    pdf = EfficientNetReportPDF("P", "mm", "A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    # Title Box
    pdf.title_box("ZARI.ai EfficientNetV2-B2, EDL & SCRC System Report", "Model Selection Rationale, Total Parameters, Dirichlet EDL Uncertainty & SCRC Safety Gates")

    pdf.sec("1. Why EfficientNetV2-B2 Was Selected")
    pdf.body("""EfficientNetV2-B2 was chosen as the backbone architecture for ZARI.ai's 2-stage hierarchical vision pipeline over heavy Vision Transformers (Swin-Base, ViT-B/16) and ResNet baselines after rigorous empirical benchmark evaluations.""")

    pdf.sub("Key Architecture Advantages")
    pdf.bul("Fused-MBConv Layers: Replaces early 3x3 depthwise convolutions with standard 3x3 convolutions in lower stages, boosting training GPU throughput by 3.1x.")
    pdf.bul("Parametric Efficiency: Achieves 99.48% Crop Routing Accuracy with ~7.77 Million parameters per model (11x smaller than Swin-Base 88M).")
    pdf.bul("Progressive Learning: Jointly scales regularization and image resolution (384x384) during training to prevent over-fitting on plant leaf datasets.")
    pdf.bul("Ultra-Fast GPU Latency: ~3.2 ms CUDA latency per image tensor, enabling real-time edge execution.")

    pdf.sec("2. Total System Parameters & Empirical Checkpoints Registry")
    pdf.body("ZARI.ai utilizes a 2-Stage Hierarchical suite of 4 distinct EfficientNetV2-B2 PyTorch models. The table below lists exact empirical state dict parameter counts and checkpoint file sizes on disk:")

    pdf.th(["Model Component", "Architecture", "Output Head", "Exact Params", "Checkpoint Path", "Metric / Accuracy"], [35, 30, 35, 25, 40, 25])
    pdf.tr(["Model A (Router)", "EfficientNetV2-B2", "3 Crop Classes", "7,772,858", "best_model_a_efficientnetv2_b2.pth", "99.48% Accuracy"], [35, 30, 35, 25, 40, 25], fill=True)
    pdf.tr(["Model B (Tomato)", "EDLEfficientNetB2", "13 Diseases", "7,786,948", "best_model_b_tomato.pth", "0.9787 Macro F1"], [35, 30, 35, 25, 40, 25])
    pdf.tr(["Model B (Potato)", "EDLEfficientNetB2", "3 Diseases", "7,772,858", "best_model_b_potato.pth", "0.9718 Macro F1"], [35, 30, 35, 25, 40, 25], fill=True)
    pdf.tr(["Model B (Pepper)", "EDLEfficientNetB2", "6 Diseases", "7,777,085", "best_model_b_pepper.pth", "0.9963 Macro F1"], [35, 30, 35, 25, 40, 25])
    pdf.tr(["TOTAL SYSTEM", "4-Model Suite", "22 Classes Total", "31,109,749", "355.73 MB Total Checkpoints", "99.1% Reliability"], [35, 30, 35, 25, 40, 25], fill=True)

    pdf.sec("3. Evidential Deep Learning (EDL) Formulation")
    pdf.body("""Standard neural networks use Softmax to output point probabilities, which causes overconfidence on out-of-distribution (OOD) crops (e.g. Corn, Wheat). EDL replaces Softmax with Subjective Logic and Dirichlet distributions:""")

    pdf.kv("1. Evidence Softplus", "e_k = Softplus(z_k) = ln(1 + exp(z_k))")
    pdf.kv("2. Dirichlet Alpha Params", "alpha_k = e_k + 1.0")
    pdf.kv("3. Total Dirichlet Intensity", "S = sum_{k=1}^K alpha_k")
    pdf.kv("4. Evidential Uncertainty", "u = K / S = K / sum(e_k + 1)")
    pdf.kv("5. Class Probabilities", "p_k = alpha_k / S")

    pdf.body("""For an OOD crop image (e.g., Corn or Wheat), the model accumulates zero evidence (e_k -> 0), so S -> K, resulting in Evidential Uncertainty u -> 1.0 (High Uncertainty).""")

    pdf.sec("4. Selective Classification Risk Control (SCRC)")
    pdf.body("""SCRC enforces strict statistical risk control to limit the false acceptance rate of misclassified or OOD images to <= 5.0%:""")
    pdf.kv("Calibrated Safety Threshold", "tau_SCRC = 0.3175 (from scrc_threshold.json)")
    pdf.kv("Model A Confidence Gate", "p_crop >= 0.80 (Rejects non-target crops like Corn/Wheat)")
    pdf.kv("Model B Confidence Gate", "p_disease >= 0.70")
    pdf.kv("SCRC Action Rule", "Accept if (u <= 0.3175 AND p_crop >= 0.80 AND p_disease >= 0.70) else REJECT")

    pdf.sec("5. Production Input/Output & Latency Specification")
    pdf.kv("Input Image Shape", "RGB Image File (JPEG/PNG/WebP), resized to (384, 384, 3)")
    pdf.kv("Normalization Stats", "Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]")
    pdf.kv("Vision GPU Latency", "3.24 ms CUDA inference time")
    pdf.kv("Offline End-to-End Latency", "9.40 ms (Vision + ChromaDB RAG + Offline Synthesis)")
    pdf.kv("Online LLM Latency", "389 ms (Vision + ChromaDB RAG + Groq Llama 3.1 8B API)")
    pdf.kv("Output Payload", "status, disease_class, confidence, uncertainty, scrc_threshold, advisory")

    # Save PDF
    out_dir = Path("/home/hammad/Desktop/project zari - experimental/ml_pipeline/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ZARI_EFFICIENTNETV2_EDL_SCRC_TECHNICAL_REPORT.pdf"
    pdf.output(str(out_path))
    print(f"✓ Re-generated ZARI_EFFICIENTNETV2_EDL_SCRC_TECHNICAL_REPORT.pdf successfully at: {out_path}")

if __name__ == "__main__":
    build_pdf()
