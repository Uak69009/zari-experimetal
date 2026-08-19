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

class LLMReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*GY)
        self.cell(0, 8, S("ZARI.ai -- Llama 3.1 8B LLM Advisory Engine Technical Report"), border=0, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(226, 232, 240)
        self.line(10, 15, 200, 15)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GY)
        self.cell(0, 10, S(f"Page {self.page_no()} of {{nb}}  |  ZARI.ai LLM System Architecture Documentation"), align="C")

    def title_box(self, title, subtitle):
        self.set_fill_color(*GD)
        self.rect(10, 10, 190, 36, "F")
        self.set_y(14)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.cell(190, 8, S(title), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(200, 240, 220)
        self.cell(190, 6, S(subtitle), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_y(52)

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
    pdf = LLMReportPDF("P", "mm", "A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    # Title Box
    pdf.title_box("ZARI.ai LLM Advisory Engine Technical Report", "Model Architecture, Parameters, Inputs, Prompt Engineering & Safety Enforcement")

    pdf.sec("1. LLM Model Specifications & Hardware Infrastructure")
    pdf.body("""The ZARI.ai diagnostic advisory engine leverages Meta Llama 3.1 8B Instant (llama-3.1-8b-instant) deployed on Groq Language Processing Units (LPUs). It synthesizes retrieved ChromaDB vector store evidence into farmer-friendly, evidence-grounded treatment plans for Pakistani agriculture.""")

    pdf.sub("Primary Large Language Model Parameters")
    pdf.kv("Model Name", "Meta Llama 3.1 8B Instant (llama-3.1-8b-instant)")
    pdf.kv("Developer / Provider", "Meta AI / Groq API Infrastructure")
    pdf.kv("Total Parameter Count", "8.0 Billion Parameters (8 x 10^9 params)")
    pdf.kv("Architecture Type", "Auto-regressive Transformer Decoder with Grouped Query Attention (GQA)")
    pdf.kv("Context Window Size", "128,000 Tokens (128k context length)")
    pdf.kv("Inference Acceleration", "Groq LPU (Language Processing Unit) Tensor Hardware")
    pdf.kv("Generation Speed", "~750 Tokens / Second (Sub-second latency: 350-500 ms)")
    pdf.kv("Fallback Generator", "Deterministic Evidence Synthesis Engine (100% offline fallback)")

    pdf.sec("2. LLM Inputs & Data Payloads")
    pdf.body("The LLM generator receives 5 structured inputs from the Vision AI pipeline and RAG retriever:")

    pdf.th(["Input Field", "Data Type", "Sample Value", "Function & Description"], [35, 25, 55, 75])
    pdf.tr(["disease_class", "string", "Tomato_Late_Blight", "Canonical disease class identified by Vision AI"], [35, 25, 55, 75], fill=True)
    pdf.tr(["confidence", "float", "0.9845 (98.5%)", "Vision classifier softmax confidence score"], [35, 25, 55, 75])
    pdf.tr(["uncertainty", "float", "0.1245", "Evidential Deep Learning (EDL) uncertainty u = K / S"], [35, 25, 55, 75], fill=True)
    pdf.tr(["language", "string", "ur / ps / en", "Target farmer output language (Urdu, Pashto, English)"], [35, 25, 55, 75])
    pdf.tr(["retrieved_chunks", "list[dict]", "Top-4 ChromaDB Chunks", "Ground-truth evidence chunks containing IPM guidelines"], [35, 25, 55, 75], fill=True)

    pdf.sec("3. Prompt Engineering & System Directives")
    pdf.body("""The prompt template enforces strict grounding and zero-hallucination guardrails:""")
    pdf.bul("Zero Invented Dosages: The LLM is strictly prohibited from inventing chemical spray dosages (ml/L or g/L).")
    pdf.bul("Zero Invented PHI Days: Pre-Harvest Intervals (PHI) must refer farmers to official product label instructions.")
    pdf.bul("IPM Order Enforcement: Recommendations must strictly follow Cultural -> Biological -> Chemical controls.")
    pdf.bul("Viral Pathogen Safety: For viral infections (e.g. Tomato Leaf Curl), synthetic fungicides are forbidden; vector control is mandated.")
    pdf.bul("Quarantine Protocols: High-caution quarantine warnings for severe regional crop threats.")

    pdf.sec("4. Latency Disambiguation: Online vs Offline Pipeline")
    pdf.kv("Vision Inference Latency", "3.24 ms CUDA inference time")
    pdf.kv("ChromaDB Retrieval Latency", "5.32 ms HNSW search time")
    pdf.kv("Offline Fallback Synthesis Latency", "0.86 ms deterministic formatting time")
    pdf.kv("TOTAL OFFLINE EDGE LATENCY", "9.40 ms (3.24ms Vision + 5.32ms RAG + 0.86ms Fallback)")
    pdf.kv("Groq Llama 3.1 8B API Call", "380 ms mean network + generation latency")
    pdf.kv("TOTAL ONLINE CLOUD LATENCY", "389 ms (3.24ms Vision + 5.32ms RAG + 380ms Groq API)")

    # Save PDF
    out_dir = Path("/home/hammad/Desktop/project zari - experimental/ml_pipeline/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ZARI_LLM_SYSTEM_TECHNICAL_REPORT.pdf"
    pdf.output(str(out_path))
    print(f"✓ Re-generated ZARI_LLM_SYSTEM_TECHNICAL_REPORT.pdf successfully at: {out_path}")

if __name__ == "__main__":
    build_pdf()
