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

class TechStackReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*GY)
        self.cell(0, 8, S("ZARI.ai -- Full Stack System Architecture & Technology Specification"), border=0, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(226, 232, 240)
        self.line(10, 15, 200, 15)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GY)
        self.cell(0, 10, S(f"Page {self.page_no()} of {{nb}}  |  ZARI.ai Production Architecture Documentation"), align="C")

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
    pdf = TechStackReportPDF("P", "mm", "A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    # Title Box
    pdf.title_box("ZARI.ai Full Stack Technology & System Architecture Report", "Frontend, Backend, Integration Layer, Vision Models, Parameters, RAG & LLM Engine")

    pdf.sec("1. Frontend Technology Stack")
    pdf.body("""The user interface of ZARI.ai is built using modern web development frameworks optimized for high performance, accessibility, and offline capability:""")
    pdf.kv("Core Framework", "Next.js 14 (App Router)")
    pdf.kv("UI Library", "React 18 with TypeScript")
    pdf.kv("Styling Engine", "TailwindCSS v3 + Vanilla CSS Micro-animations")
    pdf.kv("Icons & Motion", "Lucide-React + Framer Motion (hydration-safe direct imports)")
    pdf.kv("Static Export", "Next.js Static Site Generation (output: 'export' -> frontend/out)")
    pdf.kv("Image Handling", "HTML5 FileReader API & Canvas for client-side image preview")

    pdf.sec("2. Backend Technology Stack")
    pdf.body("""The backend diagnostic microservice is built in Python, optimized for CUDA GPU acceleration and high concurrency:""")
    pdf.kv("Language & Runtime", "Python 3.10+ / Uvicorn ASGI High-Performance Server")
    pdf.kv("Web Framework", "FastAPI (Asynchronous REST API Endpoints)")
    pdf.kv("Deep Learning Framework", "PyTorch 2.x + Torchvision (CUDA 12.x Accelerated)")
    pdf.kv("Image Preprocessing", "Pillow (PIL) + Torchvision Transforms (384x384 Tensor Scaling)")
    pdf.kv("Log Monitoring", "JSONL Production Monitor Log (backend/monitor_log.jsonl)")

    pdf.sec("3. Single-Server Integration Architecture (Connecting Both)")
    pdf.body("""To eliminate cross-origin complexity and allow deployment on a single port for the defense demonstration, the frontend static build is mounted directly inside FastAPI:""")
    pdf.bul("Single Server Port: http://127.0.0.1:8000 (serves both Web UI and AI API)")
    pdf.bul("FastAPI Mount Code: app.mount('/', StaticFiles(directory='frontend/out', html=True))")
    pdf.bul("Diagnostic API Endpoint: POST /api/diagnose (Receives multipart image file + language code)")
    pdf.bul("Health Check Endpoint: GET /health (Returns model status, CUDA device, and SCRC threshold)")
    pdf.bul("Class Registry Endpoint: GET /api/classes (Returns 22 active production target classes)")

    pdf.sec("4. Machine Learning Vision Model Suite & Parameters")
    pdf.body("""ZARI.ai utilizes a 2-Stage Hierarchical Model Suite consisting of 4 PyTorch EfficientNetV2-B2 models:""")

    pdf.th(["Model Component", "Architecture", "Output Head", "Exact Params", "Checkpoint Path", "Metric / Accuracy"], [35, 30, 35, 25, 40, 25])
    pdf.tr(["Model A (Router)", "EfficientNetV2-B2", "3 Crop Classes", "7,772,858", "best_model_a_efficientnetv2_b2.pth", "99.48% Accuracy"], [35, 30, 35, 25, 40, 25], fill=True)
    pdf.tr(["Model B (Tomato)", "EDLEfficientNetB2", "13 Diseases", "7,786,948", "best_model_b_tomato.pth", "0.9787 Macro F1"], [35, 30, 35, 25, 40, 25])
    pdf.tr(["Model B (Potato)", "EDLEfficientNetB2", "3 Diseases", "7,772,858", "best_model_b_potato.pth", "0.9718 Macro F1"], [35, 30, 35, 25, 40, 25], fill=True)
    pdf.tr(["Model B (Pepper)", "EDLEfficientNetB2", "6 Diseases", "7,777,085", "best_model_b_pepper.pth", "0.9963 Macro F1"], [35, 30, 35, 25, 40, 25])
    pdf.tr(["TOTAL SYSTEM", "4-Model Suite", "22 Classes Total", "31,109,749", "355.73 MB Total Weights", "99.1% Reliability"], [35, 30, 35, 25, 40, 25], fill=True)

    pdf.sec("5. Evidential Deep Learning (EDL) & SCRC Safety Control")
    pdf.kv("Dirichlet Softplus Evidence", "e_k = Softplus(z_k) = ln(1 + exp(z_k))")
    pdf.kv("Evidential Uncertainty Equation", "u = K / S = K / sum(e_k + 1)")
    pdf.kv("Calibrated Safety Threshold", "tau_SCRC = 0.3175")
    pdf.kv("Dual Safety Gate Action Rule", "Accept if (u <= 0.3175 AND p_crop >= 0.80 AND p_disease >= 0.70) else REJECT")

    pdf.sec("6. Vector RAG & LLM Advisory Engine")
    pdf.kv("Vector Store Database", "ChromaDB v0.5+ HNSW Cosine Index (ml_pipeline/rag/chroma_db/)")
    pdf.kv("Dense Embedder Model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384d)")
    pdf.kv("Knowledge Base Scope", "208 Chunks across 26 Records (22 Production Classes + 4 Quarantine Records)")
    pdf.kv("Primary LLM Engine", "Meta Llama 3.1 8B Instant (8 billion params) on Groq LPUs (~750 tok/s)")
    pdf.kv("Offline Fallback Synthesizer", "Deterministic Evidence Synthesis Engine (100% Offline Edge Mode)")

    pdf.sec("7. Performance Benchmarks")
    pdf.kv("Vision GPU Latency", "3.24 ms CUDA inference time")
    pdf.kv("ChromaDB Retrieval Latency", "5.32 ms HNSW search time")
    pdf.kv("Offline Synthesis Latency", "0.86 ms deterministic formatting time")
    pdf.kv("TOTAL OFFLINE EDGE LATENCY", "9.40 ms (Vision + RAG + Offline Synthesis)")
    pdf.kv("Groq Llama 3.1 8B API Call", "380 ms mean network + generation latency")
    pdf.kv("TOTAL ONLINE CLOUD LATENCY", "389 ms (Vision + RAG + Groq API)")

    # Save PDF
    out_dir = Path("/home/hammad/Desktop/project zari - experimental/ml_pipeline/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ZARI_FULL_STACK_SYSTEM_DOCUMENTATION.pdf"
    pdf.output(str(out_path))
    print(f"✓ Created ZARI_FULL_STACK_SYSTEM_DOCUMENTATION.pdf successfully at: {out_path}")

if __name__ == "__main__":
    build_pdf()
