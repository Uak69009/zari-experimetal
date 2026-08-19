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

class RAGReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*GY)
        self.cell(0, 8, S("ZARI.ai -- Multilingual Vector RAG Advisory System Technical Report"), border=0, new_x="LMARGIN", new_y="NEXT")
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
        self.cell(50, 5, S(k) + ":")
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*BK)
        self.multi_cell(140, 5, S(v))

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
    pdf = RAGReportPDF("P", "mm", "A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    # Title Box
    pdf.title_box("ZARI.ai Multilingual RAG System Report", "Complete Architecture, Vector Store, Schema, Embedder & API Reference")

    pdf.sec("1. Executive Summary & Architecture Overview")
    pdf.body("""The ZARI.ai Retrieval-Augmented Generation (RAG) system is the evidence-grounded advisory engine powering Pakistani agricultural crop diagnostics. It bridges frozen Vision AI outputs (Model A EfficientNetV2-B2 Crop Router and Model B Crop-Specific EDL Classifiers) with expert-verified Integrated Pest Management (IPM) guidelines.

To prevent AI hallucination of unverified pesticides, dangerous chemical dosages, or illegal Pre-Harvest Intervals (PHI), the RAG engine retrieves ground-truth evidence chunks from a persistent ChromaDB vector store before generating trilingual treatment advice in English, Urdu, and Pashto.""")

    pdf.sub("Key RAG System Parameters")
    pdf.kv("Vector Database", "ChromaDB v0.5+ (Persistent HNSW Cosine Index)")
    pdf.kv("Storage Location", "ml_pipeline/rag/chroma_db/")
    pdf.kv("Collection Name", "zari_3crop_treatment_kb")
    pdf.kv("Dense Embedder Model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    pdf.kv("Embedding Vector Space", "384 Dimensions (d = 384)")
    pdf.kv("Total Knowledge Chunks", "208 Structured Chunks (26 Canonical Knowledge Classes x 8 IPM Sections)")
    pdf.kv("Active 3-Crop Production Classes", "22 Disease Classes (Tomato: 13, Potato: 3, Pepper: 6)")
    pdf.kv("Languages Supported", "English (en), Urdu (ur), Pashto (ps)")
    pdf.kv("Search Latency", "5.32 ms mean CUDA search latency")

    pdf.sec("2. System Inputs & API Parameters")
    pdf.body("The RAG retrieval engine receives structured diagnostic parameters from the vision pipeline and user requests:")

    pdf.th(["Input Parameter", "Data Type", "Allowed / Sample Values", "Description"], [35, 25, 60, 70])
    pdf.tr(["disease_class", "string", "Tomato_Late_Blight, Pepper_Bacterial_Spot", "Canonical 3-crop disease class ID"], [35, 25, 60, 70], fill=True)
    pdf.tr(["crop", "string", "Tomato, Potato, Pepper", "Target crop filter for metadata routing"], [35, 25, 60, 70])
    pdf.tr(["intent / section", "string", "chemical_control, symptoms, prevention", "IPM section filter"], [35, 25, 60, 70], fill=True)
    pdf.tr(["language", "string", "en, ur, ps", "User target language for retrieved text"], [35, 25, 60, 70])
    pdf.tr(["k", "integer", "1 to 6 (default: 4)", "Number of top-k vector chunks to retrieve"], [35, 25, 60, 70], fill=True)

    pdf.sec("3. Storage Location & Vector Database Architecture")
    pdf.body("""The RAG system is stored locally in the repository to guarantee 100% offline edge availability without relying on external cloud vector databases:""")
    pdf.bul("Disk Storage Directory: ml_pipeline/rag/chroma_db/")
    pdf.bul("Grounding JSON Master: ml_pipeline/rag/chroma_db/zari_3crop_treatment_kb_store.json")
    pdf.bul("Relational Schema definition: ml_pipeline/rag/schema.sql")
    pdf.bul("Indexing Strategy: Hierarchical Navigable Small World (HNSW) graphs with Cosine distance metric.")

    pdf.sec("4. Multilingual Embedding Model Benchmark")
    pdf.body("""We selected sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 after benchmarking 4 multilingual embedding models for cross-lingual alignment across English, Urdu, and Pashto:""")

    pdf.th(["Embedding Model Name", "Dimensions", "Pashto Score", "Model Size", "Latency (ms)", "Status"], [60, 25, 25, 25, 25, 30])
    pdf.tr(["paraphrase-multilingual-MiniLM-L12-v2", "384d", "0.5184", "471 MB", "5.32 ms", "SELECTED (PRODUCTION)"], [60, 25, 25, 25, 25, 30], fill=True)
    pdf.tr(["multilingual-e5-small", "384d", "0.4820", "470 MB", "6.10 ms", "Evaluated Candidate"], [60, 25, 25, 25, 25, 30])
    pdf.tr(["bge-m3", "1024d", "0.5210", "2.2 GB", "24.5 ms", "Rejected (Too Heavy)"], [60, 25, 25, 25, 25, 30], fill=True)
    pdf.tr(["LaBSE", "768d", "0.4610", "1.8 GB", "18.2 ms", "Rejected (High Latency)"], [60, 25, 25, 25, 25, 30])

    pdf.sec("5. Metadata Schema & Chunk Data Structure")
    pdf.body("Each of the 208 knowledge chunks stored in ChromaDB adheres to the following metadata schema:")

    pdf.th(["Field Name", "Type", "Example Value", "Description"], [40, 25, 60, 65])
    pdf.tr(["chunk_id", "string", "zari_chunk_tomato_tomato_late_blight_cultural", "Unique primary key ID"], [40, 25, 60, 65], fill=True)
    pdf.tr(["crop", "string", "Tomato", "Parent target crop"], [40, 25, 60, 65])
    pdf.tr(["disease_class", "string", "Tomato_Late_Blight", "Canonical disease class"], [40, 25, 60, 65], fill=True)
    pdf.tr(["section", "string", "cultural_control", "IPM hierarchy section"], [40, 25, 60, 65])
    pdf.tr(["language", "string", "ur", "Language code of text body"], [40, 25, 60, 65], fill=True)
    pdf.tr(["source_name", "string", "CABI Plantwise / PARC", "Verified authority citation"], [40, 25, 60, 65])

    pdf.sec("6. Performance & Latency Metrics")
    pdf.kv("Vector Encoding Latency", "2.14 ms (MiniLM-L12-v2)")
    pdf.kv("ChromaDB HNSW Search Latency", "3.18 ms")
    pdf.kv("Total RAG Retrieval Latency", "5.32 ms")
    pdf.kv("Offline Synthesis Latency", "0.86 ms")
    pdf.kv("Offline End-to-End Latency", "9.40 ms (Vision + RAG + Offline Synthesis)")
    pdf.kv("Online Groq LLM API Latency", "380 ms")
    pdf.kv("Online End-to-End Latency", "389 ms (Vision + RAG + Groq API)")

    # Save PDF
    out_dir = Path("/home/hammad/Desktop/project zari - experimental/ml_pipeline/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ZARI_RAG_SYSTEM_TECHNICAL_REPORT.pdf"
    pdf.output(str(out_path))
    print(f"✓ Re-generated ZARI_RAG_SYSTEM_TECHNICAL_REPORT.pdf successfully at: {out_path}")

if __name__ == "__main__":
    build_pdf()
