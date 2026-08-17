"""
ZARI.ai — ChromaDB & SQLite Vector Store Inspection Utility

Allows interactive viewing and querying of ChromaDB vector store collections,
chunk metadata, embeddings shape, and SQLite table schemas.
"""

import os
import sqlite3
import json
import pandas as pd
from pathlib import Path
import chromadb

REPO_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DB_DIR = REPO_ROOT / "ml_pipeline" / "rag" / "chroma_db"
SQLITE_DB_PATH = CHROMA_DB_DIR / "chroma.sqlite3"

def inspect_sqlite():
    print("=" * 75)
    print("  1. SQLITE DATABASE INSPECTOR (chroma.sqlite3)")
    print("=" * 75)
    
    if not SQLITE_DB_PATH.exists():
        print(f"❌ SQLite database not found at {SQLITE_DB_PATH}")
        return
        
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"✓ Found {len(tables)} SQLite Tables: {tables}\n")
    
    for tbl in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {tbl};")
        cnt = cursor.fetchone()[0]
        cursor.execute(f"PRAGMA table_info({tbl});")
        cols = [col[1] for col in cursor.fetchall()]
        print(f"  • Table '{tbl}' — {cnt} rows | Columns: {cols}")
        
    conn.close()

def inspect_chromadb():
    print("\n" + "=" * 75)
    print("  2. CHROMADB VECTOR STORE INSPECTOR")
    print("=" * 75)
    
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collections = client.list_collections()
    print(f"✓ Total ChromaDB Collections: {len(collections)}")
    
    for col in collections:
        print(f"\n  Collection Name: '{col.name}'")
        print(f"  Total Chunks   : {col.count()}")
        
        # Sample query
        sample = col.get(limit=3, include=["documents", "metadatas", "embeddings"])
        print("\n  Sample Ingested Document Metadata:")
        for idx in range(len(sample["ids"])):
            doc_id = sample["ids"][idx]
            meta = sample["metadatas"][idx]
            text_snippet = sample["documents"][idx][:120].replace("\n", " ") + "..."
            emb_shape = len(sample["embeddings"][idx]) if sample.get("embeddings") is not None and len(sample["embeddings"]) > idx else "N/A"
            
            print(f"    [{idx+1}] ID: {doc_id} | Embedding Dim: {emb_shape}")
            print(f"        Crop: {meta.get('crop')} | Class: {meta.get('disease_class')} | Section: {meta.get('section')}")
            print(f"        Snippet: \"{text_snippet}\"")
            print(f"        Source URL: {meta.get('source_url')}\n")

def main():
    print("=" * 75)
    print("  ZARI.ai — ChromaDB Vector Store & SQL Database Inspector")
    print("=" * 75)
    inspect_sqlite()
    inspect_chromadb()
    print("=" * 75)
    print("✓ Inspection complete. You can open 'chroma.sqlite3' using VS Code SQLite Viewer extension.")

if __name__ == "__main__":
    main()
