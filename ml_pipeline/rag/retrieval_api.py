"""
ZARI.ai — Multilingual Semantic RAG Retrieval API
Queries ChromaDB Persistent Collection 'zari_3crop_treatment_kb' using
SentenceTransformers 'paraphrase-multilingual-MiniLM-L12-v2' (384-dimensional dense vectors).

Supports:
- Multilingual query embedding (English, Urdu, Pashto)
- Metadata filtering: crop, disease_class, section, evidence_level
- Cosine distance / similarity scoring
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CHROMA_DIR = SCRIPT_DIR / "chroma_db"
COLLECTION_NAME = "zari_3crop_treatment_kb"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_chroma_client = None
_collection = None
_embedder = None

def get_retriever():
    global _chroma_client, _collection, _embedder
    if _collection is None:
        print(f"Connecting to ChromaDB at: {CHROMA_DIR}")
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"Loaded ChromaDB collection '{COLLECTION_NAME}' with {_collection.count()} chunks.")
        print(f"Loading embedding model '{MODEL_NAME}'...")
        _embedder = SentenceTransformer(MODEL_NAME)
    return _collection, _embedder

def retrieve(
    query: str,
    disease_class: Optional[str] = None,
    crop: Optional[str] = None,
    intent: Optional[str] = None,
    language: str = "en",
    k: int = 6
) -> List[Dict[str, Any]]:
    """
    Multilingual semantic retrieval from ChromaDB vector store.
    
    Args:
        query: Search query in English, Urdu, or Pashto
        disease_class: Optional exact canonical disease class filter
        crop: Optional crop filter ('Tomato', 'Potato', 'Pepper')
        intent: Optional intent/section filter (e.g. 'chemical_control', 'prevention')
        language: Query language code ('en', 'ur', 'ps')
        k: Number of chunks to retrieve (default 6)
        
    Returns:
        List of dicts with: id, text, metadata, similarity_score, distance
    """
    collection, embedder = get_retriever()
    
    # 0. Auto-detect disease_class and crop if query contains canonical class name
    if not disease_class and query:
        for crop_candidate in ["Tomato", "Potato", "Pepper"]:
            if crop_candidate in query:
                if not crop:
                    crop = crop_candidate
                if "_" in query:
                    disease_class = query
                break

    if not crop and disease_class and "_" in disease_class:
        crop = disease_class.split("_")[0]

    # 1. Generate multilingual dense embedding
    query_vector = embedder.encode(query, convert_to_numpy=True).tolist()
    
    # 2. Build Chroma metadata filter (`where` dict)
    where_clauses = []
    if crop:
        where_clauses.append({"crop": {"$eq": crop}})
    if disease_class:
        where_clauses.append({"disease_class": {"$eq": disease_class}})
    if intent:
        where_clauses.append({"section": {"$eq": intent}})
        
    if len(where_clauses) == 1:
        where_filter = where_clauses[0]
    elif len(where_clauses) > 1:
        where_filter = {"$and": where_clauses}
    else:
        where_filter = None
        
    # 3. Query ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    # Fallback: if strict section/disease filter returned 0 results, retry with crop filter only
    if (not results or not results["documents"] or len(results["documents"][0]) == 0) and crop:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where={"crop": {"$eq": crop}},
            include=["documents", "metadatas", "distances"]
        )
    
    output_chunks = []
    if results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        ids = results["ids"][0]
        
        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            # Convert cosine distance to similarity score (cosine space: dist in [0, 2])
            sim_score = max(0.0, 1.0 - float(dist))
            output_chunks.append({
                "id": doc_id,
                "chunk_id": doc_id,
                "text": doc,
                "metadata": meta,
                "distance": round(float(dist), 4),
                "similarity_score": round(sim_score, 4)
            })
            
    return output_chunks

# ── Convenience Aliases ───────────────────────────────────────────────────────
def retrieve_symptoms(query: str, disease_class: Optional[str] = None, crop: Optional[str] = None, language: str = "en", k: int = 5) -> List[Dict[str, Any]]:
    return retrieve(query, disease_class=disease_class, crop=crop, intent="symptoms", language=language, k=k)

def retrieve_treatment(query: str, disease_class: Optional[str] = None, crop: Optional[str] = None, language: str = "en", k: int = 5) -> List[Dict[str, Any]]:
    return retrieve(query, disease_class=disease_class, crop=crop, intent="chemical_control", language=language, k=k)

def retrieve_prevention(query: str, disease_class: Optional[str] = None, crop: Optional[str] = None, language: str = "en", k: int = 5) -> List[Dict[str, Any]]:
    return retrieve(query, disease_class=disease_class, crop=crop, intent="prevention", language=language, k=k)

def retrieve_pakistan(query: str, disease_class: Optional[str] = None, crop: Optional[str] = None, language: str = "en", k: int = 5) -> List[Dict[str, Any]]:
    return retrieve(query, disease_class=disease_class, crop=crop, intent="safety", language=language, k=k)


# ── Test Suite Execution ──────────────────────────────────────────────────────
def run_test_suite():
    print("=" * 75)
    print("  ZARI.ai — RAG MULTILINGUAL RETRIEVAL API VERIFICATION TEST SUITE")
    print("=" * 75)
    
    collection, _ = get_retriever()
    total_count = collection.count()
    print(f"\n✓ CHROMADB COLLECTION SIZE: {total_count} total chunks")
    
    test_queries = [
        # Query 1: English - Tomato early blight treatment
        {
            "num": 1,
            "title": "Query 1 (English): 'Tomato early blight treatment'",
            "query": "Tomato early blight treatment",
            "crop": "Tomato",
            "disease_class": "Tomato_Early_Blight",
            "intent": "chemical_control",
            "language": "en"
        },
        # Query 2: English - Potato viral disease management
        {
            "num": 2,
            "title": "Query 2 (English): 'Potato viral disease management'",
            "query": "Potato viral disease management",
            "crop": "Potato",
            "disease_class": None,  # Open across potato viral classes
            "intent": None,
            "language": "en"
        },
        # Query 3: English - Pepper bacterial spot prevention
        {
            "num": 3,
            "title": "Query 3 (English): 'Pepper bacterial spot prevention'",
            "query": "Pepper bacterial spot prevention",
            "crop": "Pepper",
            "disease_class": "Pepper_Bacterial_Spot",
            "intent": "prevention",
            "language": "en"
        },
        # Query 4a: Urdu - Tomato early blight treatment
        {
            "num": "4a",
            "title": "Query 4a (Urdu): 'ٹماٹر کا اگیتا جھلساؤ کا علاج' (Tomato early blight treatment)",
            "query": "ٹماٹر کا اگیتا جھلساؤ کا علاج",
            "crop": "Tomato",
            "disease_class": "Tomato_Early_Blight",
            "intent": None,
            "language": "ur"
        },
        # Query 4b: Urdu - Potato viral disease management
        {
            "num": "4b",
            "title": "Query 4b (Urdu): 'آلو کے وائرس کی بیماریوں کا بندوبست' (Potato viral disease management)",
            "query": "آلو کے وائرس کی بیماریوں کا بندوبست",
            "crop": "Potato",
            "disease_class": None,
            "intent": None,
            "language": "ur"
        },
        # Query 4c: Urdu - Pepper bacterial spot prevention
        {
            "num": "4c",
            "title": "Query 4c (Urdu): 'شملہ مرچ کا بیکٹیریائی دھبے کا بچاؤ' (Pepper bacterial spot prevention)",
            "query": "شملہ مرچ کا بیکٹیریائی دھبے کا بچاؤ",
            "crop": "Pepper",
            "disease_class": "Pepper_Bacterial_Spot",
            "intent": "prevention",
            "language": "ur"
        }
    ]
    
    all_results = {}
    
    for tq in test_queries:
        print(f"\n{'─'*75}")
        print(f"  TEST QUERY #{tq['num']}: {tq['title']}")
        print(f"{'─'*75}")
        
        t0 = time.time()
        chunks = retrieve(
            query=tq["query"],
            crop=tq["crop"],
            disease_class=tq["disease_class"],
            intent=tq["intent"],
            language=tq["language"],
            k=5
        )
        latency_ms = (time.time() - t0) * 1000
        
        print(f"  Query Text  : \"{tq['query']}\"")
        print(f"  Filters     : Crop={tq['crop']} | Class={tq['disease_class']} | Intent={tq['intent']}")
        print(f"  Latency     : {latency_ms:.2f} ms")
        print(f"  Retrieved   : {len(chunks)} chunks")
        
        all_results[tq['title']] = []
        
        for idx, c in enumerate(chunks, 1):
            m = c["metadata"]
            preview = c["text"][:120].replace("\n", " ") + "..."
            print(f"\n    [{idx}] Chunk ID     : {c['id']}")
            print(f"        Sim Score    : {c['similarity_score']:.4f} (Distance: {c['distance']:.4f})")
            print(f"        Class/Sec    : {m['disease_class']} / {m['section']}")
            print(f"        Source URL   : {m['source_url']}")
            print(f"        Text Preview : {preview}")
            
            all_results[tq['title']].append({
                "rank": idx,
                "id": c["id"],
                "similarity_score": c["similarity_score"],
                "distance": c["distance"],
                "disease_class": m["disease_class"],
                "section": m["section"],
                "source_url": m["source_url"],
                "text_snippet": preview
            })
            
    # Save verification output JSON
    test_out_path = REPO_ROOT / "ml_pipeline" / "data" / "retrieval_test_results.json"
    with open(test_out_path, "w") as f:
        json.dump({
            "collection_size": total_count,
            "test_queries_count": len(test_queries),
            "results": all_results
        }, f, indent=2)
    print(f"\n✓ Test results JSON saved to: {test_out_path.relative_to(REPO_ROOT)}")
    print("\nSTOP — Phase 5 Multilingual ChromaDB Ingestion & Retrieval API Complete.")

if __name__ == "__main__":
    run_test_suite()
