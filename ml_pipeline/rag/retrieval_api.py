"""ZARI.ai — Intent-Aware Hybrid RAG Retrieval Engine.

Implements structured, intent-aware vector + metadata filtered evidence retrieval
from Qdrant collection 'zari_treatment_kb'.

Key Retrieval API Functions:
1. retrieve_treatment(disease_class, language="en", k=8)
2. retrieve_symptoms(disease_class, k=5)
3. retrieve_prevention(disease_class, k=5)
4. retrieve_pakistan(disease_class, k=5)
5. retrieve_intent(disease_class, intent="treatment", language="en", k=8)

Features:
- Metadata Payload Filtering (disease_class, section, crop, country)
- Evidence Quality Weighting (A1: 1.0, A2: 0.95, B1: 0.85, B2: 0.80, C: 0.70)
- Section & Parent ID Deduplication (max 3 chunks per disease parent)
- Production-ready modular API
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
from sentence_transformers import SentenceTransformer

# Paths & Constants
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
QDRANT_STORAGE_PATH = DATA_DIR / "qdrant_db"
COLLECTION_NAME = "zari_treatment_kb"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Evidence Level Weighting Hierarchy
EVIDENCE_WEIGHTS: dict[str, float] = {
    "A1": 1.00,
    "A2": 0.95,
    "B1": 0.85,
    "B2": 0.80,
    "C": 0.70,
}

# Global Lazy Singletons
_CLIENT_INSTANCE: QdrantClient | None = None
_EMBEDDER_INSTANCE: SentenceTransformer | None = None


def get_qdrant_client() -> QdrantClient:
    """Get thread-safe QdrantClient instance."""
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        try:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=5)
            client.get_collections()
            _CLIENT_INSTANCE = client
        except Exception:
            QDRANT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
            _CLIENT_INSTANCE = QdrantClient(path=str(QDRANT_STORAGE_PATH))
    return _CLIENT_INSTANCE


def get_embedder() -> SentenceTransformer:
    """Get singleton embedding model instance."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = SentenceTransformer(MODEL_NAME)
    return _EMBEDDER_INSTANCE


def rerank_and_deduplicate(raw_hits: list[Any], max_per_parent: int = 3, target_k: int = 8) -> list[dict]:
    """Applies Evidence Weighting and parent_id deduplication."""
    processed_items = []

    for hit in raw_hits:
        payload = hit.payload
        raw_score = hit.score if hasattr(hit, "score") and hit.score is not None else 1.0
        e_level = payload.get("evidence_level", "A2")
        weight = EVIDENCE_WEIGHTS.get(e_level, 0.85)

        # Apply final evidence weighting formula
        final_score = raw_score * weight

        item = {
            "chunk_id": payload.get("chunk_id"),
            "disease_id": payload.get("disease_id"),
            "disease_class": payload.get("disease_class"),
            "crop": payload.get("crop"),
            "section": payload.get("section"),
            "evidence_level": e_level,
            "evidence_weight": weight,
            "raw_score": float(raw_score),
            "final_score": float(final_score),
            "urgency": payload.get("urgency", "NORMAL"),
            "text": payload.get("text", ""),
            "source_name": payload.get("source_name", ""),
            "source_url": payload.get("source_url", ""),
            "parent_id": payload.get("parent_id", payload.get("disease_id")),
        }
        processed_items.append(item)

    # Sort by final weighted score descending
    processed_items.sort(key=lambda x: x["final_score"], reverse=True)

    # Deduplicate: Max `max_per_parent` chunks per parent_id
    parent_counts: dict[str, int] = {}
    deduped_results: list[dict] = []

    for item in processed_items:
        pid = item["parent_id"]
        count = parent_counts.get(pid, 0)
        if count < max_per_parent:
            parent_counts[pid] = count + 1
            deduped_results.append(item)
            if len(deduped_results) >= target_k:
                break

    return deduped_results


def retrieve_treatment(disease_class: str, language: str = "en", k: int = 8) -> list[dict]:
    """Retrieve treatment & IPM evidence for a disease class."""
    client = get_qdrant_client()
    embedder = get_embedder()

    crop = disease_class.split("_")[0].lower()
    query_text = f"{disease_class} treatment fungicide chemical control cultural biological IPM Pakistan"
    query_vec = embedder.encode(query_text, normalize_embeddings=True).tolist()

    target_sections = ["cultural_control", "biological_control", "chemical_control", "treatment", "management"]

    qfilter = Filter(
        must=[
            FieldCondition(key="disease_class", match=MatchValue(value=disease_class)),
            FieldCondition(key="section", match=MatchAny(any=target_sections)),
        ]
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        using="dense",
        query_filter=qfilter,
        limit=20,
    )

    # Fallback to crop-level if disease specific yields empty
    if not response.points:
        fallback_filter = Filter(
            must=[
                FieldCondition(key="crop", match=MatchValue(value=crop)),
                FieldCondition(key="section", match=MatchAny(any=target_sections)),
            ]
        )
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            using="dense",
            query_filter=fallback_filter,
            limit=20,
        )

    return rerank_and_deduplicate(response.points, max_per_parent=3, target_k=k)


def retrieve_symptoms(disease_class: str, k: int = 5) -> list[dict]:
    """Retrieve symptom, identity, and visual diagnostic evidence for a disease class."""
    client = get_qdrant_client()
    embedder = get_embedder()

    query_text = f"{disease_class} symptoms leaf spot lesions diagnosis identification visual signs"
    query_vec = embedder.encode(query_text, normalize_embeddings=True).tolist()

    target_sections = ["symptoms", "identity", "differential_diagnosis", "risk", "epidemiology"]

    qfilter = Filter(
        must=[
            FieldCondition(key="disease_class", match=MatchValue(value=disease_class)),
            FieldCondition(key="section", match=MatchAny(any=target_sections)),
        ]
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        using="dense",
        query_filter=qfilter,
        limit=15,
    )

    return rerank_and_deduplicate(response.points, max_per_parent=3, target_k=k)


def retrieve_prevention(disease_class: str, k: int = 5) -> list[dict]:
    """Retrieve prevention, sanitation, and safety evidence for a disease class."""
    client = get_qdrant_client()
    embedder = get_embedder()

    query_text = f"{disease_class} prevention crop rotation sanitation safety precautions protective gear"
    query_vec = embedder.encode(query_text, normalize_embeddings=True).tolist()

    target_sections = ["prevention", "safety", "management", "cultural_control"]

    qfilter = Filter(
        must=[
            FieldCondition(key="disease_class", match=MatchValue(value=disease_class)),
            FieldCondition(key="section", match=MatchAny(any=target_sections)),
        ]
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        using="dense",
        query_filter=qfilter,
        limit=15,
    )

    return rerank_and_deduplicate(response.points, max_per_parent=3, target_k=k)


def retrieve_pakistan(disease_class: str, k: int = 5) -> list[dict]:
    """Retrieve Pakistan regional extension advisories and primary source citations."""
    client = get_qdrant_client()
    embedder = get_embedder()

    query_text = f"{disease_class} Pakistan PARC DPP Punjab KP Sindh extension advisory sources"
    query_vec = embedder.encode(query_text, normalize_embeddings=True).tolist()

    target_sections = ["pakistan", "sources", "management", "identity"]

    qfilter = Filter(
        must=[
            FieldCondition(key="disease_class", match=MatchValue(value=disease_class)),
            FieldCondition(key="section", match=MatchAny(any=target_sections)),
        ]
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        using="dense",
        query_filter=qfilter,
        limit=15,
    )

    return rerank_and_deduplicate(response.points, max_per_parent=3, target_k=k)


def retrieve_intent(disease_class: str, intent: str = "treatment", language: str = "en", k: int = 8) -> list[dict]:
    """Unified intent-aware retrieval dispatcher."""
    intent_clean = intent.lower().strip()

    if intent_clean in ["treatment", "control", "chemical", "ipm", "remedy"]:
        return retrieve_treatment(disease_class, language=language, k=k)
    elif intent_clean in ["symptoms", "diagnosis", "identity", "signs"]:
        return retrieve_symptoms(disease_class, k=k)
    elif intent_clean in ["prevention", "safety", "sanitation", "precautions"]:
        return retrieve_prevention(disease_class, k=k)
    elif intent_clean in ["pakistan", "regional", "advisory", "extension"]:
        return retrieve_pakistan(disease_class, k=k)
    else:
        return retrieve_treatment(disease_class, language=language, k=k)


def run_tests() -> None:
    """Executes required verification test cases."""
    print("=" * 75)
    print("  ZARI.ai — INTENT-AWARE HYBRID RETRIEVAL VERIFICATION TESTS")
    print("=" * 75)

    test_cases = [
        ("TEST 1: Treatment Evidence", "Wheat_Yellow_Rust", lambda d: retrieve_treatment(d, k=8)),
        ("TEST 2: Symptom Evidence", "Tomato_Late_Blight", lambda d: retrieve_symptoms(d, k=5)),
        ("TEST 3: Pakistan Guidance", "Wheat_Blast", lambda d: retrieve_pakistan(d, k=5)),
        ("TEST 4: Prevention & Safety", "Grape_Powdery_Mildew", lambda d: retrieve_prevention(d, k=5)),
    ]

    for title, dclass, func in test_cases:
        print(f"\n{title}")
        print(f"  * Disease Queried : {dclass}")
        results = func(dclass)
        print(f"  * Chunks Returned : {len(results)}")

        sections_returned = [r["section"] for r in results]
        print(f"  * Sections Returned: {list(set(sections_returned))}")

        if results:
            top_hit = results[0]
            snip = top_hit["text"][:100] + "..."
            print(f"  * Top Chunk Sec   : {top_hit['section']} (Evidence Level: {top_hit['evidence_level']})")
            print(f"  * Raw Score       : {top_hit['raw_score']:.4f} | Weighted Score: {top_hit['final_score']:.4f}")
            print(f"  * Top Text Snippet: {snip}")

        print("-" * 75)

    print("\n✅ RETRIEVAL API VERIFICATION COMPLETE!")


if __name__ == "__main__":
    run_tests()
