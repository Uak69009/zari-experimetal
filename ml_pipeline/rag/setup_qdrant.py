"""ZARI.ai — Qdrant Collection Setup for RAG Knowledgebase.

This script creates and configures the Qdrant vector database collection:
- Collection Name      : "zari_treatment_kb"
- Dense Vector Config  : 384 dimensions (MiniLM-L12-v2), Cosine distance ("dense")
- Sparse Vector Config : SparseVectorParams ("sparse")
- Payload Indexes      : disease_id, crop, country, province, section,
                         evidence_level, verified, parent_id

Client Connection:
- QDRANT_URL environment variable or localhost:6333
- Fallback: Local persistent Qdrant storage at ml_pipeline/data/qdrant_db
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PayloadSchemaType,
        SparseVectorParams,
        VectorParams,
    )
except ImportError:
    QdrantClient = None

# Paths & Collection Constants
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
QDRANT_STORAGE_PATH = DATA_DIR / "qdrant_db"
COLLECTION_NAME = "zari_treatment_kb"

# Fields to index in Payload
PAYLOAD_INDEX_FIELDS = [
    ("disease_id", PayloadSchemaType.KEYWORD),
    ("crop", PayloadSchemaType.KEYWORD),
    ("country", PayloadSchemaType.KEYWORD),
    ("province", PayloadSchemaType.KEYWORD),
    ("section", PayloadSchemaType.KEYWORD),
    ("evidence_level", PayloadSchemaType.KEYWORD),
    ("verified", PayloadSchemaType.BOOL),
    ("parent_id", PayloadSchemaType.KEYWORD),
]


def get_qdrant_client() -> tuple[QdrantClient, str]:
    """Initialize QdrantClient connecting to server or local disk storage fallback."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # Try connecting to HTTP Qdrant server
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=5)
        # Test connection with healthcheck call
        client.get_collections()
        return client, f"Qdrant Server ({qdrant_url})"
    except Exception as e:
        print(f"Note: Qdrant server unavailable at '{qdrant_url}' ({e}).")
        print(f"Fallback to local persistent storage: {QDRANT_STORAGE_PATH}")
        QDRANT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(QDRANT_STORAGE_PATH))
        return client, f"Local Disk Storage ({QDRANT_STORAGE_PATH})"


def main() -> None:
    print("=" * 70)
    print("  ZARI.ai — QDRANT RAG COLLECTION INITIALIZATION")
    print("=" * 70)

    if QdrantClient is None:
        raise ImportError("qdrant-client is required. Run: pip install qdrant-client sentence-transformers")

    client, storage_desc = get_qdrant_client()
    print(f"\nUsing Qdrant Storage: {storage_desc}")

    # 1. Check if Collection Exists
    existing_collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing_collections:
        print(f"✓ Collection '{COLLECTION_NAME}' already exists.")
    else:
        print(f"Creating collection '{COLLECTION_NAME}'...")

        # 2. Vector Configurations
        vectors_config = {
            "dense": VectorParams(
                size=384,  # MiniLM-L12-v2 dimension
                distance=Distance.COSINE,
            )
        }

        sparse_vectors_config = {
            "sparse": SparseVectorParams()
        }

        # Create Collection
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        print(f"✓ Created collection '{COLLECTION_NAME}' with dense (384-dim, Cosine) and sparse vectors.")

    # 3. Create Payload Indexes
    print("\nCreating Payload Indexes...")
    for field_name, schema_type in PAYLOAD_INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema_type,
            )
            print(f"  ✓ Indexed field: '{field_name}' ({schema_type.name})")
        except Exception as e:
            # Handle if payload index already exists
            print(f"  - Field '{field_name}': {e}")

    # 4. Final Confirmation
    collection_info = client.get_collection(COLLECTION_NAME)
    print("\n" + "=" * 70)
    print("  COLLECTION CREATED CONFIRMATION")
    print("=" * 70)
    print(f"✓ Collection Name       : {COLLECTION_NAME}")
    print(f"✓ Storage Mechanism     : {storage_desc}")
    print(f"✓ Dense Vector          : 'dense' (size=384, distance=COSINE)")
    print(f"✓ Sparse Vector         : 'sparse' (SparseVectorParams)")
    print(f"✓ Total Payload Indexes : {len(PAYLOAD_INDEX_FIELDS)} fields indexed")
    print(f"✓ Indexed Fields        : {[f[0] for f in PAYLOAD_INDEX_FIELDS]}")
    print(f"✓ Points Count          : {collection_info.points_count}")
    print("\n✅ QDRANT RAG COLLECTION INITIALIZATION COMPLETE!")


if __name__ == "__main__":
    main()
