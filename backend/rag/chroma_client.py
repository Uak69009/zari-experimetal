"""
ZARI.ai Backend — ChromaDB RAG Client
Manages the agricultural document vector store for retrieval-augmented generation.
"""

import os
from typing import List, Optional

from core.config import settings

# Lazy-loaded ChromaDB client
_chroma_client = None
_collection = None


def _get_collection():
    """Lazy-load the ChromaDB collection (singleton)."""
    global _chroma_client, _collection

    if _collection is None:
        import chromadb

        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
        )
        _collection = _chroma_client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"description": "ZARI.ai agricultural disease treatment documents"},
        )
    return _collection


def add_documents(
    documents: List[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
):
    """
    Add agricultural documents to the ChromaDB collection.

    Args:
        documents: List of text documents to store.
        metadatas: Optional metadata dicts for each document.
        ids: Optional unique IDs for each document.
    """
    collection = _get_collection()

    if ids is None:
        ids = [f"doc_{i}" for i in range(len(documents))]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )


def get_disease_context(
    disease_label: str,
    crop_name: str = "",
    top_k: int = 5,
) -> List[dict]:
    """
    Retrieve relevant agricultural documents for a diagnosed disease.

    Args:
        disease_label: Canonical disease label from CV inference.
        crop_name: Optional crop name for more specific retrieval.
        top_k: Number of documents to retrieve.

    Returns:
        List of dicts with 'document' and 'metadata' keys.
    """
    collection = _get_collection()

    query = f"{disease_label} treatment Pakistan {crop_name}".strip()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    # Format results
    context_docs = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = (
                results["metadatas"][0][i] if results["metadatas"] else {}
            )
            context_docs.append({
                "document": doc,
                "metadata": metadata,
                "distance": results["distances"][0][i] if results["distances"] else None,
            })

    return context_docs


def get_collection_count() -> int:
    """Return the number of documents in the collection."""
    collection = _get_collection()
    return collection.count()
