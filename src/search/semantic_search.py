"""
FAISS-based semantic search over document collection.
Uses sentence-transformers to embed documents and queries.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class SemanticSearchEngine:
    """
    Semantic document search using sentence embeddings + FAISS index.

    Workflow:
        1. Embed documents with SentenceTransformer
        2. Index embeddings in FAISS (L2 distance)
        3. At query time: embed query → search FAISS → return top-k docs
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "data/processed/faiss.index",
        metadata_path: str = "data/processed/metadata.pkl",
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.metadata: List[Dict] = []
        self.index: Optional[faiss.IndexFlatL2] = None

        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Search engine ready (dim={self.embedding_dim})")

        # Load existing index if available
        if self.index_path.exists() and self.metadata_path.exists():
            self._load_index()

    # ── Indexing ────────────────────────────────────────────────────────────────

    def add_document(self, doc_id: str, text: str, metadata: Dict = None) -> None:
        """Add a single document to the search index."""
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Embed the document text
        embedding = self.model.encode([text[:2000]], normalize_embeddings=True)
        self.index.add(embedding.astype(np.float32))

        # Store metadata
        entry = {"id": doc_id, "text_snippet": text[:300], **(metadata or {})}
        self.metadata.append(entry)

        self._save_index()

    def add_documents(self, documents: List[Dict]) -> None:
        """
        Batch add documents.
        Each doc dict must have: id, text, and optional metadata fields.
        """
        if not documents:
            return

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        texts = [doc["text"][:2000] for doc in documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

        self.index.add(embeddings.astype(np.float32))

        for doc in documents:
            entry = {
                "id": doc["id"],
                "text_snippet": doc["text"][:300],
                **{k: v for k, v in doc.items() if k not in ("id", "text")},
            }
            self.metadata.append(entry)

        self._save_index()

    # ── Search ──────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents semantically similar to the query.

        Returns:
            List of dicts with document metadata and similarity score.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Embed query
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        # Search FAISS
        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            result = {
                **self.metadata[idx],
                "similarity_score": round(float(1 / (1 + dist)), 4),  # Convert L2 to similarity
                "rank": len(results) + 1,
            }
            results.append(result)

        return results

    def get_index_size(self) -> int:
        """Return number of documents in the index."""
        return self.index.ntotal if self.index else 0

    def clear_index(self) -> None:
        """Clear the FAISS index and metadata."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        self._save_index()

    # ── Persistence ─────────────────────────────────────────────────────────────

    def _save_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def _load_index(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"📂 Loaded index with {self.index.ntotal} documents")
