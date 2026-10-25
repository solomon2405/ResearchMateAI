"""
Project-Scoped Persistent FAISS Vector Store.
Ensures strict multi-user and multi-project isolation by storing separate
FAISS indexes and ID mappings per project.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import faiss
from django.conf import settings

logger = logging.getLogger(__name__)

class ProjectVectorStore:
    def __init__(self, project_id: int):
        self.project_id = project_id
        self.index_dir = Path(getattr(settings, 'VECTOR_INDEX_DIR', settings.BASE_DIR / 'vector_indexes'))
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.index_dir / f"project_{project_id}.index"
        self.mapping_path = self.index_dir / f"project_{project_id}_map.json"
        
        self.dimension = 384
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunk_ids: List[int] = []
        self._load_or_create()

    def _load_or_create(self):
        """Load index and ID mapping from disk, or initialize fresh index."""
        if self.index_path.exists() and self.mapping_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.mapping_path, 'r', encoding='utf-8') as f:
                    self.chunk_ids = json.load(f)
                return
            except Exception as e:
                logger.error(f"Error reading vector index for project {self.project_id}: {e}. Creating new index.")

        # Create IndexFlatIP (Inner Product = Cosine Similarity for unit vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_ids = []

    def save(self):
        """Persist index and chunk ID mapping to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.mapping_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunk_ids, f)

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Add new chunk embeddings to this project's vector store.
        
        Args:
            chunks: List of chunk objects/dicts containing 'id'.
            embeddings: 2D numpy array of shape (len(chunks), dimension).
        """
        if len(chunks) == 0:
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of chunks does not match embeddings count.")

        # Ensure float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        start_vector_id = self.index.ntotal
        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            chunk_id = chunk['id'] if isinstance(chunk, dict) else chunk.id
            self.chunk_ids.append(chunk_id)

        self.save()
        logger.info(f"Added {len(chunks)} chunks to project {self.project_id} index (total: {self.index.ntotal}).")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Perform similarity search using query vector.
        
        Returns:
            List of (chunk_id, similarity_score) sorted by score descending.
        """
        if self.index is None or self.index.ntotal == 0 or len(self.chunk_ids) == 0:
            return []

        # Prepare query vector
        query_vec = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
                results.append((chunk_id, float(dist)))

        return results

    def rebuild_from_database(self):
        """
        Reconstruct this project's index from all DocumentChunk records in DB.
        Useful after deleting documents or when recovering from index inconsistencies.
        """
        from documents.models import DocumentChunk
        from .embeddings import embed_texts

        chunks = list(DocumentChunk.objects.filter(project_id=self.project_id).order_by('id'))
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_ids = []

        if not chunks:
            self.save()
            return

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)
        self.add_chunks(chunks, embeddings)

    def delete_index(self):
        """Remove index files from disk when project is deleted."""
        try:
            if self.index_path.exists():
                os.remove(self.index_path)
            if self.mapping_path.exists():
                os.remove(self.mapping_path)
            self.index = None
            self.chunk_ids = []
            logger.info(f"Deleted vector index files for project {self.project_id}.")
        except Exception as e:
            logger.error(f"Error deleting index for project {self.project_id}: {e}")


def get_project_vector_store(project_id: int) -> ProjectVectorStore:
    """Helper to instantiate or retrieve project vector store."""
    return ProjectVectorStore(project_id)
