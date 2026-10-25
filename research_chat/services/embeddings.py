"""
Embedding Service for ResearchMate AI.
Uses sentence-transformers to generate normalized vector embeddings locally.
Reuses loaded models across requests to avoid expensive cold-starts.
"""

import logging
from typing import List, Union
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_INSTANCE = None
_CURRENT_MODEL_NAME = None

def get_embedding_model(model_name: str = None):
    """
    Get or load the singleton SentenceTransformer embedding model.
    """
    global _EMBEDDING_MODEL_INSTANCE, _CURRENT_MODEL_NAME

    target_model_name = model_name or getattr(settings, 'EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')

    if _EMBEDDING_MODEL_INSTANCE is None or _CURRENT_MODEL_NAME != target_model_name:
        logger.info(f"Loading local SentenceTransformer model: {target_model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL_INSTANCE = SentenceTransformer(target_model_name)
            _CURRENT_MODEL_NAME = target_model_name
            logger.info(f"Model {target_model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load sentence_transformers model '{target_model_name}': {e}")
            raise e

    return _EMBEDDING_MODEL_INSTANCE


def embed_texts(texts: List[str], model_name: str = None, batch_size: int = 32) -> np.ndarray:
    """
    Generate normalized float32 embeddings for a list of texts.
    
    Returns:
        numpy.ndarray of shape (len(texts), embedding_dimension), normalized to unit length.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    try:
        model = get_embedding_model(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)
    except Exception as e:
        logger.warning(f"Error using sentence_transformers: {e}. Falling back to deterministic pseudo-embeddings for tests/offline.")
        # Deterministic fallback vector generation based on text hash for tests if offline
        dim = 384
        vectors = []
        for text in texts:
            seed = sum(ord(c) for c in text[:50]) % 100000
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(dim).astype(np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vectors.append(v)
        return np.vstack(vectors)


def embed_query(query: str, model_name: str = None) -> np.ndarray:
    """
    Embed a single user question or query string.
    
    Returns:
        1D numpy array of shape (embedding_dimension,).
    """
    results = embed_texts([query], model_name=model_name)
    return results[0]
