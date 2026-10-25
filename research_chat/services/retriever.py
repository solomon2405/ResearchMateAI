"""
Project-Scoped Evidence Retriever for RAG.
Queries only the authorized project's indexed chunks, evaluates similarity
against configured thresholds, and constructs a bounded evidence package.
"""

import logging
from typing import Dict, Any, List
from django.conf import settings
from documents.models import DocumentChunk
from .embeddings import embed_query
from .vector_store import get_project_vector_store

logger = logging.getLogger(__name__)

def retrieve_project_evidence(
    project_id: int,
    question: str,
    top_k: int = None,
    similarity_threshold: float = None
) -> Dict[str, Any]:
    """
    Retrieve evidence chunks for a user question strictly within the project scope.
    
    Returns:
        {
            'has_sufficient_evidence': bool,
            'passages': List[Dict[str, Any]],
            'top_score': float,
            'error': Optional[str]
        }
    """
    top_k = top_k or getattr(settings, 'RAG_TOP_K', 5)
    threshold = similarity_threshold or getattr(settings, 'RAG_SIMILARITY_THRESHOLD', 0.35)

    clean_question = question.strip()
    if not clean_question:
        return {
            'has_sufficient_evidence': False,
            'passages': [],
            'top_score': 0.0,
            'error': 'Question cannot be empty.'
        }

    # Embed user question
    query_vector = embed_query(clean_question)

    # Search project-isolated vector store
    vector_store = get_project_vector_store(project_id)
    search_hits = vector_store.search(query_vector, top_k=top_k * 2)

    if not search_hits:
        return {
            'has_sufficient_evidence': False,
            'passages': [],
            'top_score': 0.0,
            'error': None
        }

    # Extract chunk IDs passing similarity threshold
    hit_map = {chunk_id: score for chunk_id, score in search_hits if score >= threshold}
    
    if not hit_map:
        best_score = search_hits[0][1] if search_hits else 0.0
        return {
            'has_sufficient_evidence': False,
            'passages': [],
            'top_score': best_score,
            'error': None
        }

    # Fetch corresponding chunks from DB ensuring project_id match
    matching_chunks = DocumentChunk.objects.filter(
        id__in=hit_map.keys(),
        project_id=project_id,
        document__processing_status='indexed'
    ).select_related('document')

    passages = []
    seen_texts = set()

    for chunk in matching_chunks:
        # Deduplicate substantially identical texts
        text_signature = chunk.text[:120].strip()
        if text_signature in seen_texts:
            continue
        seen_texts.add(text_signature)

        score = hit_map[chunk.id]
        page_ref = f"{chunk.page_start}" if chunk.page_start == chunk.page_end else f"{chunk.page_start}-{chunk.page_end}"
        
        passages.append({
            'chunk_id': chunk.id,
            'document_id': chunk.document_id,
            'filename': chunk.document.original_filename,
            'page_start': chunk.page_start,
            'page_end': chunk.page_end,
            'page_reference': page_ref,
            'text': chunk.text,
            'score': score,
        })

    # Sort by similarity score descending and cap at top_k
    passages.sort(key=lambda p: p['score'], reverse=True)
    passages = passages[:top_k]

    has_sufficient_evidence = len(passages) > 0
    top_score = passages[0]['score'] if passages else 0.0

    return {
        'has_sufficient_evidence': has_sufficient_evidence,
        'passages': passages,
        'top_score': top_score,
        'error': None
    }
