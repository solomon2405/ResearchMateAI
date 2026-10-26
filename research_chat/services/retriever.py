"""
Project-Scoped Evidence Retriever for RAG.
Queries only the authorized project's indexed chunks, evaluates similarity
against configured thresholds, and constructs a bounded evidence package.
Includes contextual query expansion, overview/abstract prioritization, and hybrid retrieval.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from django.conf import settings
from documents.models import DocumentChunk
from .embeddings import embed_query
from .vector_store import get_project_vector_store

logger = logging.getLogger(__name__)

def retrieve_project_evidence(
    project_id: int,
    question: str,
    top_k: int = None,
    similarity_threshold: float = None,
    project = None
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
    threshold = similarity_threshold if similarity_threshold is not None else getattr(settings, 'RAG_SIMILARITY_THRESHOLD', 0.20)

    clean_question = question.strip()
    if not clean_question:
        return {
            'has_sufficient_evidence': False,
            'passages': [],
            'top_score': 0.0,
            'error': 'Question cannot be empty.'
        }

    # Retrieve project model if not provided
    if project is None:
        try:
            from research_projects.models import ResearchProject
            project = ResearchProject.objects.filter(pk=project_id).first()
        except Exception:
            project = None

    # Topic & workspace context
    topic_context = ""
    if project:
        parts = [p for p in [getattr(project, 'name', ''), getattr(project, 'research_topic', '')] if p]
        topic_context = " ".join(parts).strip()

    # Detect overview, summary, abstract, or broad exploratory queries
    is_overview_query = bool(re.search(
        r'\b(summa\w*|abstract\w*|overview|about|what\s+is|what\s+are|tell\s+me|intro\w*|background|explain|stack|baseline|technolog\w*)\b',
        clean_question,
        re.IGNORECASE
    ))

    # Search project-isolated vector store
    vector_store = get_project_vector_store(project_id)
    search_k = max(top_k * 3, 12)
    merged_hits: Dict[int, float] = {}

    # 1. Primary vector search with raw query
    base_vector = embed_query(clean_question)
    base_hits = vector_store.search(base_vector, top_k=search_k)
    for cid, score in base_hits:
        merged_hits[cid] = max(merged_hits.get(cid, 0.0), score)

    # 2. Context-augmented vector search using project topic keywords
    if topic_context:
        context_query = f"{clean_question} {topic_context}"
        context_vector = embed_query(context_query)
        context_hits = vector_store.search(context_vector, top_k=search_k)
        for cid, score in context_hits:
            merged_hits[cid] = max(merged_hits.get(cid, 0.0), score)

    # 3. Overview / Abstract expansion vector search
    if is_overview_query and topic_context:
        overview_query = f"abstract summary introduction overview {topic_context}"
        overview_vector = embed_query(overview_query)
        overview_hits = vector_store.search(overview_vector, top_k=search_k)
        for cid, score in overview_hits:
            merged_hits[cid] = max(merged_hits.get(cid, 0.0), score)

    # 4. For overview/abstract queries, prioritize page-1 / introductory chunks across documents
    if is_overview_query:
        abstract_chunks = DocumentChunk.objects.filter(
            project_id=project_id,
            document__processing_status='indexed',
            chunk_index__in=[0, 1]
        ).values_list('id', flat=True)
        for cid in abstract_chunks:
            current_score = merged_hits.get(cid, 0.0)
            merged_hits[cid] = max(current_score, 0.35)

    # 5. Hybrid keyword assistance for specific terms in question
    stop_words = {'what', 'when', 'where', 'which', 'about', 'tell', 'this', 'that', 'from', 'with', 'have', 'been', 'paper', 'papers', 'choose', 'appropriate', 'basically'}
    significant_words = [
        w for w in re.findall(r'\b[A-Za-z0-9_-]{4,}\b', clean_question.lower())
        if w not in stop_words
    ]
    if significant_words:
        keyword_chunks = DocumentChunk.objects.filter(
            project_id=project_id,
            document__processing_status='indexed'
        )
        for word in significant_words[:4]:
            matched_ids = list(keyword_chunks.filter(text__icontains=word).values_list('id', flat=True)[:6])
            for cid in matched_ids:
                merged_hits[cid] = max(merged_hits.get(cid, 0.0), 0.30)

    if not merged_hits:
        return {
            'has_sufficient_evidence': False,
            'passages': [],
            'top_score': 0.0,
            'error': None
        }

    # Extract chunk IDs passing calibrated similarity threshold
    hit_map = {chunk_id: score for chunk_id, score in merged_hits.items() if score >= threshold}

    if not hit_map:
        best_score = max(merged_hits.values()) if merged_hits else 0.0
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

    # Sort by similarity score descending
    passages.sort(key=lambda p: p['score'], reverse=True)

    # When multiple documents exist in project, ensure diverse document representation
    diverse_passages = []
    seen_docs = set()
    remaining_passages = []

    for p in passages:
        if p['document_id'] not in seen_docs and len(diverse_passages) < top_k:
            seen_docs.add(p['document_id'])
            diverse_passages.append(p)
        else:
            remaining_passages.append(p)

    passages = (diverse_passages + remaining_passages)[:top_k]

    has_sufficient_evidence = len(passages) > 0
    top_score = passages[0]['score'] if passages else 0.0

    return {
        'has_sufficient_evidence': has_sufficient_evidence,
        'passages': passages,
        'top_score': top_score,
        'error': None
    }
