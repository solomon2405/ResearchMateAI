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
    Distinguishes raw dense similarity from ranking features (keyword matches, overview boosts),
    enforces real evidence thresholds, and bounds the context token budget.
    
    Returns:
        {
            'has_sufficient_evidence': bool,
            'passages': List[Dict[str, Any]],
            'top_score': float,
            'top_dense_similarity': float,
            'error': Optional[str],
            'insufficient_reason': Optional[str]
        }
    """
    top_k = top_k or getattr(settings, 'RAG_TOP_K', 5)
    threshold = similarity_threshold if similarity_threshold is not None else getattr(settings, 'RAG_SIMILARITY_THRESHOLD', 0.35)
    max_context_tokens = getattr(settings, 'RAG_MAX_CONTEXT_TOKENS', 3000)

    clean_question = question.strip()
    if not clean_question:
        return {
            'has_sufficient_evidence': False,
            'passages': [],
            'top_score': 0.0,
            'top_dense_similarity': 0.0,
            'error': 'Question cannot be empty.',
            'insufficient_reason': 'Question cannot be empty.'
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

    # Detect overview, summary, or broad workspace review queries
    is_overview_query = bool(re.search(
        r'\b(summary|summarize|overview|all\s+papers|entire\s+workspace|review\s+of\s+papers|workspace\s+summary|literature\s+review)\b',
        clean_question,
        re.IGNORECASE
    ))

    # Search project-isolated vector store
    vector_store = get_project_vector_store(project_id)
    search_k = max(top_k * 3, 12)

    # Track scores per chunk: chunk_id -> {'dense': float, 'keyword_boost': float, 'overview_boost': float}
    chunk_scores: Dict[int, Dict[str, float]] = {}

    def get_or_create_record(cid: int) -> Dict[str, float]:
        if cid not in chunk_scores:
            chunk_scores[cid] = {'dense': 0.0, 'keyword_boost': 0.0, 'overview_boost': 0.0}
        return chunk_scores[cid]

    # 1. Primary vector search with raw query
    base_vector = embed_query(clean_question)
    base_hits = vector_store.search(base_vector, top_k=search_k)
    for cid, score in base_hits:
        rec = get_or_create_record(cid)
        rec['dense'] = max(rec['dense'], float(score))

    # 2. Context-augmented vector search using project topic keywords (only for overview/exploratory queries)
    if is_overview_query and topic_context:
        context_query = f"{clean_question} {topic_context}"
        context_vector = embed_query(context_query)
        context_hits = vector_store.search(context_vector, top_k=search_k)
        for cid, score in context_hits:
            rec = get_or_create_record(cid)
            # Weight context search lower than direct question similarity
            rec['dense'] = max(rec['dense'], float(score) * 0.85)

    # 3. Overview / Abstract expansion vector search
    if is_overview_query and topic_context:
        overview_query = f"abstract summary introduction overview {topic_context}"
        overview_vector = embed_query(overview_query)
        overview_hits = vector_store.search(overview_vector, top_k=search_k)
        for cid, score in overview_hits:
            rec = get_or_create_record(cid)
            rec['dense'] = max(rec['dense'], float(score) * 0.80)


    # 4. Explicit overview ranking boost for introductory chunks (without fabricating dense score)
    if is_overview_query:
        abstract_chunks = list(DocumentChunk.objects.filter(
            project_id=project_id,
            document__processing_status='indexed',
            chunk_index__in=[0, 1]
        ).values_list('id', flat=True))
        for cid in abstract_chunks:
            rec = get_or_create_record(cid)
            rec['overview_boost'] = 0.08

    # 5. Lexical keyword assistance: record explicit lexical feature boost
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
                rec = get_or_create_record(cid)
                rec['keyword_boost'] = max(rec['keyword_boost'], 0.05)

    if not chunk_scores:
        return {
            'has_sufficient_evidence': False,
            'passages': [],
            'top_score': 0.0,
            'top_dense_similarity': 0.0,
            'error': None,
            'insufficient_reason': 'No candidate passages retrieved in active project.'
        }

    # Evaluate candidate chunks against genuine dense similarity threshold
    eligible_cids = []
    for cid, scores in chunk_scores.items():
        dense = scores['dense']
        total_rank_score = dense + scores['keyword_boost'] + scores['overview_boost']
        scores['final_rank_score'] = total_rank_score
        
        # Dense threshold gating: require meaningful dense similarity
        if dense >= threshold or (is_overview_query and dense >= threshold * 0.75):
            eligible_cids.append(cid)

    if not eligible_cids:
        max_dense = max(s['dense'] for s in chunk_scores.values()) if chunk_scores else 0.0
        return {
            'has_sufficient_evidence': False,
            'passages': [],
            'top_score': max_dense,
            'top_dense_similarity': max_dense,
            'error': None,
            'insufficient_reason': (
                f"Retrieved passages did not meet similarity threshold "
                f"({max_dense:.2f} < {threshold:.2f})."
            )
        }

    # Fetch corresponding chunks from DB ensuring project_id match and indexed status
    matching_chunks = DocumentChunk.objects.filter(
        id__in=eligible_cids,
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

        scores = chunk_scores[chunk.id]
        final_rank_score = scores.get('final_rank_score', scores['dense'])
        page_ref = f"{chunk.page_start}" if chunk.page_start == chunk.page_end else f"{chunk.page_start}-{chunk.page_end}"

        passages.append({
            'chunk_id': chunk.id,
            'document_id': chunk.document_id,
            'filename': chunk.document.original_filename,
            'page_start': chunk.page_start,
            'page_end': chunk.page_end,
            'page_reference': page_ref,
            'text': chunk.text,
            'score': round(final_rank_score, 4),
            'dense_similarity': round(scores['dense'], 4),
            'keyword_boost': round(scores['keyword_boost'], 4),
            'overview_boost': round(scores['overview_boost'], 4),
        })

    # Sort by final ranking score descending
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

    selected_passages = (diverse_passages + remaining_passages)[:top_k]

    # Enforce token context budget (approx 1.3 tokens per word)
    budgeted_passages = []
    accumulated_tokens = 0
    for p in selected_passages:
        est_tokens = int(len(p['text'].split()) * 1.3)
        if accumulated_tokens + est_tokens <= max_context_tokens or not budgeted_passages:
            budgeted_passages.append(p)
            accumulated_tokens += est_tokens
        else:
            break

    has_sufficient_evidence = len(budgeted_passages) > 0
    top_score = budgeted_passages[0]['score'] if budgeted_passages else 0.0
    top_dense = budgeted_passages[0]['dense_similarity'] if budgeted_passages else 0.0

    return {
        'has_sufficient_evidence': has_sufficient_evidence,
        'passages': budgeted_passages,
        'top_score': top_score,
        'top_dense_similarity': top_dense,
        'error': None,
        'insufficient_reason': None if has_sufficient_evidence else "Insufficient matching evidence."
    }
