"""
Citation Validator for ResearchMate AI.
Ensures that assistant answers cite only real, retrieved passages belonging
to authorized documents in the active project, and maps citations to database records.
"""

import re
from typing import List, Dict, Any, Tuple

def extract_and_validate_citations(
    raw_answer: str,
    passages: List[Dict[str, Any]],
    project_id: int
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Validate source citations in the raw LLM answer against the supplied passages.
    Cross-verifies chunks against project database records, removes hallucinated tags,
    and explicitly marks whether citations were explicitly cited in text or are general references.
    
    Args:
        raw_answer: Text returned by LLM (may contain [Source 1], [Source 2], etc.).
        passages: List of retrieved evidence dicts containing 'chunk_id', 'document_id', 'page_reference'.
        project_id: Current project ID for security verification.
        
    Returns:
        (sanitized_answer, valid_citations_list)
        where each citation has:
            {
                'chunk_id': int,
                'document_id': int,
                'page_reference': str,
                'excerpt': str,
                'similarity_score': float,
                'filename': str,
                'is_explicit': bool
            }
    """
    if not passages:
        # If no passages were retrieved, remove any hallucinated citations
        cleaned = re.sub(r'\[Source\s*\d+\]', '', raw_answer)
        return cleaned.strip(), []

    from documents.models import DocumentChunk
    # Cross-verify chunk IDs against actual project chunks in DB
    supplied_cids = [p['chunk_id'] for p in passages if 'chunk_id' in p]
    verified_db_cids = set(
        DocumentChunk.objects.filter(
            id__in=supplied_cids,
            project_id=project_id,
            document__processing_status='indexed'
        ).values_list('id', flat=True)
    )

    # Filter passages to only those verified in DB
    verified_passages = [p for p in passages if p['chunk_id'] in verified_db_cids]
    if not verified_passages:
        cleaned = re.sub(r'\[Source\s*\d+\]', '', raw_answer)
        return cleaned.strip(), []

    # Map 1-based source indices (e.g., [Source 1]) to passage
    source_index_map = {i + 1: passage for i, passage in enumerate(verified_passages)}
    
    # Find all source mentions like [Source 1], [Source 2]
    found_matches = re.findall(r'\[Source\s*(\d+)\]', raw_answer, flags=re.IGNORECASE)
    
    cited_indices = set()
    for m in found_matches:
        try:
            cited_indices.add(int(m))
        except ValueError:
            pass

    valid_citations = []
    seen_chunks = set()

    # If the LLM explicitly cited sources:
    if cited_indices:
        for idx in sorted(cited_indices):
            if idx in source_index_map:
                p = source_index_map[idx]
                chunk_id = p['chunk_id']
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    valid_citations.append({
                        'chunk_id': chunk_id,
                        'document_id': p['document_id'],
                        'page_reference': p['page_reference'],
                        'excerpt': p['text'][:350] + ("..." if len(p['text']) > 350 else ""),
                        'similarity_score': p.get('score'),
                        'filename': p['filename'],
                        'is_explicit': True
                    })

        # Remove any hallucinated source references that exceed supplied passage count
        def replace_source_tag(match):
            idx = int(match.group(1))
            if idx in source_index_map:
                return f"[Source {idx}]"
            return ""  # Safely strip fabricated reference

        sanitized_answer = re.sub(r'\[Source\s*(\d+)\]', replace_source_tag, raw_answer, flags=re.IGNORECASE)
    else:
        # If the LLM synthesized without explicit tags, record top relevant passages
        # with is_explicit=False so researchers know they are background references rather than exact cited claims
        for p in verified_passages[:3]:
            chunk_id = p['chunk_id']
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                valid_citations.append({
                    'chunk_id': chunk_id,
                    'document_id': p['document_id'],
                    'page_reference': p['page_reference'],
                    'excerpt': p['text'][:350] + ("..." if len(p['text']) > 350 else ""),
                    'similarity_score': p.get('score'),
                    'filename': p['filename'],
                    'is_explicit': False
                })
        sanitized_answer = raw_answer

    return sanitized_answer.strip(), valid_citations
