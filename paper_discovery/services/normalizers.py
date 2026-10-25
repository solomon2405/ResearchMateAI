"""
Academic search result normalizers.
Reconstructs inverted indexes (e.g. OpenAlex) into clean text,
cleans author lists, formats venue details, and ensures consistent metadata schema.
"""

from typing import Dict, Any, List, Optional

def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    Reconstruct an abstract from OpenAlex's abstract_inverted_index.
    The index maps words to list of 0-based token positions.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return "Abstract unavailable."

    try:
        # Determine total tokens by max position
        position_word_map = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                position_word_map[pos] = word

        if not position_word_map:
            return "Abstract unavailable."

        max_pos = max(position_word_map.keys())
        words = [position_word_map.get(i, '') for i in range(max_pos + 1)]
        abstract_text = ' '.join(filter(None, words)).strip()

        return abstract_text if abstract_text else "Abstract unavailable."
    except Exception:
        return "Abstract unavailable."


def normalize_openalex_work(work: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an OpenAlex JSON work object into a clean, uniform dictionary.
    Never invents data or placeholders.
    """
    title = work.get('title') or "Untitled Academic Work"
    
    # Extract authors
    authors = []
    for authorship in work.get('authorships', []):
        author_obj = authorship.get('author', {})
        display_name = author_obj.get('display_name')
        if display_name:
            authors.append(display_name)
    
    publication_year = work.get('publication_year')
    
    # Primary venue / host organization
    primary_location = work.get('primary_location') or {}
    source = primary_location.get('source') or {}
    venue = source.get('display_name') or "Academic Venue / Repository"
    
    # Reconstruct abstract
    abstract = reconstruct_abstract(work.get('abstract_inverted_index'))
    
    # DOI and landing URLs
    doi = work.get('doi')
    landing_page_url = primary_location.get('landing_page_url') or doi
    
    # PDF URL check (only if explicitly available)
    pdf_url = primary_location.get('pdf_url')
    open_access_info = work.get('open_access') or {}
    is_oa = open_access_info.get('is_oa', False)
    oa_status = open_access_info.get('oa_status', 'closed')
    oa_url = open_access_info.get('oa_url')
    
    if not pdf_url and oa_url and oa_url.lower().endswith('.pdf'):
        pdf_url = oa_url

    citation_count = work.get('cited_by_count', 0)
    openalex_id = work.get('id', '')

    return {
        'id': openalex_id,
        'title': title,
        'authors': authors,
        'authors_display': ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "") if authors else "Unknown Authors",
        'publication_year': publication_year or "N/A",
        'venue': venue,
        'abstract': abstract,
        'doi': doi,
        'landing_page_url': landing_page_url,
        'pdf_url': pdf_url,
        'is_oa': is_oa,
        'oa_status': oa_status,
        'citation_count': citation_count,
        'source_provider': 'OpenAlex',
    }
