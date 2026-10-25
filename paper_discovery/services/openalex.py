"""
OpenAlex Academic Literature API Service Adapter.
Queries the OpenAlex Works API with query validation, filter mapping,
polite pool headers, pagination, timeouts, and normalization.
"""

import logging
from typing import Dict, Any, Optional
import httpx
from django.conf import settings
from .normalizers import normalize_openalex_work

logger = logging.getLogger(__name__)

class OpenAlexServiceError(Exception):
    """Exception raised when an OpenAlex query encounters a network, API or validation failure."""
    pass

class OpenAlexService:
    def __init__(self, api_url: Optional[str] = None, mailto: Optional[str] = None, timeout: float = 12.0):
        self.api_url = api_url or getattr(settings, 'OPENALEX_API_URL', 'https://api.openalex.org/works')
        self.mailto = mailto or getattr(settings, 'OPENALEX_MAILTO', 'researcher@example.com')
        self.timeout = timeout

    def search_works(
        self,
        query: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        sort_by: Optional[str] = None,
        page: int = 1,
        per_page: int = 10
    ) -> Dict[str, Any]:
        """
        Execute an academic literature search on OpenAlex.
        
        Args:
            query: Research keywords or topic.
            start_year: Minimum publication year (inclusive).
            end_year: Maximum publication year (inclusive).
            sort_by: Field to sort by ('relevance_score:desc', 'cited_by_count:desc', 'publication_date:desc').
            page: 1-indexed page number.
            per_page: Number of items per page (default: 10, max: 50).
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                'results': [],
                'total_count': 0,
                'page': page,
                'per_page': per_page,
                'total_pages': 0,
                'error': 'Please enter a search query or topic.'
            }

        params = {
            'search': clean_query,
            'page': max(1, page),
            'per_page': min(max(1, per_page), 50),
        }

        # Polite pool parameter
        if self.mailto:
            params['mailto'] = self.mailto

        # Filters
        filter_parts = []
        if start_year and end_year:
            if start_year <= end_year:
                filter_parts.append(f"publication_year:{start_year}-{end_year}")
        elif start_year:
            filter_parts.append(f"from_publication_date:{start_year}-01-01")
        elif end_year:
            filter_parts.append(f"to_publication_date:{end_year}-12-31")

        if filter_parts:
            params['filter'] = ",".join(filter_parts)

        # Sorting
        allowed_sorts = {
            'relevance': 'relevance_score:desc',
            'citations': 'cited_by_count:desc',
            'date_desc': 'publication_date:desc',
            'date_asc': 'publication_date:asc',
        }
        if sort_by and sort_by in allowed_sorts:
            params['sort'] = allowed_sorts[sort_by]

        headers = {
            'User-Agent': f'ResearchMateAI/1.0 (mailto:{self.mailto})',
            'Accept': 'application/json',
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self.api_url, params=params, headers=headers)
                
                if response.status_code == 429:
                    logger.warning("OpenAlex API rate limit hit.")
                    return {
                        'results': [],
                        'total_count': 0,
                        'page': page,
                        'per_page': per_page,
                        'total_pages': 0,
                        'error': 'OpenAlex search rate limit reached. Please wait a moment and try again.'
                    }
                
                response.raise_for_status()
                data = response.json()

        except httpx.TimeoutException:
            logger.error("OpenAlex API request timed out.")
            return {
                'results': [],
                'total_count': 0,
                'page': page,
                'per_page': per_page,
                'total_pages': 0,
                'error': 'Academic search provider timed out. Please try again with more specific terms.'
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAlex HTTP error: {e.response.status_code}")
            return {
                'results': [],
                'total_count': 0,
                'page': page,
                'per_page': per_page,
                'total_pages': 0,
                'error': f'Academic provider returned status {e.response.status_code}.'
            }
        except Exception as e:
            logger.error(f"Unexpected error querying OpenAlex: {e}")
            return {
                'results': [],
                'total_count': 0,
                'page': page,
                'per_page': per_page,
                'total_pages': 0,
                'error': 'Failed to connect to academic search provider. Please check network connection.'
            }

        # Parse and normalize works
        raw_works = data.get('results', [])
        meta = data.get('meta', {})
        total_count = meta.get('count', len(raw_works))
        
        normalized_results = []
        seen_ids = set()
        for work in raw_works:
            normalized = normalize_openalex_work(work)
            work_id = normalized['id']
            if work_id and work_id not in seen_ids:
                seen_ids.add(work_id)
                normalized_results.append(normalized)

        per_page_actual = params['per_page']
        total_pages = (total_count + per_page_actual - 1) // per_page_actual if total_count > 0 else 0

        return {
            'results': normalized_results,
            'total_count': total_count,
            'page': params['page'],
            'per_page': per_page_actual,
            'total_pages': total_pages,
            'error': None
        }
