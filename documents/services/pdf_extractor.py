"""
PyMuPDF Text Extractor for Academic Research Papers.
Extracts text page-by-page, retains page numbers, detects document titles,
checks for scanned/empty pages, and cleans extraction artifacts without
corrupting scientific notation, units, mathematical symbols, or negations.
"""

import re
import logging
from typing import Dict, Any, List
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

class PDFExtractionError(Exception):
    pass

class ScannedDocumentError(PDFExtractionError):
    pass

def clean_academic_text(text: str) -> str:
    """
    Clean extraction artifacts while preserving scientific content.
    - Fixes hyphenated line breaks (e.g. 'bio-\\nmarkers' -> 'biomarkers')
    - Replaces non-breaking spaces, zero-width spaces, and soft hyphens
    - Preserves negative signs, mathematical symbols (+, -, ±, ≤, ≥, <, >, =), decimals, and units
    """
    if not text:
        return ""

    # Replace zero-width and non-breaking spaces, soft hyphens
    cleaned = text.replace('\xa0', ' ').replace('\u200b', '').replace('\xad', '')

    # Replace null and non-printable control characters (except newline, tab)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)

    # Rejoin words split across line breaks with a hyphen
    cleaned = re.sub(r'(\b[a-zA-Z]{2,})-\n([a-zA-Z]{2,}\b)', r'\1\2', cleaned)

    # Normalize carriage returns
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')

    # Collapse multiple horizontal whitespace while preserving paragraph breaks
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


def extract_pdf_pages(file_path: str) -> Dict[str, Any]:
    """
    Extract structured text from a local PDF file with multi-column layout sorting.
    
    Returns:
        {
            'title': str,
            'total_pages': int,
            'pages': [{'page_number': int, 'text': str, 'char_count': int}],
            'total_char_count': int,
            'is_scanned': bool,
            'warning': Optional[str]
        }
    """
    doc = None
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise PDFExtractionError(f"Failed to open PDF document: {str(e)}")

    try:
        if doc.is_encrypted:
            raise PDFExtractionError("PDF is encrypted or password-protected.")

        total_pages = doc.page_count
        if total_pages == 0:
            raise PDFExtractionError("Document contains zero pages.")

        # Try metadata title
        doc_title = doc.metadata.get('title', '').strip() if doc.metadata else ''
        
        pages = []
        total_chars = 0

        for page_idx in range(total_pages):
            page = doc.load_page(page_idx)
            # Use sort=True for correct reading order in multi-column research papers
            try:
                raw_text = page.get_text("text", sort=True) or ""
            except Exception:
                raw_text = page.get_text("text") or ""

            cleaned = clean_academic_text(raw_text)
            char_count = len(cleaned)
            total_chars += char_count

            pages.append({
                'page_number': page_idx + 1,  # 1-indexed
                'text': cleaned,
                'char_count': char_count
            })

        # If title wasn't in metadata, extract first prominent line from page 1
        if not doc_title and pages and pages[0]['text']:
            first_page_lines = [line.strip() for line in pages[0]['text'].split('\n') if line.strip()]
            if first_page_lines:
                doc_title = first_page_lines[0][:150]

        # Detect scanned or image-only documents:
        # If average characters per page is below 100, document is likely scanned / image-based
        avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0
        is_scanned = total_chars < 150 or avg_chars_per_page < 100

        warning = None
        if is_scanned:
            warning = (
                "This document appears to be scanned or contains very little extractable text. "
                "An OCR (Optical Character Recognition) pipeline is required to index the contents."
            )

        return {
            'title': doc_title,
            'total_pages': total_pages,
            'pages': pages,
            'total_char_count': total_chars,
            'is_scanned': is_scanned,
            'warning': warning,
        }
    finally:
        if doc:
            doc.close()
