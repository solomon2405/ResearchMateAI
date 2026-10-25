"""
Academic Document Text Chunker.
Splits extracted document pages into overlapping semantic passages
while preserving paragraph boundaries, sentence integrity, and page numbers.
"""

import re
from typing import List, Dict, Any

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences while respecting common academic abbreviations
    such as 'et al.', 'Fig.', 'e.g.', 'i.e.', 'Dr.', etc.
    """
    # Replace common abbreviations with a placeholder
    abbreviations = {
        r'\bet al\.': 'et_al_DOT',
        r'\be\.g\.': 'e_g_DOT',
        r'\bi\.e\.': 'i_e_DOT',
        r'\bFig\.': 'Fig_DOT',
        r'\bEq\.': 'Eq_DOT',
        r'\bRef\.': 'Ref_DOT',
        r'\bvs\.': 'vs_DOT',
        r'\bDr\.': 'Dr_DOT',
        r'\bProf\.': 'Prof_DOT',
        r'\bVol\.': 'Vol_DOT',
        r'\bNo\.': 'No_DOT',
        r'\bp\.': 'p_DOT',
        r'\bpp\.': 'pp_DOT',
    }
    
    masked = text
    for pattern, placeholder in abbreviations.items():
        masked = re.sub(pattern, placeholder, masked, flags=re.IGNORECASE)

    # Split by standard sentence terminators followed by whitespace or uppercase
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])', masked)

    # Restore placeholders
    sentences = []
    for s in raw_sentences:
        restored = s
        for _, placeholder in abbreviations.items():
            original = placeholder.replace('_DOT', '.').replace('_', '.')
            restored = restored.replace(placeholder, original)
        cleaned = restored.strip()
        if cleaned:
            sentences.append(cleaned)

    return sentences


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    target_words: int = 500,
    overlap_words: int = 100
) -> List[Dict[str, Any]]:
    """
    Chunk extracted pages into semantic passages.
    
    Args:
        pages: List of {'page_number': int, 'text': str}
        target_words: Desired maximum word count per chunk (~500-700 tokens)
        overlap_words: Number of trailing words carried over to next chunk
        
    Returns:
        List of chunks with metadata:
        [{
            'chunk_index': int,
            'page_start': int,
            'page_end': int,
            'text': str,
            'word_count': int
        }]
    """
    if not pages:
        return []

    # Flatten sentences with their source page number
    sentence_items = []
    for page_info in pages:
        page_num = page_info['page_number']
        page_text = page_info.get('text', '')
        if not page_text:
            continue
        
        # Split page text by paragraphs first
        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
        for para in paragraphs:
            para_sentences = split_into_sentences(para)
            for sentence in para_sentences:
                words = sentence.split()
                if words:
                    sentence_items.append({
                        'text': sentence,
                        'page': page_num,
                        'word_count': len(words)
                    })

    if not sentence_items:
        return []

    chunks = []
    chunk_index = 0
    current_sentences = []
    current_word_count = 0

    idx = 0
    while idx < len(sentence_items):
        item = sentence_items[idx]
        current_sentences.append(item)
        current_word_count += item['word_count']

        # If we have reached or exceeded target word count, finalize this chunk
        if current_word_count >= target_words or idx == len(sentence_items) - 1:
            chunk_text = " ".join([s['text'] for s in current_sentences]).strip()
            page_start = current_sentences[0]['page']
            page_end = current_sentences[-1]['page']

            chunks.append({
                'chunk_index': chunk_index,
                'page_start': page_start,
                'page_end': page_end,
                'text': chunk_text,
                'word_count': current_word_count
            })
            chunk_index += 1

            if idx == len(sentence_items) - 1:
                break

            # Overlap: keep trailing sentences that approximate overlap_words
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_sentences):
                if overlap_count + s['word_count'] <= overlap_words or not overlap_sentences:
                    overlap_sentences.insert(0, s)
                    overlap_count += s['word_count']
                else:
                    break

            current_sentences = overlap_sentences
            current_word_count = overlap_count

        idx += 1

    return chunks
