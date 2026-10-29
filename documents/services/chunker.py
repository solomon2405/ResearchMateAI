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
    and protecting decimals and scientific notation.
    """
    if not text:
        return []

    # Protect common academic abbreviations
    abbreviations = {
        r'\bet al\.': 'et_al_DOT',
        r'\be\.g\.': 'e_g_DOT',
        r'\bi\.e\.': 'i_e_DOT',
        r'\bcf\.': 'cf_DOT',
        r'\bca\.': 'ca_DOT',
        r'\bvs\.': 'vs_DOT',
        r'\bFig\.\s*(\d+)': r'Fig_DOT_\1',
        r'\bFigs\.\s*(\d+)': r'Figs_DOT_\1',
        r'\bEq\.\s*(\d+)': r'Eq_DOT_\1',
        r'\bEqs\.\s*(\d+)': r'Eqs_DOT_\1',
        r'\bRef\.\s*(\d+)': r'Ref_DOT_\1',
        r'\bRefs\.\s*(\d+)': r'Refs_DOT_\1',
        r'\bDr\.': 'Dr_DOT',
        r'\bProf\.': 'Prof_DOT',
        r'\bVol\.\s*(\d+)': r'Vol_DOT_\1',
        r'\bNo\.\s*(\d+)': r'No_DOT_\1',
        r'\bp\.\s*(\d+)': r'p_DOT_\1',
        r'\bpp\.\s*(\d+)': r'pp_DOT_\1',
    }
    
    masked = text
    for pattern, placeholder in abbreviations.items():
        masked = re.sub(pattern, placeholder, masked, flags=re.IGNORECASE)

    # Protect decimals like 3.14 or p < 0.01 by temporarily masking digit.digit
    masked = re.sub(r'(\d)\.(\d)', r'\1_DECIMAL_\2', masked)

    # Split by standard sentence terminators (. ! ?) followed by whitespace or quote/bracket
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[\u201c\u2018])', masked)

    # Restore placeholders
    sentences = []
    for s in raw_sentences:
        restored = s.replace('_DECIMAL_', '.')
        for _, placeholder in abbreviations.items():
            base_token = placeholder.split('_')[0]
            restored = restored.replace(f"{base_token}_DOT", f"{base_token}.")
        # Clean any remaining internal DOT tags
        restored = re.sub(r'_DOT_?', '.', restored)
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
    Safely bounds chunk sizes even when confronted with unpunctuated or pathological text.
    
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
                if not words:
                    continue
                # Pathological sentence guard: if a single sentence exceeds target_words,
                # break it into bounded word slices to protect embedding context
                if len(words) > target_words:
                    step = max(50, target_words - overlap_words)
                    for start_idx in range(0, len(words), step):
                        sub_slice = words[start_idx:start_idx + target_words]
                        sentence_items.append({
                            'text': " ".join(sub_slice),
                            'page': page_num,
                            'word_count': len(sub_slice)
                        })
                else:
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

        # Finalize this chunk when target word count is reached or at last sentence
        if current_word_count >= target_words or idx == len(sentence_items) - 1:
            chunk_text = " ".join([s['text'] for s in current_sentences]).strip()
            page_start = min(s['page'] for s in current_sentences)
            page_end = max(s['page'] for s in current_sentences)

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

            # Safety check: if overlap would retain entire chunk, advance by at least one item
            if len(overlap_sentences) == len(current_sentences) and len(overlap_sentences) > 1:
                overlap_sentences.pop(0)
                overlap_count = sum(s['word_count'] for s in overlap_sentences)

            current_sentences = overlap_sentences
            current_word_count = overlap_count

        idx += 1

    return chunks
