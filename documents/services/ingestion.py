"""
Document Ingestion Pipeline Service.
Extracts text via PyMuPDF, chunks passages, computes sentence embeddings,
stores DocumentChunks in the database, and indexes vectors in the project's vector store.
"""

import logging
from django.db import transaction
from django.conf import settings
from documents.models import ResearchDocument, DocumentChunk
from .pdf_extractor import extract_pdf_pages, PDFExtractionError
from .chunker import chunk_document_pages
from research_chat.services.embeddings import embed_texts
from research_chat.services.vector_store import get_project_vector_store

logger = logging.getLogger(__name__)

def process_and_index_document(document_id: int) -> bool:
    """
    Execute full ingestion pipeline for a ResearchDocument:
    1. Extract pages (PyMuPDF)
    2. Validate extractable text (check OCR requirement)
    3. Semantic chunking with page tracking
    4. Local sentence embeddings
    5. Save DocumentChunks
    6. Index in project-isolated vector store
    7. Update processing_status to 'indexed'
    
    Returns:
        True if indexing succeeded, False otherwise.
    """
    try:
        doc = ResearchDocument.objects.select_related('project').get(pk=document_id)
    except ResearchDocument.DoesNotExist:
        logger.error(f"Document ID {document_id} not found for ingestion.")
        return False

    try:
        # Stage 1: Extraction
        doc.processing_status = 'extracting'
        doc.error_details = ''
        doc.save(update_fields=['processing_status', 'error_details'])

        file_path = doc.file.path
        extraction_result = extract_pdf_pages(file_path)

        doc.total_pages = extraction_result['total_pages']
        doc.save(update_fields=['total_pages'])

        # Check for scanned or empty document
        if extraction_result['is_scanned']:
            doc.processing_status = 'failed'
            doc.error_details = (
                "Document appears to be scanned or contains insufficient extractable text. "
                "An OCR (Optical Character Recognition) pipeline is required to process this file."
            )
            doc.save(update_fields=['processing_status', 'error_details'])
            logger.warning(f"Document {doc.id} ({doc.original_filename}) flagged as scanned.")
            return False

        # Stage 2: Chunking
        doc.processing_status = 'chunking'
        doc.save(update_fields=['processing_status'])

        raw_chunks = chunk_document_pages(
            extraction_result['pages'],
            target_words=500,
            overlap_words=100
        )

        if not raw_chunks:
            doc.processing_status = 'failed'
            doc.error_details = "Could not extract readable semantic paragraphs from this PDF."
            doc.save(update_fields=['processing_status', 'error_details'])
            return False

        # Stage 3: Embedding
        doc.processing_status = 'embedding'
        doc.save(update_fields=['processing_status'])

        chunk_texts = [c['text'] for c in raw_chunks]
        embeddings = embed_texts(chunk_texts)

        # Stage 4 & 5: Persist chunks and update vector store under project lock
        vector_store = get_project_vector_store(doc.project_id)
        with vector_store.get_lock():
            # Atomically save DB chunks and mark indexed
            with transaction.atomic():
                doc.chunks.all().delete()

                chunk_objects = []
                for i, c in enumerate(raw_chunks):
                    chunk_obj = DocumentChunk(
                        project=doc.project,
                        document=doc,
                        chunk_index=c['chunk_index'],
                        page_start=c['page_start'],
                        page_end=c['page_end'],
                        text=c['text'],
                        embedding_model=getattr(settings, 'EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2'),
                        index_version='2.0',
                    )
                    chunk_objects.append(chunk_obj)

                created_chunks = DocumentChunk.objects.bulk_create(chunk_objects)
                doc.processing_status = 'indexed'
                doc.error_details = ''
                doc.save(update_fields=['processing_status', 'error_details'])

            # Add to project FAISS vector store
            try:
                vector_store.add_chunks(created_chunks, embeddings)
            except Exception as idx_err:
                logger.exception(f"Vector indexing failed for document {doc.id}: {idx_err}")
                with transaction.atomic():
                    doc.processing_status = 'failed'
                    doc.error_details = f"Vector indexing failed: {str(idx_err)}"
                    doc.save(update_fields=['processing_status', 'error_details'])
                    doc.chunks.all().delete()
                # Restore index consistency from verified database records
                vector_store.rebuild_from_database()
                return False

        logger.info(f"Document {doc.id} ({doc.original_filename}) successfully indexed ({len(created_chunks)} chunks).")
        return True

    except PDFExtractionError as e:
        logger.error(f"PDF extraction error for doc {document_id}: {e}")
        doc.processing_status = 'failed'
        doc.error_details = f"PDF extraction error: {str(e)}"
        doc.save(update_fields=['processing_status', 'error_details'])
        return False
    except Exception as e:
        logger.exception(f"Unexpected ingestion failure for doc {document_id}: {e}")
        doc.processing_status = 'failed'
        doc.error_details = "An unexpected error occurred while processing and indexing this document."
        doc.save(update_fields=['processing_status', 'error_details'])
        return False
