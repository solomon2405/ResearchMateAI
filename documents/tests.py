import io
import fitz  # PyMuPDF
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth.models import User
from django.conf import settings
from research_projects.models import ResearchProject
from documents.models import ResearchDocument, DocumentChunk
from documents.services.pdf_extractor import extract_pdf_pages, clean_academic_text
from documents.services.chunker import chunk_document_pages
from documents.services.ingestion import process_and_index_document

def create_test_pdf_bytes(title: str, body_text: str, page_count: int = 1) -> bytes:
    """Helper to generate a real, valid in-memory PDF using PyMuPDF."""
    doc = fitz.open()
    for p in range(page_count):
        page = doc.new_page()
        text = f"{title}\n\nSection {p+1}: Findings and Methodology\n{body_text}"
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class DocumentPipelineTests(TestCase):
    def setUp(self):
        self.user1 = User.objects.create_user(username='doc_owner', password='Password123!')
        self.user2 = User.objects.create_user(username='other_user', password='Password123!')

        self.client1 = Client()
        self.client1.login(username='doc_owner', password='Password123!')

        self.client2 = Client()
        self.client2.login(username='other_user', password='Password123!')

        self.project1 = ResearchProject.objects.create(
            owner=self.user1,
            name='Kidney Research Workspace',
            research_topic='Nephrology'
        )

        self.valid_pdf_bytes = create_test_pdf_bytes(
            title="Biomarker Discovery in Renal Disease",
            body_text=(
                "Serum creatinine and cystatin C remain standard biomarkers for assessing estimated "
                "glomerular filtration rate (eGFR). Recent single-cell RNA sequencing reveals novel podocyte injury markers. "
                "Our experiments demonstrate that early intervention reduces disease progression by 42.5% (p < 0.01). "
                "However, longitudinal cohorts are necessary to validate these results across diverse demographics."
            ),
            page_count=2
        )

    def test_clean_academic_text(self):
        dirty = "bio-\nmarkers evaluated across p < 0.01\r\n\r\nand ± 5.2 mg/dL.\x00"
        cleaned = clean_academic_text(dirty)
        self.assertIn("biomarkers", cleaned)
        self.assertIn("p < 0.01", cleaned)
        self.assertIn("± 5.2 mg/dL", cleaned)
        self.assertNotIn("\x00", cleaned)

    def test_extract_pdf_pages(self):
        # Save temporary PDF
        pdf_file = SimpleUploadedFile("paper1.pdf", self.valid_pdf_bytes, content_type="application/pdf")
        doc = ResearchDocument.objects.create(
            project=self.project1,
            uploader=self.user1,
            original_filename="paper1.pdf",
            file=pdf_file,
            file_size=len(self.valid_pdf_bytes),
            checksum="testchecksum1",
            processing_status="uploaded"
        )

        extracted = extract_pdf_pages(doc.file.path)
        self.assertEqual(extracted['total_pages'], 2)
        self.assertFalse(extracted['is_scanned'])
        self.assertIn("Biomarker Discovery", extracted['pages'][0]['text'])

    def test_chunking_preserves_pages_and_sentences(self):
        pages = [
            {'page_number': 1, 'text': "Sentence one on page one. Sentence two on page one."},
            {'page_number': 2, 'text': "Sentence three on page two. Sentence four on page two."}
        ]
        chunks = chunk_document_pages(pages, target_words=10, overlap_words=4)
        self.assertTrue(len(chunks) >= 1)
        for c in chunks:
            self.assertIn('page_start', c)
            self.assertIn('page_end', c)
            self.assertIn('text', c)
            self.assertIn('chunk_index', c)

    def test_scanned_pdf_detection(self):
        # Create an almost empty PDF (< 150 chars)
        empty_pdf = fitz.open()
        empty_pdf.new_page()
        empty_bytes = empty_pdf.tobytes()
        empty_pdf.close()

        pdf_file = SimpleUploadedFile("scanned.pdf", empty_bytes, content_type="application/pdf")
        doc = ResearchDocument.objects.create(
            project=self.project1,
            uploader=self.user1,
            original_filename="scanned.pdf",
            file=pdf_file,
            file_size=len(empty_bytes),
            checksum="scannedchecksum",
            processing_status="uploaded"
        )

        success = process_and_index_document(doc.pk)
        self.assertFalse(success)
        doc.refresh_from_db()
        self.assertEqual(doc.processing_status, 'failed')
        self.assertIn('OCR', doc.error_details)

    def test_full_document_ingestion_pipeline(self):
        pdf_file = SimpleUploadedFile("valid_biomarkers.pdf", self.valid_pdf_bytes, content_type="application/pdf")
        doc = ResearchDocument.objects.create(
            project=self.project1,
            uploader=self.user1,
            original_filename="valid_biomarkers.pdf",
            file=pdf_file,
            file_size=len(self.valid_pdf_bytes),
            checksum="unique_hash_123",
            processing_status="uploaded"
        )

        success = process_and_index_document(doc.pk)
        self.assertTrue(success)

        doc.refresh_from_db()
        self.assertEqual(doc.processing_status, 'indexed')
        self.assertEqual(doc.total_pages, 2)
        
        # Check chunks were created
        chunks = doc.chunks.all()
        self.assertTrue(chunks.count() >= 1)
        first_chunk = chunks.first()
        self.assertEqual(first_chunk.project_id, self.project1.id)
        self.assertIn("glomerular filtration rate", first_chunk.text)

    def test_upload_view_success_and_duplicate_rejection(self):
        pdf_file = SimpleUploadedFile("upload_test.pdf", self.valid_pdf_bytes, content_type="application/pdf")
        response = self.client1.post(
            reverse('document_upload', kwargs={'project_id': self.project1.pk}),
            {'files': [pdf_file]},
            follow=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(ResearchDocument.objects.filter(project=self.project1, original_filename="upload_test.pdf").exists())

        # Second upload of identical file to same project -> rejected by checksum
        pdf_file_dup = SimpleUploadedFile("upload_test.pdf", self.valid_pdf_bytes, content_type="application/pdf")
        dup_response = self.client1.post(
            reverse('document_upload', kwargs={'project_id': self.project1.pk}),
            {'files': [pdf_file_dup]}
        )
        self.assertContains(dup_response, "already been uploaded to this project")

    def test_private_document_security_and_isolation(self):
        pdf_file = SimpleUploadedFile("confidential_data.pdf", self.valid_pdf_bytes, content_type="application/pdf")
        doc = ResearchDocument.objects.create(
            project=self.project1,
            uploader=self.user1,
            original_filename="confidential_data.pdf",
            file=pdf_file,
            file_size=len(self.valid_pdf_bytes),
            checksum="private_checksum_999",
            processing_status="indexed"
        )

        # Owner can view private PDF
        owner_response = self.client1.get(reverse('document_serve', kwargs={'pk': doc.pk}))
        self.assertEqual(owner_response.status_code, 200)
        self.assertEqual(owner_response['Content-Type'], 'application/pdf')
        self.assertIn('confidential_data.pdf', owner_response['Content-Disposition'])

        # Unauthorized User 2 tries to download User 1's PDF -> must be 403 Forbidden
        unauthorized_response = self.client2.get(reverse('document_serve', kwargs={'pk': doc.pk}))
        self.assertEqual(unauthorized_response.status_code, 403)
