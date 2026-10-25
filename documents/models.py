from django.db import models
from django.contrib.auth.models import User
from research_projects.models import ResearchProject

class ResearchDocument(models.Model):
    """
    Metadata and private storage reference for a PDF paper
    uploaded to a specific research project.
    """
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('queued', 'Queued for Ingestion'),
        ('extracting', 'Extracting Text'),
        ('chunking', 'Chunking Passages'),
        ('embedding', 'Generating Embeddings'),
        ('indexed', 'Indexed & Ready for RAG'),
        ('failed', 'Processing Failed'),
    ]

    project = models.ForeignKey(
        ResearchProject,
        on_delete=models.CASCADE,
        related_name='documents'
    )
    uploader = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='uploaded_documents'
    )
    original_filename = models.CharField(max_length=255)
    file = models.FileField(upload_to='private_documents/%Y/%m/')
    file_size = models.BigIntegerField(default=0, help_text="Size in bytes")
    checksum = models.CharField(max_length=64, db_index=True, help_text="SHA-256 hash")
    detected_mime = models.CharField(max_length=100, default='application/pdf')
    processing_status = models.CharField(
        max_length=30,
        choices=STATUS_CHOICES,
        default='uploaded',
        db_index=True
    )
    error_details = models.TextField(blank=True)
    total_pages = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Research Document"
        verbose_name_plural = "Research Documents"
        constraints = [
            models.UniqueConstraint(
                fields=['project', 'checksum'],
                name='unique_document_per_project_checksum'
            )
        ]

    def __str__(self):
        return f"{self.original_filename} [{self.get_processing_status_display()}]"

    @property
    def file_size_display(self):
        """Human-readable file size."""
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"


class DocumentChunk(models.Model):
    """
    A discrete textual chunk extracted from a document, indexed
    for vector search and citation grounding.
    """
    project = models.ForeignKey(
        ResearchProject,
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    document = models.ForeignKey(
        ResearchDocument,
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    chunk_index = models.PositiveIntegerField()
    page_start = models.PositiveIntegerField()
    page_end = models.PositiveIntegerField()
    text = models.TextField()
    vector_id = models.IntegerField(null=True, blank=True, help_text="Index position in FAISS vector store")
    embedding_model = models.CharField(max_length=100, default='all-MiniLM-L6-v2')
    index_version = models.CharField(max_length=50, default='1.0')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['document', 'chunk_index']
        indexes = [
            models.Index(fields=['project', 'document']),
        ]

    def __str__(self):
        return f"Chunk #{self.chunk_index} of {self.document.original_filename} (p. {self.page_start})"
