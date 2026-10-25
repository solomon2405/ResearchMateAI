from django.db import models
from django.contrib.auth.models import User

class ResearchProject(models.Model):
    """
    A workspace where a student organizes research papers,
    PDF documents, and dedicated RAG chat conversations.
    """
    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='research_projects',
        help_text="The authenticated user who owns this research project."
    )
    name = models.CharField(
        max_length=255,
        help_text="Descriptive title of the research project."
    )
    research_topic = models.CharField(
        max_length=255,
        help_text="Core topic or keywords guiding this project."
    )
    description = models.TextField(
        blank=True,
        help_text="Optional project background, methodology, or objectives."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = "Research Project"
        verbose_name_plural = "Research Projects"

    def __str__(self):
        return f"{self.name} ({self.owner.username})"

    @property
    def document_count(self):
        return self.documents.count()

    @property
    def indexed_document_count(self):
        return self.documents.filter(processing_status='indexed').count()
