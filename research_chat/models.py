from django.db import models
from django.contrib.auth.models import User
from research_projects.models import ResearchProject
from documents.models import ResearchDocument, DocumentChunk

class ChatConversation(models.Model):
    """
    A persistent chat thread within a research project, maintaining
    isolated question-answering history grounded in the project's documents.
    """
    project = models.ForeignKey(
        ResearchProject,
        on_delete=models.CASCADE,
        related_name='conversations'
    )
    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='conversations'
    )
    title = models.CharField(max_length=255, default='New Research Chat')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = "Chat Conversation"
        verbose_name_plural = "Chat Conversations"

    def __str__(self):
        return f"{self.title} ({self.project.name})"


class ChatMessage(models.Model):
    """
    An individual message (user question or assistant evidence-grounded answer)
    within a research chat conversation.
    """
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]

    conversation = models.ForeignKey(
        ChatConversation,
        on_delete=models.CASCADE,
        related_name='messages'
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    token_usage = models.JSONField(default=dict, blank=True)
    processing_status = models.CharField(max_length=30, default='completed')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"[{self.role.upper()}] {self.content[:40]}..."


class MessageSource(models.Model):
    """
    Normalized source citation linking an assistant answer to the
    exact document, chunk, and page number that provided evidence.
    """
    message = models.ForeignKey(
        ChatMessage,
        on_delete=models.CASCADE,
        related_name='sources'
    )
    document = models.ForeignKey(
        ResearchDocument,
        on_delete=models.CASCADE,
        related_name='citations'
    )
    chunk = models.ForeignKey(
        DocumentChunk,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='citations'
    )
    page_reference = models.CharField(max_length=50, blank=True)
    excerpt = models.TextField(blank=True)
    similarity_score = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['id']

    def __str__(self):
        return f"Source: {self.document.original_filename} (p. {self.page_reference})"
