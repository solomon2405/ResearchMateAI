"""
Full RAG Orchestration Pipeline for ResearchMate AI.
Ensures authorization, executes scoped retrieval, generates evidence-grounded answers,
validates citations, and records message history and source citations in the database.
"""

import logging
from typing import Dict, Any
from django.db import transaction
from django.core.exceptions import PermissionDenied
from research_chat.models import ChatConversation, ChatMessage, MessageSource
from documents.models import DocumentChunk, ResearchDocument
from .retriever import retrieve_project_evidence
from .answer_generator import get_answer_generator
from .citation_validator import extract_and_validate_citations

logger = logging.getLogger(__name__)

def execute_rag_query(
    conversation_id: int,
    user,
    question: str
) -> Dict[str, Any]:
    """
    Execute an evidence-grounded question-answering workflow:
    1. Authenticate user & verify conversation ownership
    2. Persist user message
    3. Retrieve relevant project passages
    4. Generate grounded answer
    5. Validate citations against project documents
    6. Persist assistant message & normalized MessageSources
    
    Returns:
        {
            'user_message_id': int,
            'assistant_message_id': int,
            'answer': str,
            'citations': List[Dict[str, Any]],
            'has_sufficient_evidence': bool,
            'top_score': float,
        }
    """
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Question cannot be empty.")

    # 1. Ownership & Authorization Check
    try:
        conversation = ChatConversation.objects.select_related('project').get(pk=conversation_id)
    except ChatConversation.DoesNotExist:
        raise ValueError("Conversation not found.")

    if conversation.owner != user or conversation.project.owner != user:
        raise PermissionDenied("You do not have permission to access this research conversation.")

    project = conversation.project

    # 2. Persist User Question
    with transaction.atomic():
        user_msg = ChatMessage.objects.create(
            conversation=conversation,
            role='user',
            content=clean_question,
            processing_status='completed'
        )

        # Auto-update initial conversation title if still default
        if conversation.title == 'New Research Chat' and conversation.messages.count() <= 2:
            title_candidate = clean_question[:45].strip()
            if len(clean_question) > 45:
                title_candidate += "..."
            conversation.title = title_candidate
            conversation.save(update_fields=['title', 'updated_at'])

    # 3. Retrieve Scoped Project Evidence
    retrieval_result = retrieve_project_evidence(
        project_id=project.id,
        question=clean_question
    )

    passages = retrieval_result['passages']
    has_sufficient_evidence = retrieval_result['has_sufficient_evidence']
    top_score = retrieval_result['top_score']

    # 4. Generate Answer via Configured Generator
    generator = get_answer_generator()
    
    # Fetch recent conversation history for flow
    recent_messages = list(conversation.messages.order_by('-created_at')[:4])
    recent_messages.reverse()
    history = [{'role': m.role, 'content': m.content} for m in recent_messages if m.id != user_msg.id]

    generation_result = generator.generate_answer(
        question=clean_question,
        passages=passages,
        has_sufficient_evidence=has_sufficient_evidence,
        conversation_history=history
    )
    raw_answer = generation_result['answer']

    # 5. Validate Citations
    sanitized_answer, valid_citations = extract_and_validate_citations(
        raw_answer=raw_answer,
        passages=passages,
        project_id=project.id
    )

    # 6. Persist Assistant Answer and Citations
    with transaction.atomic():
        assistant_msg = ChatMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=sanitized_answer,
            token_usage=generation_result.get('token_usage', {}),
            processing_status='completed'
        )

        # Create normalized source links
        for cit in valid_citations:
            MessageSource.objects.create(
                message=assistant_msg,
                document_id=cit['document_id'],
                chunk_id=cit['chunk_id'],
                page_reference=cit['page_reference'],
                excerpt=cit['excerpt'],
                similarity_score=cit.get('similarity_score')
            )

        conversation.save(update_fields=['updated_at'])

    return {
        'user_message_id': user_msg.id,
        'assistant_message_id': assistant_msg.id,
        'answer': sanitized_answer,
        'citations': valid_citations,
        'has_sufficient_evidence': has_sufficient_evidence,
        'top_score': top_score,
    }
