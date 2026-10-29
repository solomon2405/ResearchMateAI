import json
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.core.exceptions import PermissionDenied
from django.views.decorators.http import require_POST
from django.contrib import messages
from research_projects.models import ResearchProject
from .models import ChatConversation, ChatMessage, MessageSource
from .services.rag_pipeline import execute_rag_query

logger = logging.getLogger(__name__)

@login_required
def chat_view(request, conversation_id=None):
    """
    Main Research Chat interface:
    Sidebar of conversations, active project display, chronological message area,
    citations with direct links to PDFs and pages, and real-time interactive input.
    """
    user_projects = request.user.research_projects.prefetch_related('conversations', 'documents').all()
    user_conversations = request.user.conversations.select_related('project').order_by('-updated_at')

    active_conversation = None
    messages_list = []
    active_project = None

    if conversation_id:
        active_conversation = get_object_or_404(
            ChatConversation.objects.select_related('project'),
            pk=conversation_id,
            owner=request.user
        )
        active_project = active_conversation.project
        messages_list = active_conversation.messages.prefetch_related('sources__document').all()
    elif user_conversations.exists():
        # Open most recent conversation by default if available
        active_conversation = user_conversations.first()
        active_project = active_conversation.project
        messages_list = active_conversation.messages.prefetch_related('sources__document').all()
    elif user_projects.exists():
        # If no conversation exists yet, pick first project
        active_project = user_projects.first()

    context = {
        'user_projects': user_projects,
        'user_conversations': user_conversations,
        'active_conversation': active_conversation,
        'active_project': active_project,
        'messages_list': messages_list,
    }
    return render(request, 'research_chat/chat.html', context)


@login_required
def create_conversation_view(request, project_id):
    """Create a new research conversation inside a project."""
    project = get_object_or_404(ResearchProject, pk=project_id, owner=request.user)

    conversation = ChatConversation.objects.create(
        project=project,
        owner=request.user,
        title=f"Chat: {project.name[:30]}"
    )
    return redirect('chat_conversation', conversation_id=conversation.id)


@login_required
@require_POST
def send_message_api(request, conversation_id):
    """
    AJAX endpoint to submit a research question, run the RAG pipeline,
    and return evidence-grounded answer with source citations.
    """
    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        data = request.POST

    question = data.get('question', '').strip()
    if not question:
        return JsonResponse({'error': 'Question cannot be empty.'}, status=400)

    try:
        conversation = ChatConversation.objects.select_related('project').get(pk=conversation_id)
        if conversation.owner != request.user:
            return JsonResponse({'error': 'Unauthorized access.'}, status=403)
    except ChatConversation.DoesNotExist:
        return JsonResponse({'error': 'Conversation not found.'}, status=404)

    try:
        result = execute_rag_query(conversation_id, request.user, question)
        return JsonResponse({
            'status': 'success',
            'user_message_id': result['user_message_id'],
            'assistant_message_id': result['assistant_message_id'],
            'answer': result['answer'],
            'citations': result['citations'],
            'has_sufficient_evidence': result['has_sufficient_evidence'],
            'top_score': result['top_score'],
        })
    except PermissionDenied as e:
        logger.warning(f"Permission denied during RAG query for user {request.user.id}: {e}")
        return JsonResponse({'status': 'error', 'error': 'You do not have permission to query this conversation.'}, status=403)
    except ValueError as e:
        logger.warning(f"Validation error during RAG query: {e}")
        return JsonResponse({'status': 'error', 'error': str(e)}, status=400)
    except Exception as e:
        logger.exception(f"Unexpected error in send_message_api for conversation {conversation_id}: {e}")
        return JsonResponse({
            'status': 'error',
            'error': 'An internal error occurred while generating your research answer. Please try again later.'
        }, status=500)


@login_required
@require_POST
def rename_conversation_view(request, conversation_id):
    """Rename a research conversation."""
    conversation = get_object_or_404(ChatConversation, pk=conversation_id, owner=request.user)
    new_title = request.POST.get('title', '').strip()
    if new_title:
        conversation.title = new_title[:200]
        conversation.save(update_fields=['title', 'updated_at'])
        messages.success(request, "Conversation renamed.")
    return redirect('chat_conversation', conversation_id=conversation.id)


@login_required
@require_POST
def delete_conversation_view(request, conversation_id):
    """Delete a research conversation."""
    conversation = get_object_or_404(ChatConversation, pk=conversation_id, owner=request.user)
    project_id = conversation.project_id
    conversation.delete()
    messages.success(request, "Conversation deleted.")
    return redirect('chat_main')
