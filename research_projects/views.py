from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from .models import ResearchProject
from .forms import ResearchProjectForm
from documents.models import ResearchDocument
from research_chat.models import ChatConversation

@login_required
def dashboard_view(request):
    """
    Main Research Dashboard:
    Displays metrics, recent projects, active chats, processing documents,
    and a quick search field.
    """
    user = request.user
    projects = user.research_projects.all()
    recent_projects = projects[:4]
    
    total_projects = projects.count()
    total_documents = user.uploaded_documents.count()
    indexed_documents = user.uploaded_documents.filter(processing_status='indexed').count()
    
    # In-progress documents
    processing_docs = user.uploaded_documents.exclude(
        processing_status__in=['indexed', 'failed']
    ).order_by('-updated_at')[:6]
    
    # Recent conversations
    recent_conversations = user.conversations.select_related('project').order_by('-updated_at')[:5]

    context = {
        'total_projects': total_projects,
        'total_documents': total_documents,
        'indexed_documents': indexed_documents,
        'recent_projects': recent_projects,
        'processing_docs': processing_docs,
        'recent_conversations': recent_conversations,
    }
    return render(request, 'research_projects/dashboard.html', context)


@login_required
def project_list_view(request):
    """List all research projects owned by the authenticated researcher."""
    projects_list = request.user.research_projects.all()
    paginator = Paginator(projects_list, 9)
    page_number = request.GET.get('page')
    projects = paginator.get_page(page_number)
    
    return render(request, 'research_projects/project_list.html', {
        'projects': projects,
    })


@login_required
def project_create_view(request):
    """Create a new research project workspace."""
    if request.method == 'POST':
        form = ResearchProjectForm(request.POST)
        if form.is_valid():
            project = form.save(commit=False)
            project.owner = request.user
            project.save()
            messages.success(request, f"Research project '{project.name}' created successfully!")
            return redirect('project_detail', pk=project.pk)
        else:
            messages.error(request, "Please fix the form errors below.")
    else:
        form = ResearchProjectForm()

    return render(request, 'research_projects/project_form.html', {
        'form': form,
        'action_title': 'Create New Research Project',
        'is_create': True
    })


@login_required
def project_detail_view(request, pk):
    """
    Display project details:
    Uploaded research papers, processing statuses, conversations, and workspace controls.
    """
    project = get_object_or_404(ResearchProject, pk=pk, owner=request.user)
    documents = project.documents.all()
    conversations = project.conversations.all()
    
    context = {
        'project': project,
        'documents': documents,
        'conversations': conversations,
        'document_count': documents.count(),
        'indexed_count': documents.filter(processing_status='indexed').count(),
    }
    return render(request, 'research_projects/project_detail.html', context)


@login_required
def project_edit_view(request, pk):
    """Edit existing research project details."""
    project = get_object_or_404(ResearchProject, pk=pk, owner=request.user)

    if request.method == 'POST':
        form = ResearchProjectForm(request.POST, instance=project)
        if form.is_valid():
            form.save()
            messages.success(request, "Project details updated successfully.")
            return redirect('project_detail', pk=project.pk)
    else:
        form = ResearchProjectForm(instance=project)

    return render(request, 'research_projects/project_form.html', {
        'form': form,
        'project': project,
        'action_title': f"Edit Project: {project.name}",
        'is_create': False
    })


@login_required
def project_delete_view(request, pk):
    """Delete a research project and all its associated documents, chunks, and vector index."""
    project = get_object_or_404(ResearchProject, pk=pk, owner=request.user)

    if request.method == 'POST':
        project_name = project.name
        # Delete FAISS vector index if exists on disk
        from research_chat.services.vector_store import get_project_vector_store
        vector_store = get_project_vector_store(project.pk)
        vector_store.delete_index()

        project.delete()
        messages.success(request, f"Project '{project_name}' and all its research data were deleted.")
        return redirect('project_list')

    return render(request, 'research_projects/project_confirm_delete.html', {'project': project})
