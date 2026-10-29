import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse, FileResponse, Http404, HttpResponseForbidden
from django.core.exceptions import PermissionDenied
from research_projects.models import ResearchProject
from .models import ResearchDocument
from .forms import DocumentUploadForm
from .services.ingestion import process_and_index_document
from research_chat.services.vector_store import get_project_vector_store

@login_required
def upload_documents_view(request, project_id):
    """
    Handle single or multiple PDF uploads for a specific project.
    Validates ownership, file signatures, quotas, checksums, and kicks off ingestion.
    """
    project = get_object_or_404(ResearchProject, pk=project_id, owner=request.user)

    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES, project=project)
        if form.is_valid():
            uploaded_files = form.cleaned_data['files']
            indexed_count = 0
            failed_count = 0

            for f in uploaded_files:
                try:
                    doc = ResearchDocument.objects.create(
                        project=project,
                        uploader=request.user,
                        original_filename=f.name,
                        file=f,
                        file_size=f.size,
                        checksum=f.checksum,
                        detected_mime='application/pdf',
                        processing_status='uploaded'
                    )
                except Exception as e:
                    # Catch concurrent duplicate uploads or database integrity errors safely
                    failed_count += 1
                    continue

                # Process synchronous for immediate indexing availability
                success = process_and_index_document(doc.id)
                if success:
                    indexed_count += 1
                else:
                    failed_count += 1

            if indexed_count > 0 and failed_count == 0:
                messages.success(request, f"Successfully uploaded and indexed {indexed_count} research paper(s).")
            elif indexed_count > 0 and failed_count > 0:
                messages.warning(
                    request,
                    f"Uploaded {indexed_count + failed_count} paper(s): {indexed_count} indexed successfully, "
                    f"{failed_count} failed to index (check error details below)."
                )
            else:
                messages.error(
                    request,
                    "Document(s) were uploaded but failed text extraction or indexing. Please inspect the error messages."
                )

            return redirect('project_detail', pk=project.pk)
        else:
            messages.error(request, "Upload validation failed. Please check the file requirements.")
    else:
        form = DocumentUploadForm(project=project)

    return render(request, 'documents/upload.html', {
        'form': form,
        'project': project,
    })


@login_required
def delete_document_view(request, pk):
    """
    Delete a document, remove its file from private storage,
    delete its chunks, and rebuild the project's vector store.
    """
    document = get_object_or_404(ResearchDocument, pk=pk, project__owner=request.user)
    project_id = document.project_id
    project = document.project
    filename = document.original_filename

    if request.method == 'POST':
        # Remove file from private disk
        try:
            if document.file and os.path.exists(document.file.path):
                os.remove(document.file.path)
        except Exception:
            pass

        document.delete()

        # Rebuild vector index from remaining chunks
        vector_store = get_project_vector_store(project_id)
        vector_store.rebuild_from_database()

        messages.success(request, f"Document '{filename}' was removed from the project.")
        return redirect('project_detail', pk=project_id)

    return render(request, 'documents/document_confirm_delete.html', {
        'document': document,
        'project': project
    })


@login_required
def retry_document_ingestion_view(request, pk):
    """Retry indexing a document that previously failed."""
    document = get_object_or_404(ResearchDocument, pk=pk, project__owner=request.user)
    
    if request.method == 'POST':
        success = process_and_index_document(document.pk)
        if success:
            messages.success(request, f"Document '{document.original_filename}' successfully indexed!")
        else:
            messages.error(request, f"Re-indexing failed: {document.error_details}")
        return redirect('project_detail', pk=document.project_id)

    return redirect('project_detail', pk=document.project_id)


@login_required
def serve_private_document_view(request, pk):
    """
    Authorized document viewer:
    Streams the private PDF file ONLY if the authenticated user owns the parent project.
    Direct access to media/private_documents/ is blocked at the web server level.
    """
    import re
    document = get_object_or_404(ResearchDocument.objects.select_related('project'), pk=pk)

    # Multi-user isolation enforcement:
    if document.project.owner != request.user:
        raise PermissionDenied("You do not have permission to view or download this research paper.")

    if not document.file or not os.path.exists(document.file.path):
        raise Http404("Document file not found in private storage.")

    safe_filename = re.sub(r'[\r\n"\\/]', '_', document.original_filename)
    response = FileResponse(open(document.file.path, 'rb'), content_type='application/pdf')
    # Use inline so browser PDF viewer can display it (and support #page=N links)
    response['Content-Disposition'] = f'inline; filename="{safe_filename}"'
    return response
