from django.urls import path
from . import views

urlpatterns = [
    path('upload/<int:project_id>/', views.upload_documents_view, name='document_upload'),
    path('<int:pk>/delete/', views.delete_document_view, name='document_delete'),
    path('<int:pk>/retry/', views.retry_document_ingestion_view, name='document_retry'),
    path('<int:pk>/view/', views.serve_private_document_view, name='document_serve'),
]
