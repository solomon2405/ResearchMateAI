"""
URL Configuration for ResearchMate AI.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path('discovery/', include('paper_discovery.urls')),
    path('documents/', include('documents.urls')),
    path('chat/', include('research_chat.urls')),
    path('', include('research_projects.urls')),
]

# In development, serve static assets.
# Note: Private research PDFs in media/private_documents are strictly protected
# and served exclusively via documents.views.serve_private_document_view with ownership checks.
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
