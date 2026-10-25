from django.shortcuts import render
from django.contrib import messages
from .forms import PaperSearchForm
from .services.openalex import OpenAlexService

def search_view(request):
    """
    Search academic literature via OpenAlex.
    Retrieves genuine publication metadata, citations, venues,
    and direct source links for manual download.
    """
    form = PaperSearchForm(request.GET or None)
    search_data = None
    query = request.GET.get('q', '').strip()

    if query:
        if form.is_valid():
            q = form.cleaned_data['q']
            start_year = form.cleaned_data.get('start_year')
            end_year = form.cleaned_data.get('end_year')
            sort_by = form.cleaned_data.get('sort_by')
            page = form.cleaned_data.get('page') or 1
            
            service = OpenAlexService()
            search_data = service.search_works(
                query=q,
                start_year=start_year,
                end_year=end_year,
                sort_by=sort_by,
                page=page,
                per_page=10
            )

            if search_data.get('error'):
                messages.warning(request, search_data['error'])
        else:
            messages.error(request, "Please check the search filters and try again.")

    # Calculate pagination navigation
    has_prev = False
    has_next = False
    prev_page = 1
    next_page = 1

    if search_data and search_data.get('total_pages', 0) > 1:
        current_page = search_data['page']
        total_pages = search_data['total_pages']
        has_prev = current_page > 1
        has_next = current_page < total_pages
        prev_page = current_page - 1
        next_page = current_page + 1

    # Get user's projects for quick "Note for project" or upload context
    user_projects = None
    if request.user.is_authenticated:
        user_projects = request.user.research_projects.all()

    context = {
        'form': form,
        'search_data': search_data,
        'query': query,
        'has_prev': has_prev,
        'has_next': has_next,
        'prev_page': prev_page,
        'next_page': next_page,
        'user_projects': user_projects,
    }
    return render(request, 'paper_discovery/search.html', context)
