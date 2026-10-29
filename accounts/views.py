from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect
from django.utils.http import url_has_allowed_host_and_scheme
from django.contrib import messages
from .forms import RegisterForm, LoginForm

def register_view(request):
    """Register a new student/researcher."""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            login(request, user)
            messages.success(request, f"Welcome to ResearchMate AI, {user.username}! Your research workspace is ready.")
            return redirect('dashboard')
        else:
            messages.error(request, "Please correct the registration errors below.")
    else:
        form = RegisterForm()

    return render(request, 'accounts/register.html', {'form': form})


def login_view(request):
    """Authenticate and sign in a researcher with safe redirect validation."""
    if request.user.is_authenticated:
        return redirect('dashboard')

    next_url = request.POST.get('next') or request.GET.get('next') or ''

    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            
            # Enforce safe URL validation against open redirect attacks
            if next_url and url_has_allowed_host_and_scheme(
                url=next_url,
                allowed_hosts={request.get_host()},
                require_https=request.is_secure()
            ):
                return redirect(next_url)
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid username or password. Please try again.")
    else:
        form = LoginForm()

    return render(request, 'accounts/login.html', {'form': form, 'next_url': next_url})


@require_POST
@csrf_protect
def logout_view(request):
    """Log out the current user via secure CSRF-protected POST."""
    logout(request)
    messages.info(request, "You have been logged out securely.")
    return redirect('login')


@login_required
def profile_view(request):
    """Display researcher account details and system metrics."""
    user = request.user
    projects_count = user.research_projects.count()
    docs_count = user.uploaded_documents.count()
    convs_count = user.conversations.count()

    context = {
        'user': user,
        'projects_count': projects_count,
        'docs_count': docs_count,
        'convs_count': convs_count,
    }
    return render(request, 'accounts/profile.html', context)
