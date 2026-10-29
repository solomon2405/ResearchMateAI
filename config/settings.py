"""
Django settings for ResearchMate AI project.
"""

from pathlib import Path
import os
import urllib.parse
from dotenv import load_dotenv
from django.core.exceptions import ImproperlyConfigured

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if it exists
load_dotenv(BASE_DIR / '.env')

# SECURITY WARNING: keep the secret key used in production secret!
DEFAULT_INSECURE_DEV_KEY = 'django-insecure-researchmate-development-key-change-in-production-123456'
SECRET_KEY = os.getenv('SECRET_KEY', DEFAULT_INSECURE_DEV_KEY)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 'yes')

if not DEBUG:
    if not SECRET_KEY or SECRET_KEY == DEFAULT_INSECURE_DEV_KEY or len(SECRET_KEY) < 32:
        raise ImproperlyConfigured(
            "A strong, unique SECRET_KEY (min 32 characters) must be configured in environment when DEBUG=False."
        )

ALLOWED_HOSTS = [h.strip() for h in os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1,0.0.0.0').split(',') if h.strip()]
if not DEBUG and not ALLOWED_HOSTS:
    raise ImproperlyConfigured("ALLOWED_HOSTS must not be empty in production.")

CSRF_TRUSTED_ORIGINS = [
    origin.strip() for origin in os.getenv('CSRF_TRUSTED_ORIGINS', '').split(',') if origin.strip()
]

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # ResearchMate AI Applications
    'accounts.apps.AccountsConfig',
    'research_projects.apps.ResearchProjectsConfig',
    'paper_discovery.apps.PaperDiscoveryConfig',
    'documents.apps.DocumentsConfig',
    'research_chat.apps.ResearchChatConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'
ASGI_APPLICATION = 'config.asgi.application'

# Database
# Default: SQLite for reliable local testing & rapid development.
# If DATABASE_URL is configured (e.g. postgresql://user:pass@host:5432/dbname), parse it.
DATABASE_URL = os.getenv('DATABASE_URL', '').strip()

if DATABASE_URL and (DATABASE_URL.startswith('postgresql://') or DATABASE_URL.startswith('postgres://')):
    url = urllib.parse.urlparse(DATABASE_URL)
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': url.path[1:],
            'USER': url.username,
            'PASSWORD': url.password,
            'HOST': url.hostname,
            'PORT': url.port or '5432',
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 8,
        }
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files & Private document storage
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
PRIVATE_MEDIA_ROOT = BASE_DIR / 'media' / 'private_documents'

# Storage directories
def ensure_storage_directories():
    """Lazily ensure required storage directories exist without top-level import side effects."""
    os.makedirs(PRIVATE_MEDIA_ROOT, exist_ok=True)
    os.makedirs(VECTOR_INDEX_DIR, exist_ok=True)

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Authentication URLs
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'dashboard'
LOGOUT_REDIRECT_URL = 'login'

# Production Security Headers & Cookie Flags
if not DEBUG:
    SECURE_SSL_REDIRECT = os.getenv('SECURE_SSL_REDIRECT', 'True').lower() in ('true', '1', 'yes')
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = 'DENY'
    SECURE_HSTS_SECONDS = int(os.getenv('SECURE_HSTS_SECONDS', 31536000))
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    if os.getenv('BEHIND_PROXY', 'False').lower() in ('true', '1', 'yes'):
        SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# ResearchMate AI Configuration Limits (safely bounded)
try:
    MAX_DOCUMENTS_PER_PROJECT = max(1, min(int(os.getenv('MAX_DOCUMENTS_PER_PROJECT', 30)), 200))
except ValueError:
    MAX_DOCUMENTS_PER_PROJECT = 30

try:
    MAX_UPLOAD_SIZE_MB = max(1, min(int(os.getenv('MAX_UPLOAD_SIZE_MB', 25)), 100))
except ValueError:
    MAX_UPLOAD_SIZE_MB = 25

# OpenAlex Academic Discovery Configuration
OPENALEX_MAILTO = os.getenv('OPENALEX_MAILTO', 'researcher@example.com')
OPENALEX_API_URL = os.getenv('OPENALEX_API_URL', 'https://api.openalex.org/works')

# Embedding & Vector Retrieval Configuration
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
VECTOR_STORE_BACKEND = os.getenv('VECTOR_STORE_BACKEND', 'faiss')

try:
    RAG_TOP_K = max(1, min(int(os.getenv('RAG_TOP_K', 5)), 20))
except ValueError:
    RAG_TOP_K = 5

try:
    RAG_SIMILARITY_THRESHOLD = max(0.0, min(float(os.getenv('RAG_SIMILARITY_THRESHOLD', 0.35)), 1.0))
except ValueError:
    RAG_SIMILARITY_THRESHOLD = 0.35

try:
    RAG_MAX_CONTEXT_TOKENS = max(500, min(int(os.getenv('RAG_MAX_CONTEXT_TOKENS', 3000)), 16000))
except ValueError:
    RAG_MAX_CONTEXT_TOKENS = 3000

# LLM Answer Generation
# Options: 'mock', 'ollama', 'gemini', 'openai_compatible'
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'mock').lower()
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'llama3.2')
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_API_BASE_URL = os.getenv('LLM_API_BASE_URL', '')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
