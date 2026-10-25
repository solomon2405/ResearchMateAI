"""
Django settings for ResearchMate AI project.
"""

from pathlib import Path
import os
import urllib.parse
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if it exists
load_dotenv(BASE_DIR / '.env')

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv(
    'SECRET_KEY',
    'django-insecure-researchmate-development-key-change-in-production-123456'
)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 'yes')

ALLOWED_HOSTS = [h.strip() for h in os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1,0.0.0.0').split(',') if h.strip()]

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

# Ensure required storage directories exist
os.makedirs(PRIVATE_MEDIA_ROOT, exist_ok=True)

# Vector store index directory (persistent local FAISS indexes per project)
VECTOR_INDEX_DIR = BASE_DIR / 'vector_indexes'
os.makedirs(VECTOR_INDEX_DIR, exist_ok=True)

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Authentication URLs
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'dashboard'
LOGOUT_REDIRECT_URL = 'login'

# ResearchMate AI Configuration Limits
MAX_DOCUMENTS_PER_PROJECT = int(os.getenv('MAX_DOCUMENTS_PER_PROJECT', 30))
MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', 25))

# OpenAlex Academic Discovery Configuration
OPENALEX_MAILTO = os.getenv('OPENALEX_MAILTO', 'researcher@example.com')
OPENALEX_API_URL = 'https://api.openalex.org/works'

# Embedding & Vector Retrieval Configuration
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
VECTOR_STORE_BACKEND = os.getenv('VECTOR_STORE_BACKEND', 'faiss')
RAG_TOP_K = int(os.getenv('RAG_TOP_K', 5))
RAG_SIMILARITY_THRESHOLD = float(os.getenv('RAG_SIMILARITY_THRESHOLD', 0.35))
RAG_MAX_CONTEXT_TOKENS = int(os.getenv('RAG_MAX_CONTEXT_TOKENS', 3000))

# LLM Answer Generation
# Options: 'mock', 'ollama', 'gemini', 'openai_compatible'
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'mock')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'llama3.2')
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_API_BASE_URL = os.getenv('LLM_API_BASE_URL', '')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
