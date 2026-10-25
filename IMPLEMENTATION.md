# ResearchMate AI — Technical Implementation Document
### Complete Architecture, Schema, Algorithmic Pipelines, Security, and Code Inventory

---

## 1. Executive Summary

This document provides a technical explanation of the implementation of **ResearchMate AI**. It outlines the project's codebase structure, database schemas, algorithmic processing pipelines (PDF extraction, text normalization, semantic passage chunking, vector embedding, FAISS indexing, similarity retrieval, and citation validation), security boundaries, and API specifications.

---

## 2. Directory & File Inventory

```
Research project/
│
├── manage.py                          # Django management CLI entrypoint
├── requirements.txt                   # Production & development dependencies
├── .env.example                       # Environment variable template with documentation
├── .env                               # Active local environment variables
├── .gitignore                         # Git exclusion rules (protects private PDFs, caches, secrets)
├── README.md                          # Project documentation & user guide
├── IMPLEMENTATION.md                  # This technical implementation document
│
├── config/                            # Core Django project configuration
│   ├── __init__.py
│   ├── settings.py                    # App configuration, auth, media, RAG, and vector settings
│   ├── urls.py                        # Root URL router with secured private storage boundaries
│   ├── wsgi.py                        # WSGI entrypoint for synchronous production servers
│   └── asgi.py                        # ASGI entrypoint for asynchronous workers
│
├── accounts/                          # Authentication & User Management App
│   ├── __init__.py
│   ├── apps.py                        # App configuration (AccountsConfig)
│   ├── models.py                      # User model associations & profile preferences
│   ├── forms.py                       # RegisterForm & LoginForm with validation
│   ├── views.py                       # register_view, login_view, logout_view, profile_view
│   ├── urls.py                        # /accounts/register/, /accounts/login/, etc.
│   └── tests.py                       # Unit tests: registration, auth guards, password hashing
│
├── research_projects/                 # Research Workspace Management App
│   ├── __init__.py
│   ├── apps.py                        # App configuration (ResearchProjectsConfig)
│   ├── models.py                      # ResearchProject model (owner, topic, timestamps)
│   ├── forms.py                       # ResearchProjectForm (name, topic, description)
│   ├── views.py                       # dashboard_view, project_list, create, detail, edit, delete
│   ├── urls.py                        # /, /projects/, /projects/<id>/, etc.
│   └── tests.py                       # Unit tests: workspace CRUD, multi-user isolation
│
├── paper_discovery/                   # Goal 1: Academic Literature Discovery App
│   ├── __init__.py
│   ├── apps.py                        # App configuration (PaperDiscoveryConfig)
│   ├── models.py                      # Models for future bookmarking / search history
│   ├── forms.py                       # PaperSearchForm (query, year range, sorting)
│   ├── services/
│   │   ├── openalex.py                # OpenAlex Works REST API adapter with polite pool & timeouts
│   │   └── normalizers.py             # Reconstructs inverted index abstracts into clean text
│   ├── views.py                       # search_view (query validation, pagination, display)
│   ├── urls.py                        # /discovery/
│   └── tests.py                       # Tests: normalizer, inverted index, mock timeouts & rate limits
│
├── documents/                         # Goal 2: PDF Ingestion & Private Storage App
│   ├── __init__.py
│   ├── apps.py                        # App configuration (DocumentsConfig)
│   ├── models.py                      # ResearchDocument & DocumentChunk models
│   ├── forms.py                       # DocumentUploadForm (multi-file, magic bytes, SHA256 check)
│   ├── services/
│   │   ├── pdf_extractor.py           # PyMuPDF text extraction, metadata, scanned PDF/OCR check
│   │   ├── chunker.py                 # Overlapping passage chunking preserving page numbers
│   │   └── ingestion.py               # Ingestion pipeline: extraction -> chunking -> embedding -> indexing
│   ├── views.py                       # upload_documents, delete, retry, serve_private_document
│   ├── urls.py                        # /documents/upload/<id>/, /documents/<id>/view/, etc.
│   └── tests.py                       # Tests: PyMuPDF extraction, chunking, OCR check, private storage 403
│
├── research_chat/                     # Goal 2: Vector Search & RAG Chat App
│   ├── __init__.py
│   ├── apps.py                        # App configuration (ResearchChatConfig)
│   ├── models.py                      # ChatConversation, ChatMessage, MessageSource models
│   ├── services/
│   │   ├── embeddings.py              # SentenceTransformer (all-MiniLM-L6-v2) singleton loader
│   │   ├── vector_store.py            # Project-isolated persistent FAISS (IndexFlatIP) engine
│   │   ├── retriever.py               # Project-scoped vector search & similarity threshold gating
│   │   ├── answer_generator.py        # Configurable LLM synthesis (Mock, Ollama, OpenAI-compatible)
│   │   ├── citation_validator.py      # Validates source references against real project chunks & pages
│   │   └── rag_pipeline.py            # Orchestrator tying query -> retrieval -> synthesis -> persistence
│   ├── views.py                       # chat_view, create_conversation, send_message_api, rename, delete
│   ├── urls.py                        # /chat/, /chat/<id>/, /chat/<id>/send/, etc.
│   └── tests.py                       # Tests: vector retrieval, cross-project isolation, citations
│
├── templates/                         # Server-rendered HTML5 templates
│   ├── base.html                      # Global layout, responsive navigation, alert toasts, footer
│   ├── accounts/
│   │   ├── login.html                 # Login page
│   │   ├── register.html              # Registration page
│   │   └── profile.html               # Researcher account statistics & model settings
│   ├── research_projects/
│   │   ├── dashboard.html             # Dashboard with database metrics, active indexing, search bar
│   │   ├── project_list.html          # Grid of user's research workspaces
│   │   ├── project_detail.html        # Workspace detail: documents collection & conversation threads
│   │   ├── project_form.html          # Workspace create / edit form
│   │   └── project_confirm_delete.html# Workspace deletion confirmation
│   ├── paper_discovery/
│   │   └── search.html                # Academic literature search form, cards, manual PDF links
│   ├── documents/
│   │   ├── upload.html                # Multi-PDF upload with preview list & quota indicators
│   │   └── document_confirm_delete.html # Document deletion confirmation
│   └── research_chat/
│       └── chat.html                  # ChatGPT-style interface with sidebar, citations, and modals
│
└── static/                            # Static assets
    ├── css/
    │   └── main.css                   # Custom academic theme (colors, badges, tables, chat layout)
    └── js/
        └── chat.js                    # Vanilla JS for async chat dispatch & clickable PDF citations
```

---

## 3. Relational Database Schema & Data Models

The relational schema is implemented with Django ORM models, enforcing referential integrity, cascading rules, and database indexes:

```
+--------------------+            +------------------------+
|    auth_user       | 1        * |    ResearchProject     |
|--------------------|------------|------------------------|
| id (PK)            |            | id (PK)                |
| username           |            | owner_id (FK -> User)  |
| email              |            | name                   |
| password           |            | research_topic         |
| date_joined        |            | description            |
+--------------------+            | created_at, updated_at |
          |                       +------------------------+
          |                                   | 1
          |                                   |
          | 1                                 | *
          |       *               +------------------------+
          +---------------------->|    ResearchDocument    |
                                  |------------------------|
                                  | id (PK)                |
                                  | project_id (FK)        |
                                  | uploader_id (FK)       |
                                  | original_filename      |
                                  | file (private path)    |
                                  | file_size, checksum    |
                                  | processing_status      |
                                  | error_details          |
                                  | total_pages            |
                                  +------------------------+
                                              | 1
                                              |
                                              | *
+------------------------+        +------------------------+
|    ChatConversation    |        |     DocumentChunk      |
|------------------------|        |------------------------|
| id (PK)                |        | id (PK)                |
| project_id (FK)        |        | project_id (FK)        |
| owner_id (FK -> User)  |        | document_id (FK)       |
| title                  |        | chunk_index            |
| created_at, updated_at |        | page_start, page_end   |
+------------------------+        | text                   |
          | 1                     | vector_id              |
          | *                     | embedding_model        |
+------------------------+        +------------------------+
|      ChatMessage       |                    | 1
|------------------------|                    |
| id (PK)                |                    |
| conversation_id (FK)   |                    |
| role ('user'/'asst')   |                    |
| content                |                    |
| token_usage (JSON)     |                    |
| processing_status      |                    |
+------------------------+                    |
          | 1                                 |
          | *                                 | *
+----------------------------------------------------------+
|                      MessageSource                       |
|----------------------------------------------------------|
| id (PK)                                                  |
| message_id (FK -> ChatMessage)                           |
| document_id (FK -> ResearchDocument)                     |
| chunk_id (FK -> DocumentChunk, nullable)                 |
| page_reference                                           |
| excerpt                                                  |
| similarity_score                                         |
+----------------------------------------------------------+
```

### Constraints and Deletion Invariants:
1. `ResearchDocument` has a unique constraint on `(project, checksum)`, ensuring no duplicate PDFs can be uploaded to the same workspace.
2. Deleting a `ResearchProject` automatically cascades to delete all associated documents, chunks, conversations, messages, citations, and calls `vector_store.delete_index()` to remove physical FAISS files on disk.
3. Deleting a `ResearchDocument` removes the physical PDF file from private disk storage and triggers `vector_store.rebuild_from_database()` to update the FAISS vector index.

---

## 4. Algorithmic Processing Pipelines

### Pipeline A: Academic Paper Discovery (Goal 1)
```
User Search Query (Keywords, Year Bounds, Sort Option)
                     │
                     ▼
       paper_discovery.views.search_view
                     │
                     ▼
  paper_discovery.services.openalex.OpenAlexService
   - Request headers: User-Agent: ResearchMateAI/1.0 (mailto:user@mail)
   - Filter query assembly: "publication_year:2020-2024"
   - Sorting query assembly: "cited_by_count:desc"
   - Timeout: 12 seconds with HTTP error & rate-limit (429) catching
                     │
                     ▼
paper_discovery.services.normalizers.normalize_openalex_work
   - reconstruct_abstract(abstract_inverted_index):
       Iterates token positions {word: [pos1, pos2...]} to reconstruct readable string
   - Normalizes authors: "Author 1, Author 2 et al."
   - Extracts Open Access flags (is_oa, oa_status, oa_url)
   - Identifies legitimate PDF URLs and DOI links
                     │
                     ▼
templates/paper_discovery/search.html
   - Displays metadata cards with "Open PDF Link", "Publisher Record", "DOI Link"
   - Does NOT automatically ingest into RAG (User retains manual selection control)
```

---

### Pipeline B: PDF Ingestion & Indexing (Goal 2)
```
User uploads 1–30 PDFs to Research Workspace
                     │
                     ▼
documents.forms.DocumentUploadForm
   - Extension verification (.pdf)
   - File size validation (<= 25 MB)
   - Magic bytes verification (starts with b'%PDF-')
   - PyMuPDF test open (detects encryption / corruption)
   - SHA-256 duplicate calculation within project scope
   - Project document quota verification (max 30 PDFs)
                     │
                     ▼
ResearchDocument created in DB (status='uploaded')
                     │
                     ▼
documents.services.ingestion.process_and_index_document(doc_id)
   │
   ├─► Stage 1: Text Extraction (pdf_extractor.py)
   │     - Opens document with PyMuPDF (fitz.open)
   │     - Extracts text page-by-page preserving page numbers (1-indexed)
   │     - clean_academic_text(): fixes hyphenated linebreaks ("bio-\nmarkers" -> "biomarkers"),
   │       normalizes control chars, preserves scientific signs (+, -, ±, <, >, =), units, and negations
   │     - Scanned Check: if total_chars < 150 or avg_chars_per_page < 100:
   │         doc.processing_status = 'failed'
   │         doc.error_details = 'Document appears to be scanned and requires OCR.'
   │         return False
   │
   ├─► Stage 2: Semantic Chunking (chunker.py)
   │     - Splits paragraphs into sentences respecting academic abbreviations ("et al.", "Fig.", "i.e.")
   │     - Aggregates sentences into target passages (~500 words, ~700 tokens)
   │     - Maintains a ~100-word overlap across contiguous passages
   │     - Records exact page_start and page_end for every chunk
   │
   ├─► Stage 3: Local Vector Embedding (embeddings.py)
   │     - Encodes chunk texts using sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
   │     - Normalizes vectors to unit length (L2 norm)
   │     - Returns 2D float32 numpy array of shape (N, 384)
   │
   ├─► Stage 4: Relational Persistence
   │     - DocumentChunk.objects.bulk_create(chunk_objects) within transaction.atomic()
   │
   └─► Stage 5: Project FAISS Indexing (vector_store.py)
         - Loads project-specific index: vector_indexes/project_<project_id>.index
         - Calls faiss.IndexFlatIP.add(embeddings)
         - Appends chunk database IDs to project_<project_id>_map.json
         - Saves index and mapping to disk
         - doc.processing_status = 'indexed'
```

---

### Pipeline C: Question Answering & Grounded RAG
```
User asks question in Research Chat conversation
                     │
                     ▼
research_chat.views.send_message_api
   - Verifies request.user owns conversation and parent project
                     │
                     ▼
research_chat.services.rag_pipeline.execute_rag_query(conv_id, user, question)
   │
   ├─► 1. Persist User Question (ChatMessage, role='user')
   │      - Auto-titles conversation from first question if default
   │
   ├─► 2. Scoped Evidence Retrieval (retriever.py)
   │      - Embeds query: embed_query(question) -> 384-d normalized vector
   │      - Queries ProjectVectorStore(project.id).search(query_vec, top_k=10)
   │      - Filters hits where similarity_score >= RAG_SIMILARITY_THRESHOLD (0.35)
   │      - Queries DocumentChunk.objects.filter(id__in=hit_ids, project_id=project.id)
   │      - Constructs bounded evidence passages with filename, page_ref, text, score
   │
   ├─► 3. Evidence Sufficiency Evaluation
   │      - If passages is empty:
   │          has_sufficient_evidence = False
   │          raw_answer = INSUFFICIENT_EVIDENCE_MESSAGE
   │          (Explains that uploaded documents do not contain enough evidence)
   │
   ├─► 4. Answer Generation (answer_generator.py)
   │      - If sufficient evidence:
   │          Constructs strict prompt with untrusted source excerpts:
   │          "--- [Source 1] Document: paper.pdf (Page: 2) ---"
   │          Generates answer via Configured Provider (Mock, Ollama, or OpenAI-compatible)
   │
   ├─► 5. Citation Validation (citation_validator.py)
   │      - Parses answer for [Source 1], [Source 2], etc.
   │      - Maps source indices to actual retrieved DocumentChunk IDs and page numbers
   │      - Strips fabricated citations (e.g. [Source 99])
   │
   └─► 6. Persistence & Response
          - Persists ChatMessage (role='assistant', content=sanitized_answer)
          - Creates normalized MessageSource records linked to document, chunk, page
          - Returns JSON to chat.js
                     │
                     ▼
chat.js dynamically renders:
   - Assistant message bubble
   - Clickable citation chips: "📄 [Source 1] kidney_biomarkers.pdf (p. 2)"
   - Clicking citation opens /documents/<id>/view/#page=2 in authorized PDF viewer
```

---

## 5. Security & Multi-User Isolation Matrix

| Layer | Security Rule | Enforcement Mechanism |
|---|---|---|
| **Private Document Files** | Uploaded PDFs must never be exposed via public web links. | Files stored in `media/private_documents/`. Excluded from `config/urls.py` static/media handlers. |
| **Document Download** | Only the owner of the parent project can download/view a PDF. | `serve_private_document_view` verifies `document.project.owner == request.user`. Returns 403 Forbidden otherwise. |
| **Workspace Isolation** | User A cannot view, edit, or delete User B's workspaces. | Views query `get_object_or_404(ResearchProject, pk=pk, owner=request.user)`. Returns 404. |
| **Vector Index Isolation** | Queries in Workspace A cannot retrieve chunks from Workspace B. | Separate FAISS index files per project (`project_<id>.index`). Retrieval queries only the active project's file. |
| **Chat Isolation** | User A cannot read or send messages to User B's conversations. | `send_message_api` verifies `conversation.owner == request.user`. Returns 403 / 404. |
| **Untrusted Source Input** | PDF text cannot override system prompts or execute commands. | LLM prompts treat excerpts as untrusted source data; no tools or code interpreters are accessible to the LLM. |
| **Form & API Protection** | CSRF protection for all state-changing operations. | Django `CsrfViewMiddleware` + `X-CSRFToken` in AJAX headers. |

---

## 6. API & Route Specifications

### Authentication Routes (`accounts/urls.py`)
- `GET/POST /accounts/register/` &rarr; Register a new student/researcher.
- `GET/POST /accounts/login/` &rarr; Session login.
- `GET /accounts/logout/` &rarr; Invalidate session and redirect to login.
- `GET /accounts/profile/` &rarr; User account overview and system configurations.

### Workspace Routes (`research_projects/urls.py`)
- `GET /` &rarr; Main research dashboard with metrics and active indexing statuses.
- `GET /projects/` &rarr; Paginated list of user's research projects.
- `GET/POST /projects/create/` &rarr; Create a new project workspace.
- `GET /projects/<id>/` &rarr; Workspace detail showing document collections & conversations.
- `GET/POST /projects/<id>/edit/` &rarr; Edit workspace name, topic, or description.
- `GET/POST /projects/<id>/delete/` &rarr; Delete workspace and cascade-delete documents/vectors.

### Literature Discovery Routes (`paper_discovery/urls.py`)
- `GET /discovery/?q=<query>&start_year=<year>&sort_by=<sort>&page=<page>` &rarr; Query OpenAlex Works API and render normalized metadata cards with direct source links.

### Document Routes (`documents/urls.py`)
- `GET/POST /documents/upload/<project_id>/` &rarr; Multi-file upload, validation, and ingestion pipeline trigger.
- `POST /documents/<id>/retry/` &rarr; Retry indexing for failed documents.
- `GET/POST /documents/<id>/delete/` &rarr; Remove PDF, delete chunks, and rebuild vector index.
- `GET /documents/<id>/view/` &rarr; Authorized private document streaming view (supports `#page=N`).

### Research Chat Routes (`research_chat/urls.py`)
- `GET /chat/` &rarr; Main chat interface opening most recent active conversation.
- `GET /chat/<conversation_id>/` &rarr; Open a specific conversation with message history.
- `GET /chat/new/<project_id>/` &rarr; Create and open a new conversation inside a project.
- `POST /chat/<conversation_id>/send/` &rarr; AJAX JSON endpoint submitting query and returning grounded answer + citations.
- `POST /chat/<conversation_id>/rename/` &rarr; Update conversation title.
- `POST /chat/<conversation_id>/delete/` &rarr; Delete conversation and cascade-delete messages/citations.

---

## 7. Verification Test Suite Implementation

The test suite contains **35 tests** executed across the Django test framework:

```bash
python manage.py test
```

### Key Test Classes:
1. `accounts.tests.AuthenticationTests`: Validates registration, duplicate email rejection, login failures, and protected view redirects.
2. `research_projects.tests.ResearchProjectTests`: Validates project CRUD and multi-user isolation (verifying User 2 receives HTTP 404 when attempting to access User 1's projects).
3. `paper_discovery.tests.PaperDiscoveryTests`: Validates inverted index reconstruction, metadata normalization, and mock network failures (timeouts, 429 rate limits).
4. `documents.tests.DocumentPipelineTests`: Uses in-memory synthetic PDFs created via PyMuPDF to test text extraction, chunking, scanned document OCR detection, SHA-256 duplicate rejection, and private document streaming authorization (HTTP 403 for unauthorized users).
5. `research_chat.tests.ResearchChatRAGTests`: Ingests real test PDFs into two separate projects (Project 1: Nephrology, Project 2: Corn Blight) and proves that querying Project 1 **never** returns chunks from Project 2 (cross-project isolation), tests out-of-scope query handling, and tests citation validation.

---
*Authored for the ResearchMate AI Project.*
