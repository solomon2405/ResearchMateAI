# ResearchMate AI 🔬📚
### Academic Literature Discovery & Evidence-Grounded Multi-PDF RAG Platform

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-6.1-green.svg)](https://www.djangoproject.com/)
[![Vector Engine](https://img.shields.io/badge/FAISS-CPU_1.15-purple.svg)](https://github.com/facebookresearch/faiss)
[![Embeddings](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Tests](https://img.shields.io/badge/Test_Suite-35_Passing-brightgreen.svg)](#-automated-testing-and-verification)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📑 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [The Solution: ResearchMate AI](#-the-solution-researchmate-ai)
3. [Core Capabilities & Workflows](#-core-capabilities--workflows)
4. [System Architecture](#-system-architecture)
5. [Technology Stack](#-technology-stack)
6. [Prerequisites & Installation](#-prerequisites--installation)
7. [Configuration & Environment Variables](#-configuration--environment-variables)
8. [Database Setup & Migrations](#-database-setup--migrations)
9. [LLM & Embedding Model Configuration](#-llm--embedding-model-configuration)
10. [Step-by-Step User Walkthrough](#-step-by-step-user-walkthrough)
11. [Automated Testing and Verification](#-automated-testing-and-verification)
12. [Security, Multi-User Isolation & Privacy](#-security-multi-user-isolation--privacy)
13. [Known Limitations](#-known-limitations)
14. [Future Work & Roadmap](#-future-work--roadmap)
15. [Deployment Guidance](#-deployment-guidance)

---

## 🎯 Problem Statement

Conducting rigorous academic research in universities, laboratories, and research institutions is hindered by three compounding obstacles:

### 1. Literature Fragmentation & Source Inconsistency
Academic researchers, graduate students, and independent scholars spend dozens of hours navigating fragmented search engines, institutional paywalls, unvetted preprints, and scattered repositories. Extracting meaningful publication metadata, assessing citation impact, and obtaining direct, legitimate PDF download links without falling victim to predatory journals or incomplete records remains tedious.

### 2. General LLM Hallucinations & Fabricated Citations
Generic AI chatbots (e.g., standard ChatGPT, Copilot) are fundamentally unsuited for academic synthesis because they generate plausible-sounding **hallucinations**:
- Fabricating non-existent paper titles, authors, and DOIs.
- Inventing experimental statistics, sample sizes, and p-values.
- Blending claims across unrelated studies without traceability.
- Possessing no verifiable link between an assistant's claim and the exact page or paragraph of a specific PDF.

### 3. Cognitive Overload in Multi-Document Synthesis
A typical literature review requires deep analysis of 20 to 30 primary research papers (totalling 300 to 800 pages of dense scientific literature). Researchers struggle to cross-reference conflicting findings, compare baseline models, isolate methodological differences, and track which specific paper established a given claim without repeatedly re-reading hundreds of pages.

### 4. Privacy, Intellectual Property & Data Leakage
Uploading confidential, unpublished pre-prints or proprietary research data to generic, multi-tenant cloud AI platforms exposes sensitive research data to model retraining, intellectual property theft, and lack of researcher-level isolation.

---

## 💡 The Solution: ResearchMate AI

**ResearchMate AI** is a production-ready, full-stack academic research platform engineered to eliminate these problems through a **two-pillar architecture**:

### Pillar 1: Verified Academic Discovery via OpenAlex
- Connects directly to the **OpenAlex API** indexing hundreds of millions of scholarly works.
- Reconstructs OpenAlex token-inverted abstracts into readable, natural prose.
- Displays publication venues, peer-reviewed open-access status, citation counts, and legitimate source links (DOIs, repository records, direct open-access PDFs).
- **Strict Manual Download Model:** Discovered papers are never automatically forced into a user's RAG collection; the researcher maintains intentional custody by reviewing metadata, downloading chosen papers, and curating their project library.

### Pillar 2: Evidence-Grounded Multi-PDF RAG Chat
- **Isolated Research Workspaces:** Researchers create dedicated workspaces for specific research topics (e.g., *"Kidney Disease Biomarkers"*, *"Transformer Attention in Genomics"*).
- **PyMuPDF Extraction & Semantic Chunking:** Ingests up to 30 research paper PDFs per workspace, extracts text page-by-page, strips extraction artifacts, detects scanned PDFs requiring OCR, and creates overlapping ~500–800 word passages preserving exact page numbers (`page_start`, `page_end`).
- **Local Dense Embeddings (Zero Cost):** Employs `sentence-transformers/all-MiniLM-L6-v2` locally on CPU or GPU to produce 384-dimensional normalized float32 vectors without external API calls or recurring charges.
- **Project-Isolated FAISS Vector Store:** Persists dedicated index files (`vector_indexes/project_<id>.index`) and ID mappings per workspace. Vector searches in Project A cannot search, retrieve, or leak passages from Project B.
- **Strict Document Grounding System Prompt:** Assistant answers are generated using **only** retrieved evidence passages. If the uploaded papers do not contain sufficient evidence, ResearchMate explicitly communicates the limitation rather than hallucinating an answer.
- **Citation Validation & Clickable Page Links:** Every assistant claim is validated against real chunks in the active project. Citations appear as clickable chips beneath answers that open the authorized PDF viewer at the exact cited page (`#page=N`).
- **Complete Multi-User Isolation & Private Storage:** Uploaded PDFs are stored in protected storage (`media/private_documents/`) and are never exposed over public URLs. Every view, query, and download enforces strict ownership verification.

---

## ⚡ Core Capabilities & Workflows

| Feature | Description |
|---|---|
| 🔍 **Academic Literature Search** | Live OpenAlex queries with topic/keyword input, year bounds, sorting, and pagination. |
| 📄 **Abstract Reconstruction** | Converts token-inverted index arrays from OpenAlex into coherent abstract paragraphs. |
| 🛡️ **Private Document Storage** | Private disk storage with ownership-gated streaming views; zero public URL leaks. |
| 📑 **PyMuPDF Ingestion** | Robust page-by-page extraction, scientific symbol preservation, and scanned PDF detection. |
| 🧩 **Semantic Passage Chunking** | Overlapping passages (500–800 words) with exact starting and ending page tracking. |
| 🧠 **Local Dense Embeddings** | Normalized sentence embeddings generated locally with `all-MiniLM-L6-v2`. |
| 🗄️ **Project-Isolated FAISS** | Separate Inner-Product (Cosine Similarity) index per workspace with disk persistence. |
| 🎯 **Strict Evidence Grounding** | Untrusted source framing; assistant answers cite claims using `[Source 1]`, `[Source 2]`. |
| 🚫 **Evidence Sufficiency Check** | Similarity threshold gating; communicates lack of evidence when queries are unsupported. |
| 🔗 **Clickable Page Citations** | Source chips beneath answers link directly to `/documents/<id>/view/#page=<N>`. |
| 💬 **ChatGPT-Inspired Chat** | Left sidebar with conversation history, multiline input, auto-titling, and renaming/deletion. |
| 🔒 **Multi-User Isolation** | Complete user separation across workspaces, documents, chunks, vectors, and chats. |

---

## 🏛️ System Architecture

```
                                  +---------------------------+
                                  |   Researcher / Student    |
                                  +-------------+-------------+
                                                |
                                                v
                        +-----------------------------------------------+
                        |        Django Web Layer (Port 8000)           |
                        |  - Authentication & Session Middleware        |
                        |  - CSRF Protection & Ownership Verification   |
                        +-------+---------------+---------------+-------+
                                |               |               |
       +------------------------+               |               +-----------------------+
       |                                        |                                       |
       v                                        v                                       v
+--------------+                      +-------------------+                    +----------------+
|  accounts    |                      |  paper_discovery  |                    |   documents    |
| - Register   |                      | - OpenAlex API    |                    | - Upload Form  |
| - Login/Out  |                      | - Reconstruct     |                    | - SHA-256 Hash |
| - Profiles   |                      |   Inverted Index  |                    | - Quotas (30)  |
+--------------+                      | - Real Source/DOI |                    | - PyMuPDF Fitz |
                                      +-------------------+                    | - Chunker.py   |
                                                                               +--------+-------+
                                                                                        |
                                                                                        v
                                  +---------------------------+                +----------------+
                                  |   Local Sentence Model    |<---------------+ Chunks & Text  |
                                  |   (all-MiniLM-L6-v2)      |                +----------------+
                                  +-------------+-------------+
                                                |
                                                v
                                  +---------------------------+
                                  | Project FAISS Store       |
                                  | project_{id}.index        |
                                  | project_{id}_map.json     |
                                  +-------------+-------------+
                                                |
                                                |
                                                v
+-----------------------------------------------+-----------------------------------------------+
|                                  research_chat RAG Pipeline                                   |
|  1. User Question --> embed_query()                                                           |
|  2. Scoped Vector Retrieval --> Project FAISS Search (Cosine Similarity >= 0.35)             |
|  3. Sufficiency Evaluation --> If insufficient, return limitation message                     |
|  4. Strict Prompt Construction --> Bounded evidence context with [Source N] tags              |
|  5. LLM Answer Synthesis --> Configurable Generator (Mock / Ollama / OpenAI-compatible)       |
|  6. Citation Validator --> Verify source IDs against real project chunks & pages              |
|  7. Persistence --> Save ChatMessage & normalized MessageSource records                       |
+-----------------------------------------------+-----------------------------------------------+
                                                |
                                                v
                                  +---------------------------+
                                  |    Interactive Chat UI    |
                                  |  - Grounded answer bubble |
                                  |  - Clickable citation     |
                                  |    chips (#page=N)        |
                                  +---------------------------+
```

---

## 🛠️ Technology Stack

| Layer | Component | Specification |
|---|---|---|
| **Language** | Python | 3.12.8 (compatible with 3.11–3.13) |
| **Framework** | Django | 6.1.2 / 5.1+ (Built-in Auth, ORM, CSRF, Templates) |
| **Relational Database** | SQLite (Dev) / PostgreSQL (Prod) | Normalized tables, foreign keys, cascade rules, indexing |
| **Vector Store** | FAISS (`faiss-cpu`) | 1.15.1 (`IndexFlatIP` on unit normalized vectors) |
| **Embeddings** | `sentence-transformers` | 5.2.2 (`all-MiniLM-L6-v2`, 384 dimensions) |
| **PDF Extraction** | PyMuPDF (`fitz`) | 1.26.7 (Page-by-page extraction, metadata, OCR detection) |
| **HTTP Client** | `httpx` & `requests` | Async/sync clients with polite pool headers and timeouts |
| **Frontend Styling** | Vanilla CSS3 | Custom Academic Design System (Dark/Light Slate palette) |
| **Frontend Behavior** | Vanilla JavaScript | Asynchronous fetch, real-time message stream, PDF page anchors |
| **Literature Provider** | OpenAlex Works REST API | Real scholarly works, DOIs, open-access URLs |

---

## 📋 Prerequisites & Installation

### 1. System Requirements
- **OS:** Windows 10/11, macOS, or Linux.
- **Python:** 3.11, 3.12, or 3.13 installed and added to `PATH`.
- **Memory:** Minimum 4 GB RAM (8 GB+ recommended for local sentence-transformer models).
- **Disk Space:** ~2 GB for Python packages, cache models, and local PDF storage.

### 2. Environment Setup

Clone or open the project directory in your terminal:

```bash
# Verify Python version:
python --version

# Create an isolated virtual environment:
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1

# Activate on macOS/Linux:
source .venv/bin/activate
```

### 3. Install Required Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Configuration & Environment Variables

ResearchMate AI reads configuration values using `python-dotenv`. An example configuration file is provided in [`.env.example`](file:///c:/SOLOMON/Research%20project/.env.example).

Copy the template to create your active [`.env`](file:///c:/SOLOMON/Research%20project/.env):

```bash
cp .env.example .env
```

### Configuration Options Reference

| Variable | Default Value | Description |
|---|---|---|
| `SECRET_KEY` | *(django-insecure-dev-key)* | Unique secret key for session and CSRF security. |
| `DEBUG` | `True` | Set `False` in production. |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1,0.0.0.0` | Permitted hostnames. |
| `DATABASE_URL` | *(empty = local SQLite)* | PostgreSQL connection URL (e.g. `postgresql://user:pass@localhost:5432/researchmate`). |
| `MAX_DOCUMENTS_PER_PROJECT` | `30` | Maximum PDF papers permitted per research workspace. |
| `MAX_UPLOAD_SIZE_MB` | `25` | Maximum file size allowed for an individual PDF. |
| `OPENALEX_MAILTO` | `researcher@example.com` | Email included in the OpenAlex polite pool for high-rate API limits. |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformer model identifier. |
| `VECTOR_STORE_BACKEND` | `faiss` | Vector store implementation (`faiss`). |
| `LLM_PROVIDER` | `mock` | Generation backend (`mock`, `ollama`, `openai_compatible`, `gemini`). |
| `LLM_MODEL_NAME` | `llama3.2` | Model name for Ollama or hosted OpenAI-compatible endpoint. |
| `LLM_API_KEY` | *(empty)* | API key if using hosted provider (OpenAI, Groq, Together, etc.). |
| `LLM_API_BASE_URL` | *(empty)* | Custom base URL for OpenAI-compatible hosted APIs. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | REST base URL for local Ollama daemon. |
| `RAG_TOP_K` | `5` | Maximum evidence passages retrieved per user question. |
| `RAG_SIMILARITY_THRESHOLD` | `0.35` | Minimum cosine similarity required to consider evidence sufficient. |
| `RAG_MAX_CONTEXT_TOKENS` | `3000` | Context token budget passed to generation model. |

---

## 🗄️ Database Setup & Migrations

The relational schema is configured with Django models across apps:
- `accounts`: Django `User` model.
- `research_projects`: `ResearchProject` (owner, topic, timestamps).
- `documents`: `ResearchDocument` (checksum, processing_status, file, page count), `DocumentChunk` (page ranges, text, index vector IDs).
- `research_chat`: `ChatConversation` (threads), `ChatMessage` (roles, content), `MessageSource` (normalized citations).

Run migrations to initialize your database:

```bash
python manage.py makemigrations
python manage.py migrate
```

---

## 🤖 LLM & Embedding Model Configuration

### 1. Local Dense Embeddings (Always 100% Free)
ResearchMate AI automatically loads and caches `sentence-transformers/all-MiniLM-L6-v2`.
- **First-time run:** The model (~90 MB) downloads automatically to your local HuggingFace cache.
- **Offline support:** Once downloaded, vector generation runs completely offline without any internet connection.
- **Performance:** Generates 384-dimensional normalized float32 embeddings at high speed on standard modern CPUs.

### 2. Configurable Answer Generation Options

#### Option A: Offline / Deterministic Evaluator (`LLM_PROVIDER=mock`)
- **Default configuration.**
- Operates deterministically without external LLM calls, zero cost, and works offline.
- Synthesizes answers directly by quoting and citing the top retrieved passages using `[Source N]` tags.
- Verifies that evidence sufficiency checks and citation validation operate reliably.

#### Option B: Local Ollama LLM (`LLM_PROVIDER=ollama`)
Run open-weight models locally on your machine with zero cloud fees:
1. Install [Ollama](https://ollama.ai/).
2. Pull your model: `ollama run llama3.2` (or `ollama run mistral` / `deepseek-r1`).
3. Set in `.env`:
   ```ini
   LLM_PROVIDER=ollama
   LLM_MODEL_NAME=llama3.2
   OLLAMA_BASE_URL=http://localhost:11434
   ```

#### Option C: Hosted OpenAI-Compatible APIs (`LLM_PROVIDER=openai_compatible`)
Use hosted inference providers such as OpenAI, Groq, DeepSeek, Together AI, or OpenRouter:
```ini
LLM_PROVIDER=openai_compatible
LLM_MODEL_NAME=gpt-4o-mini
LLM_API_KEY=your_actual_api_key_here
LLM_API_BASE_URL=https://api.openai.com/v1
```
*(For Groq, set `LLM_API_BASE_URL=https://api.groq.com/openai/v1` and `LLM_MODEL_NAME=llama-3.3-70b-versatile`)*.

---

## 🚶 Step-by-Step User Walkthrough

### Step 1: Start the Development Server
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000` in your browser.

### Step 2: Register an Account
1. Click **Sign Up** in the top navigation bar.
2. Enter your username, institutional email, and a secure password (min 8 characters).
3. You are automatically logged in and redirected to the **Research Dashboard**.

### Step 3: Discover Academic Papers (Goal 1)
1. Click **Discover Papers** in the header.
2. Enter a topic (e.g., *"Attention Is All You Need for Kidney Diagnosis"* or *"Transformer models for protein folding"*).
3. Filter by start year (e.g. `2021`) and sort by **Most Cited** or **Most Relevant**.
4. Review the returned cards displaying genuine titles, author lists, reconstructed abstracts, venues, and citation counts.
5. Click **Open PDF Link** or **Publisher / Repository Record** to open the legitimate source and download the research paper PDF manually.

### Step 4: Create a Research Workspace (Goal 2)
1. Navigate to **Dashboard** &rarr; **+ New Research Project** (or **My Projects** &rarr; **+ Create Research Project**).
2. Enter:
   - **Project Name:** *"Early Kidney Disease Diagnosis with Machine Learning"*
   - **Research Topic:** *"Clinical Biomarkers and Neural Classification"*
   - **Description:** *"Evaluating multi-cohort deep learning models on renal biomarker panels."*
3. Click **Create Research Project**.

### Step 5: Upload & Ingest Research Papers
1. On your project detail page, click **Upload PDFs** (or **+ Add Papers**).
2. Select 1 to 30 research paper PDFs (up to 25 MB each).
3. Click **Upload & Index Papers**.
4. The system automatically executes:
   - Header inspection & PyMuPDF corruption verification.
   - Page-by-page text extraction and whitespace normalization.
   - Semantic chunking into passages with page range preservation.
   - Dense vector embedding generation via `all-MiniLM-L6-v2`.
   - Indexing into the project's private FAISS vector store.
   - Status updates from `extracting` &rarr; `chunking` &rarr; `embedding` &rarr; `indexed`.

### Step 6: Chat with Your Research Papers Using RAG
1. In the project workspace, click **Start New Chat**.
2. Type an inquiry:
   > *"What baseline models, evaluation metrics, and dataset limitations were discussed across the studies?"*
3. Press **Enter** to submit.
4. The assistant:
   - Scopes search strictly to your project's indexed papers.
   - Retrieves the most relevant passages above the similarity threshold.
   - Evaluates evidence sufficiency (if out of scope, politely communicates that the documents do not provide enough evidence).
   - Generates an evidence-grounded answer citing `[Source 1]`, `[Source 2]`.
   - Renders clickable citation chips beneath the response.
5. Click any citation chip (e.g. `[Source 1] kidney_biomarkers.pdf (p. 2)`) to open the authenticated PDF viewer directly at that page!

---

## 🧪 Automated Testing and Verification

ResearchMate AI includes a comprehensive test suite of **35 automated unit and integration tests** verifying every functional and security boundary:

```bash
# Run the entire test suite:
python manage.py test
```

### Test Coverage Breakdown

```
...................................
----------------------------------------------------------------------
Ran 35 tests in 56.298s

OK
Destroying test database for alias 'default'...
```

1. **Authentication Tests ([`accounts/tests.py`](file:///c:/SOLOMON/Research%20project/accounts/tests.py)) — 7 Tests:**
   - User registration success and password hashing.
   - Password mismatch validation.
   - Duplicate email rejection.
   - User login and invalid credential rejection.
   - Secure logout.
   - Protected URL redirect verification for unauthorized visitors.

2. **Workspace & Authorization Tests ([`research_projects/tests.py`](file:///c:/SOLOMON/Research%20project/research_projects/tests.py)) — 7 Tests:**
   - Workspace creation, detail views, and edit views.
   - Multi-user boundary: User A cannot view User B's project (HTTP 404).
   - Multi-user boundary: User A cannot edit User B's project (HTTP 404).
   - Multi-user boundary: User A cannot delete User B's project (HTTP 404).
   - Owner can delete their own project and cascade cleanup.
   - Dashboard metrics accurately reflect database counts.

3. **Academic Paper Discovery Tests ([`paper_discovery/tests.py`](file:///c:/SOLOMON/Research%20project/paper_discovery/tests.py)) — 8 Tests:**
   - OpenAlex token-inverted abstract reconstruction into natural sentences.
   - Empty inverted index fallback (`"Abstract unavailable."`).
   - Work metadata normalization (authors, venue, DOIs, open-access status).
   - OpenAlex API query handling with mocks (network-independent).
   - Empty search term validation.
   - OpenAlex API timeout handling.
   - HTTP 429 rate-limit awareness and feedback.
   - Search view template integration.

4. **PDF Processing & Ingestion Tests ([`documents/tests.py`](file:///c:/SOLOMON/Research%20project/documents/tests.py)) — 7 Tests:**
   - Academic text cleaning (rejoining hyphens, preserving scientific symbols and units).
   - PyMuPDF page-by-page text extraction.
   - Semantic chunking with page number tracking.
   - Scanned PDF detection flagging OCR requirement.
   - End-to-end ingestion pipeline updating database chunks and FAISS vectors.
   - Upload view validation and SHA-256 duplicate rejection within workspace.
   - **Private document security:** Owner can view PDF; unauthorized User B receives **HTTP 403 Forbidden**.

5. **RAG Pipeline & Chat Tests ([`research_chat/tests.py`](file:///c:/SOLOMON/Research%20project/research_chat/tests.py)) — 6 Tests:**
   - Project-scoped retrieval finding relevant chunks.
   - **Cross-project isolation:** Chunks from Project B never appear in Project A's queries.
   - Out-of-scope query producing the insufficient evidence notice.
   - Citation validator preserving valid source tags and stripping fabricated tags (`[Source 99]`).
   - End-to-end RAG query execution and database persistence of `ChatMessage` and `MessageSource`.
   - Multi-user chat security: User B cannot view or send questions to User A's conversation.

---

## 🔒 Security, Multi-User Isolation & Privacy

ResearchMate AI was engineered with security and data privacy at every layer:

1. **Private Document Storage Boundary:**
   - Uploaded PDF documents are placed in `media/private_documents/` outside the web server's public document root.
   - `config/urls.py` intentionally does **not** expose the private media directory to static file handlers.
   - Document access flows exclusively through `documents.views.serve_private_document_view`, which explicitly verifies `document.project.owner == request.user`.
2. **Multi-User Isolation:**
   - Every workspace, document, chunk, vector index, conversation, and message is strictly isolated by the owner foreign key.
   - Direct URL manipulation (e.g. attempting to load `/projects/42/` or `/chat/15/` belonging to another user) is rejected immediately with HTTP 404 or HTTP 403.
3. **Project-Scoped Vector Isolation:**
   - Vector indexes are isolated on disk per project (`project_<id>.index`).
   - Queries embed the user's question and search **only** the active project's index, making cross-project vector leakage impossible.
4. **Untrusted Content Treatment:**
   - Text extracted from PDFs is treated as untrusted source data, not system instructions.
   - Prompt engineering isolates source excerpts to prevent prompt injection attempts embedded inside research papers.
5. **No Silent API Usage:**
   - Embeddings are generated locally with zero API dependency.
   - External LLMs are never called unless an administrator or user explicitly configures an API key.

---

## ⚠️ Known Limitations

1. **Scanned & Image-Only PDFs:**
   - PyMuPDF extracts text directly from PDF text streams.
   - Documents with low extractable text (e.g., scanned historical photocopies) are automatically detected and flagged as requiring OCR. They are marked with status `failed` and a clear message rather than silently indexed as blank content.
2. **Context Window Limits:**
   - In accordance with RAG best practices, the system retrieves the top 5 most relevant passages (`RAG_TOP_K=5`). Queries requiring synthesis across 50+ papers simultaneously will prioritize the top semantically relevant passages within the token budget.
3. **Single-User Workspace Collaboration:**
   - Currently, workspaces are owned by a single researcher. Multi-user shared team workspaces with role-based access control (RBAC) are planned for the next release.

---

## 🔮 Future Work & Roadmap

The modular architecture of ResearchMate AI is designed to support several planned extensions:

### Phase 1: Advanced Ingestion & OCR
- **Integrated Tesseract / Nougat OCR:** Add background OCR worker support using `pytesseract` or Meta's `nougat` for scanned scientific manuscripts and historical archives.
- **Scientific Table & Formula Parsing:** Extract LaTeX mathematical equations and tabular relationships using specialized extractors (e.g., `pdfplumber` or `Grobid`).

### Phase 2: Hybrid Retrieval & Reranking
- **Dense + Sparse Hybrid Search:** Combine dense vector similarity (FAISS) with sparse keyword retrieval (BM25) using Reciprocal Rank Fusion (RRF) for better retrieval on exact technical terms and gene symbols.
- **Cross-Encoder Reranking:** Integrate a local cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` or `bge-reranker-base`) to score retrieved candidate passages before context construction.

### Phase 3: Research Group Collaboration
- **Supervisor & Lab Workspaces:** Enable project sharing with granular permission tiers (Owner, Co-Researcher, Reviewer).
- **Shared Annotated Citations:** Allow lab members to share saved passages, highlights, and annotations within a workspace.

### Phase 4: Literature Export & Synthesis Tools
- **Automated BibTeX Generation:** One-click export of properly formatted BibTeX and APA citations for all uploaded documents in a workspace.
- **Literature Review Synthesis Drafter:** Generate structured synthesis matrices comparing methodologies, sample sizes, and empirical findings across the entire 30-paper collection in tabular Markdown or LaTeX format.

### Phase 5: Asynchronous Background Ingestion
- **Celery & Redis Worker Queues:** Add background worker support for processing large batches of 30 multi-megabyte PDFs concurrently with WebSocket progress bars.

---

## 🚀 Deployment Guidance

### Production Deployment Checklist
1. **Environment:**
   - Set `DEBUG=False` in `.env`.
   - Configure a strong random `SECRET_KEY`.
   - Update `ALLOWED_HOSTS` with your production domain names.
2. **Database:**
   - Configure a managed PostgreSQL database with pgvector support:
     ```ini
     DATABASE_URL=postgresql://dbuser:dbpassword@dbhost:5432/researchmate_prod
     ```
   - Run `python manage.py migrate`.
3. **Static Assets:**
   - Run `python manage.py collectstatic --noinput`.
   - Configure Nginx to serve `/static/` directly from `STATIC_ROOT`.
4. **Private Documents:**
   - Ensure the `media/private_documents/` folder is not directly accessible via public HTTP.
   - For high-performance private downloads, use Nginx `X-Accel-Redirect` (or Apache `X-Sendfile`).
5. **Application Server:**
   - Run with Gunicorn or Uvicorn workers:
     ```bash
     gunicorn config.wsgi:application --workers 4 --bind 127.0.0.1:8000
     ```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👥 Acknowledgments & Academic Data Sources

- **[OpenAlex](https://openalex.org/):** For providing open, comprehensive bibliographic metadata across hundreds of millions of scholarly publications.
- **[PyMuPDF](https://github.com/pymupdf/PyMuPDF):** High-performance Python bindings for MuPDF.
- **[Sentence-Transformers](https://sbert.net/):** For efficient dense vector embedding models.
- **[FAISS](https://github.com/facebookresearch/faiss):** Efficient similarity search and clustering of dense vectors.

---
*Built with precision for researchers, academics, and scholars who value rigorous evidence grounding over generic AI hallucination.*
