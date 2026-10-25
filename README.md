# ResearchMate AI — Academic Research Discovery & Grounded Multi-PDF RAG

**ResearchMate AI** is a full-stack academic research platform engineered with Python, Django, PyMuPDF, sentence-transformers, and FAISS. It provides two core capabilities for researchers and students:

1. **Academic Paper Discovery (Goal 1):** Search genuine scientific literature via the OpenAlex API, inspect publication venues, author lists, reconstructed abstracts, citation metrics, and access direct publisher, DOI, and open-access PDF download links.
2. **Evidence-Grounded Research Chat Using RAG (Goal 2):** Create isolated research project workspaces, upload up to 30 research paper PDFs per workspace, and query document collections via a ChatGPT-style interface with strict evidence grounding, page-level citations, and multi-user privacy isolation.

---

## 🏛️ System Architecture

```
                                  [ Researcher / Browser ]
                                             │
                                             ▼
                      [ Django Full-Stack Application (Port 8000) ]
                                             │
      ┌──────────────────┬───────────────────┼───────────────────┬──────────────────┐
      │                  │                   │                   │                  │
      ▼                  ▼                   ▼                   ▼                  ▼
[ Accounts & Auth ] [ Workspaces ] [ OpenAlex Discovery ] [ Documents Ingestion ] [ RAG Chat ]
  - Sessions          - Projects       - REST Works API     - PyMuPDF Extraction   - Scoped Retriever
  - Ownership Guard   - Limits (30)    - Polite Pool        - Chunker (500-800w)   - LLM Synthesizer
                                       - Inverted Abstracts - Private File Store   - Citation Validator
                                                                 │                      │
                                                                 ▼                      ▼
                                                      [ Local Embeddings ]    [ FAISS Vector Store ]
                                                      - all-MiniLM-L6-v2      - project_{id}.index
                                                      - 384-d float32         - project_{id}_map.json
```

---

## 🚀 Key Features

- **Genuine Academic Literature Search:** Live queries to OpenAlex Works API; reconstructs token-inverted index abstracts into natural text; displays open-access status, citation counts, and direct links without fabricating data.
- **Strict Evidence Grounding:** System instructions prevent external hallucinations. If documents lack sufficient evidence for a question, ResearchMate explicitly communicates the limitation.
- **Verifiable Page-Level Citations:** Assistant responses include clickable citation chips that link directly to the authorized document viewer at the cited page number (`#page=N`).
- **Multi-User & Project Isolation:** Strict ownership verification at every private resource boundary (URLs, views, downloads, vector indices). User A cannot view, download, or query User B's documents or conversations.
- **Private Document Storage:** PDFs are stored in protected storage (`media/private_documents/`) and are never exposed via public static/media URLs. Files are streamed only through authenticated, ownership-checked views.
- **Local AI Embedding Pipeline:** Uses `sentence-transformers/all-MiniLM-L6-v2` locally on CPU/GPU. No external API keys or paid services required for PDF extraction, chunking, embedding, or indexing.
- **Configurable LLM Generation:** Supports offline/mock test provider (zero cost), local Ollama (`llama3.2`), and hosted OpenAI-compatible APIs (Groq, Together, DeepSeek, OpenAI, Gemini).

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| **Backend Framework** | Python 3.12+ & Django 6.1 / 5.1 |
| **Database** | PostgreSQL (Production) / SQLite (Development) |
| **Document Processing** | PyMuPDF (`fitz`) |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-d) |
| **Vector Indexing** | Persistent project-isolated FAISS (`IndexFlatIP`) |
| **Literature Provider** | OpenAlex Works API (`httpx`) |
| **Frontend** | Semantic HTML5, Vanilla CSS3 (Academic Theme), Vanilla JavaScript |

---

## 📋 Prerequisites

- **Python:** Version 3.12 or compatible (3.11–3.13)
- **Pip:** Version 24+
- *(Optional)* **Ollama:** If using local open-weight LLMs (`ollama run llama3.2`)
- *(Optional)* **PostgreSQL & pgvector:** If deploying with PostgreSQL rather than local SQLite

---

## ⚙️ Installation & Setup

### 1. Clone & Set Up Virtual Environment

```bash
# In the project workspace directory:
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Activate on Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Customize the values in `.env`:

```ini
SECRET_KEY=your-secure-django-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database (Leave empty for default local SQLite):
DATABASE_URL=

# OpenAlex Polite Pool (Provides faster, higher rate limits):
OPENALEX_MAILTO=your.email@university.edu

# Local Embeddings:
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# LLM Provider ('mock' for offline/tests, 'ollama' for local, 'openai_compatible' for API):
LLM_PROVIDER=mock
LLM_MODEL_NAME=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

### 4. Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Create Superuser (Optional)

```bash
python manage.py createsuperuser
```

### 6. Start the Development Server

```bash
python manage.py runserver
```

Open your browser at: **`http://127.0.0.1:8000`**

---

## 🧪 Running Automated Tests

ResearchMate AI includes a comprehensive automated test suite covering:
- Authentication & registration guards (`accounts/tests.py`)
- Workspace CRUD & multi-user isolation boundaries (`research_projects/tests.py`)
- OpenAlex queries, normalizers, timeouts, rate-limits, and inverted abstracts (`paper_discovery/tests.py`)
- PyMuPDF extraction, text cleaning, chunking, OCR checks, and private file authorization (`documents/tests.py`)
- Scoped RAG retrieval, cross-project isolation, evidence sufficiency, citation validation, and message persistence (`research_chat/tests.py`)

Run all tests:

```bash
python manage.py test
```

Run a specific module:

```bash
python manage.py test accounts
python manage.py test research_projects
python manage.py test paper_discovery
python manage.py test documents
python manage.py test research_chat
```

---

## 📖 User Workflow Guide

### Workflow 1: Academic Paper Discovery
1. Click **Discover Papers** in the top navigation bar.
2. Enter your research topic or keywords (e.g., *"Attention Is All You Need for Kidney Diagnosis"*).
3. Optionally filter by start year, end year, or sort by citations / publication date.
4. Review the returned genuine metadata: reconstructed abstract, publication venue, author lists, and citation metrics.
5. Click **Open PDF Link** or **Publisher / Repository Record** to open the legitimate external source and download the research paper PDF manually.

### Workflow 2: Workspace Creation & PDF Upload
1. Sign up or log in to your account.
2. From the **Dashboard**, click **+ New Research Project**.
3. Name your project (e.g., *"Diabetic Retinopathy Classification"*), enter the research topic, and save.
4. Inside the project workspace, click **Upload PDFs**.
5. Select up to 30 research papers (up to 25 MB per PDF).
6. ResearchMate AI automatically:
   - Validates PDF signatures and checks for password protection or corruption.
   - Extracts page-by-page text with PyMuPDF.
   - Flags scanned/image-only documents requiring OCR.
   - Splits text into overlapping semantic passages (~500–800 words) with page tracking.
   - Computes local normalized sentence embeddings via `all-MiniLM-L6-v2`.
   - Indexes vectors into the project's private FAISS index.

### Workflow 3: Evidence-Grounded Research Chat
1. In the project workspace, click **Start New Chat**.
2. Type your question (e.g., *"What evaluation metrics and baseline models were compared across these studies?"*).
3. The RAG pipeline:
   - Embeds your query and searches *only* the active project's vector store.
   - Evaluates similarity against the configured threshold.
   - If no relevant evidence exists, clearly informs you that the uploaded documents do not contain sufficient evidence.
   - If evidence is found, synthesizes a grounded answer citing `[Source 1]`, `[Source 2]`, etc.
   - Validates that every citation maps to an authentic chunk and page.
   - Displays clickable citation chips beneath the answer that open the PDF at the exact page.

---

## 🔒 Security & Privacy Architecture

- **Multi-User Isolation:** Every database lookup enforces `owner=request.user` or `project__owner=request.user`. Users cannot access another researcher's projects, files, or chat threads even by guessing URLs or IDs.
- **Private Storage:** Uploaded PDFs are stored outside the public document root in `media/private_documents/`. They are streamed exclusively via `documents.views.serve_private_document_view` after verifying project ownership.
- **PDF Text as Untrusted Input:** Extracted text is treated strictly as data, never as system instructions.
- **Zero Silent Paid Calls:** Embedding generation is 100% local and free. LLM providers default to the local/offline evaluator, requiring deliberate administrator configuration before invoking external APIs.

---

## 🌐 Production Deployment Notes

1. **Web Server:** Deploy using Gunicorn or Uvicorn behind Nginx.
2. **PostgreSQL + pgvector:** Set `DATABASE_URL=postgresql://user:pass@host:5432/researchmate` in `.env`.
3. **Environment Security:** Set `DEBUG=False` and configure `ALLOWED_HOSTS`.
4. **Static Assets:** Run `python manage.py collectstatic --noinput` and configure Nginx to serve `/static/`.
5. **Private Media:** Ensure Nginx blocks direct access to `/media/private_documents/`, delegating serving to Django or using Nginx `X-Accel-Redirect`.
