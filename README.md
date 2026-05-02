# 📄 Document Q&A — CLI-Based RAG

A command-line RAG (Retrieval-Augmented Generation) tool that lets you ingest files or directories and ask questions about their content interactively.

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Load Docs  │ ──▶ │  Split into  │ ──▶ │  Embed &     │ ──▶ │  Query via  │
│  (files/dir)│     │  Chunks      │     │  Store in    │     │  dynamic    │
│             │     │  (1000 char) │     │  ChromaDB    │     │  prompt     │
└─────────────┘     └─────────────┘     └──────────────┘     └─────────────┘
```

1. **Ingestion** — Documents are loaded via `UnstructuredLoader` (supports PDF, TXT, and more), split into 1000-character chunks with 200-character overlap, and stored in a local ChromaDB vector store.
2. **Querying** — User queries are matched against stored chunks via similarity search. The top 4 results are injected as context into a system prompt using LangChain's `@dynamic_prompt` middleware, and the LLM generates an answer in a single pass.

## Tech Stack

| Component      | Choice                                      |
|----------------|---------------------------------------------|
| LLM            | `google/gemma-4-26B-A4B-it` via HuggingFace |
| Embeddings     | `sentence-transformers/all-mpnet-base-v2`    |
| Vector Store   | ChromaDB (persisted locally)                 |
| Doc Loader     | `UnstructuredLoader` (PDF, TXT, etc.)        |
| CLI Framework  | Click                                        |
| Orchestration  | LangChain `create_agent` + `dynamic_prompt`  |

## Setup

### Prerequisites

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd document_qna

# Install dependencies
uv sync
```

### Environment Variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
UNSTRUCTURED_API_KEY=your_key_here
```

You can get these from:
- [HuggingFace Settings → Access Tokens](https://huggingface.co/settings/tokens)
- [Unstructured API](https://unstructured.io/)

## Usage

```bash
# Ingest a single file and start querying
uv run main.py hello.txt

# Ingest an entire directory
uv run main.py ./data

# Ingest a PDF
uv run main.py resume.pdf
```

### Example Session

```
$ uv run main.py hello.txt
Setting up models and vector store...

Ingesting from: hello.txt
Processing file: hello.txt
Loaded 4 document(s)
Split documents into 4 sub-documents.
Stored 4 chunks in vector store

Ready! Ask questions about your documents (type 'exit' to quit)

You > who is the author?
================================== Ai Message ==================================
The author is Akshat Sipany.

You > exit
Goodbye!
```

## Project Structure

```
document_qna/
├── main.py              # All application logic
├── .env                 # API keys (not committed)
├── pyproject.toml       # Project metadata & dependencies
├── chroma_langchain_db/ # Persisted vector store (auto-created)
└── data/                # Sample documents
```
