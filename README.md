# RAG Application

A Retrieval-Augmented Generation (RAG) project that loads documents from multiple formats, builds a FAISS vector store using sentence-transformers embeddings, and supports query-based retrieval and optional LLM-driven summarization.

## Features

- Load documents from `data/` as PDF, TXT, CSV, Excel, Word, and JSON
- Split documents into chunks and generate embeddings with `SentenceTransformers`
- Persist a FAISS vector index for fast semantic search
- Query the vector store from `app.py`
- Optional RAG summarization using `langchain-groq` and `ChatGroq`

## Requirements

- Python 3.11+
- Virtual environment recommended

## Python Dependencies

The project depends on:

- `langchain`
- `langchain-core`
- `langchain-community`
- `sentence-transformers`
- `pypdf`
- `pymupdf`
- `chromadb`
- `langchain-groq`
- `python-dotenv`

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install requirements:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the repository root to hold secrets (this file must NOT be committed):

```text
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Build and query the FAISS store

Run `app.py` to load documents, build or load a FAISS index, and execute a sample query:

```powershell
python app.py
```

This script:

- reads documents from `data/`
- builds the vector store under `faiss_store/` if needed
- queries the index for a sample question
- prints relevant document snippets

### Use the modules directly

The code is organized into reusable modules:

- `src/data_loader.py` – loads supported file types from `data/`
- `src/embedding.py` – splits text into chunks and generates embeddings
- `src/vectorstore.py` – manages FAISS persistence and vector search
- `src/search.py` – optional RAG search pipeline using Groq

### Optional RAG search

To use the `RAGSearch` pipeline, set `GROQ_API_KEY` in `.env` and call the class from your own script:

```python
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

store = FaissVectorStore('faiss_store')
store.load()
rag = RAGSearch(vector_store=store)
answer = rag.search_and_summarize('What is attention mechanism?', top_k=3)
print(answer)
```

## Data structure

- `data/` – input document files for indexing
- `faiss_store/` – persisted FAISS index and metadata
- `notebook/` – example Jupyter notebooks for experimentation

## Supported document formats

- PDF (`.pdf`)
- Text (`.txt`)
- CSV (`.csv`)
- Excel (`.xlsx`)
- Word (`.docx`)
- JSON (`.json`)

## License

This repository does not include a license file. Add one if you plan to publish or share the project publicly.
