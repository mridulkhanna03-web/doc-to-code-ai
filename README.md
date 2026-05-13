# Documentation to Code: AI Assistant for Developers

A RAG-powered assistant that turns FastAPI documentation + a natural-language question into validated, runnable Python code examples and step-by-step tutorials.

> **Status:** under active construction. See the [4-day build plan](../.claude/plans/) for sequencing.

## Architecture

```
Streamlit UI  ─►  FastAPI backend  ─►  RAG pipeline
                                       ├─ Embed query (nomic-embed-text)
                                       ├─ ChromaDB k-NN retrieve (cos < 0.3)
                                       ├─ Prompt assembly with source attribution
                                       ├─ LLM: Ollama (qwen2.5-coder:1.5b) → Gemini fallback
                                       ├─ AST + ruff validation (max 2 regen attempts)
                                       └─ Response: {code, explanation, sources, valid}
```

## Setup

```bash
# 1. System deps (one-time)
sudo apt update && sudo apt install -y python3-pip python3-venv
curl -fsSL https://ollama.com/install.sh | sh

# 2. Models (one-time, ~1.3 GB total)
ollama pull qwen2.5-coder:1.5b
ollama pull nomic-embed-text

# 3. Python env
cd /home/mridul_khanna/sample_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# edit .env — add GEMINI_API_KEY

# 5. Ingest FastAPI docs into ChromaDB (~5-10 min)
python scripts/download_docs.py
python scripts/ingest_docs.py
```

## Run

```bash
# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
streamlit run ui/app.py
```

Open http://localhost:8501.

## Project layout

```
config/          central settings
src/
  ingestion/     doc fetch + parse + pipeline
  preprocessing/ chunker + cleaner
  embeddings/    nomic-embed-text via Ollama
  retrieval/     ChromaDB store + retriever
  generation/    LLM client + prompts + parser + tutorial gen
  validation/    AST syntax + ruff quality
  utils/         logger
api/             FastAPI app
ui/              Streamlit app
scripts/         download / ingest / demo entry points
tests/           pytest suite
```

## Tech stack

Python 3.10+, Ollama (qwen2.5-coder:1.5b + nomic-embed-text), Google Gemini Pro (fallback), LangChain 0.1.x, ChromaDB 0.4.22, FastAPI, Streamlit, BeautifulSoup4, ruff, loguru.

## Author

Mridul Khanna (221302183), B.Tech CSE (Data Science), SGT University. Final-year project, May 2026.
