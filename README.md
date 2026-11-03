# RepoMind: Ask Questions About a GitHub Repo (RAG)

Ask questions about a codebase in plain English and get answers
with source file references. *(Work in progress)*

**How it works:** Load files → Chunk → Embed (MiniLM) → Store (ChromaDB) → Retrieve → LLM answers

**Stack:** Python, LangChain, ChromaDB, Ollama

**Run it:**
```bash
pip install -r requirements.txt
python rag.py
```

**What I learned so far**
- Smaller chunks give closer matches but can crowd results with one file
- MMR search helps diversify results

See [experiments](docs/experiments.md) for details.