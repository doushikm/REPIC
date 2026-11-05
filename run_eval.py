import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from retriever import Retriever

K = 4
DB_PATH = "chroma_db"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = Retriever(vector_store)

with open("eval/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

SETUPS = [
    ("vector only",              dict(mode="vector")),
    ("bm25 only",                dict(mode="bm25")),
    ("hybrid",                   dict(mode="hybrid")),
    ("hybrid + boost 0.003",     dict(mode="hybrid", boost=0.003)),
    ("hybrid + boost 0.01",      dict(mode="hybrid", boost=0.01)),
    ("hybrid + rerank",          dict(mode="hybrid", rerank=True)),
    ("hybrid + boost 0.01 + rerank", dict(mode="hybrid", boost=0.01, rerank=True)),
]

print(f"{'Setup':32} {'File':>6} {'Chunk':>7}")
for name, opts in SETUPS:
    file_hits = chunk_hits = chunk_total = 0
    failed = []
    for item in questions:
        q, expected, must = item["question"], set(item["expected"]), item.get("must_contain")
        docs = retriever.search(q, k=K, **opts)

        file_ok = bool(expected & {d.metadata["source"] for d in docs})
        file_hits += file_ok
        if not file_ok:
            failed.append(q)
        if must:
            chunk_total += 1
            chunk_hits += any(must in d.page_content for d in docs)

    print(f"{name:32} {file_hits:>3}/{len(questions)} {chunk_hits:>4}/{chunk_total}")
    for q in failed:
        print("      FAIL:", q)