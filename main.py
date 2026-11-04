import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

K = 4
DB_PATH = "chroma_db"

# must be the SAME embedding model used in build_index.py
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# load the saved index from disk (no re-embedding)
vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
)

llm = ChatOllama(model="llama3.2:3b", temperature=0)

def ask(question):
    # RETRIEVE
    docs = vector_store.similarity_search(
      question, k=K, filter={"file_type": ".py"}
    )

    # AUGMENT: label each chunk with its file so the LLM knows where it came from
    context = "\n\n".join(
        f"File: {d.metadata['source']}\n{d.page_content}" for d in docs
    )
    prompt = f"""You are answering questions about a code repository.
Answer using ONLY the context below.
If the answer is not in the context, say "I don't know based on the repository."

Context:
{context}

Question: {question}

Answer:"""

    # GENERATE
    answer = llm.invoke(prompt).content
    sources = sorted({d.metadata["source"] for d in docs})
    return answer, sources

questions = [
    "how does requests handle redirects?",
    "where is the Session class defined?",
    "how does basic authentication work?",
    "what exceptions can requests raise?",
    "how are cookies handled?",
]

for q in questions:
    start = time.time()
    answer, sources = ask(q)
    print("=" * 60)
    print("Q:", q)
    print("A:", answer)
    print("Sources:", sources)
    print(f"({time.time() - start:.1f}s)")