import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from retriever import Retriever

K = 4
DB_PATH = "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = Retriever(vector_store)

llm = ChatOllama(model="llama3.2:3b", temperature=0)

def ask(question):
    # RETRIEVE: hybrid search with a small boost for code files
    docs = retriever.search(question, k=K, mode="hybrid", boost=0.003)

    # AUGMENT
    context = "\n\n".join(
        f"File: {d.metadata['source']}\n{d.page_content}" for d in docs
    )
    prompt = f"""You are answering questions about a code repository.
Use ONLY the context below. Do not invent examples, values, or function names
that are not in the context. Answer in at most 5 sentences.
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
    "how does basic authentication work?",
    "what does raise_for_status do?",
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