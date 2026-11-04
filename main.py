from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama 

CHUNK_SIZE = 300
CHUNK_OVERLAP = 20
K = 3


loader = DirectoryLoader(
    "documents/",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_documents(documents)
print(f"Made {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

# 5. LLM
llm = ChatOllama(model="llama3.2:3b", temperature=0)

def ask(question, k=3):
    # RETRIEVE: find the best chunks
    docs = vector_store.similarity_search(question, k=k)

    # AUGMENT: paste the chunks into a prompt
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Question: {question}

Answer:"""

    # GENERATE: the LLM writes the answer
    answer = llm.invoke(prompt).content

    sources = sorted({doc.metadata["source"] for doc in docs})
    return answer, sources

questions = [
    "where is user information stored?",
    "how long does the access token last?",
    "what happens when a payment fails?",
    "how do refunds work?",
    "why does RAG reduce hallucination?",
    "how are passwords protected?",
]

for q in questions:
    answer, sources = ask(q)
    print("=" * 60)
    print("Q:", q)
    print("A:", answer)
    print("Sources:", sources)