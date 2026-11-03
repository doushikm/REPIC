from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 100
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

questions = [
    "where is user information stored?",
    "how long does the access token last?",
    "what happens when a payment fails?",
    "how do refunds work?",
    "why does RAG reduce hallucination?",
    "how are passwords protected?",
]

for q in questions:
    print("=" * 60)
    print("Q:", q)

    print("\n  Normal search:")
    normal = vector_store.similarity_search_with_score(q, k=K)
    for doc, score in normal:
        print("   ", round(score, 3), "|", doc.metadata["source"])

    print("\n  MMR search:")
    mmr = vector_store.max_marginal_relevance_search(q, k=K, fetch_k=10)
    for doc in mmr:
        print("   ", doc.metadata["source"])
    print()