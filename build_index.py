import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_repo

REPO_PATH = "requests"
DB_PATH = "chroma_db"

documents = load_repo(REPO_PATH)
print(f"Loaded {len(documents)} files")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"Made {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# start fresh so re-running doesn't create duplicate chunks
shutil.rmtree(DB_PATH, ignore_errors=True)

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH,
)
print("Index saved to", DB_PATH)