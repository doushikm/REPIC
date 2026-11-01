from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# 1. Load all .txt files from the documents folder
loader = DirectoryLoader(
    "documents/",
    glob="*.txt",
    loader_cls=TextLoader
)

documents = loader.load()

print(f"Loaded {len(documents)} documents")

for document in documents:
    print(document.metadata["source"])

# 2. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Convert documents into embeddings and store them
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

# 4. Ask a question
question = "where is user information stored?"

# 5. Retrieve the most relevant document
results = vector_store.similarity_search(question, k=1)

# 6. Display the result
for result in results:
    print(result.page_content)