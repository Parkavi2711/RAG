import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# Folder containing PDFs
DATA_PATH = "data/blockchain"
VECTOR_PATH = "vector_store"

# Load PDFs
documents = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages.")

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks.")

# Create embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in Chroma
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=VECTOR_PATH
)

vectordb.persist()

print("Vector database created successfully!")