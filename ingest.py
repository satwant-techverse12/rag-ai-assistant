import os
import uuid
from dotenv import load_dotenv

import pinecone  # ✅ OLD SDK

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone  # ✅ correct import

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = "pdf-openai"

# ✅ initialize pinecone (OLD STYLE)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

# embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

# create index if not exists
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine"
    )

# text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# FUNCTION 1: Bulk load
def ingest_all_documents():
    loader = PyPDFDirectoryLoader("documents")
    docs = loader.load()

    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["source"] = "bulk_upload"
        chunk.metadata["id"] = str(uuid.uuid4())

    Pinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print("✅ All documents indexed successfully!")


# FUNCTION 2: Single file upload
def process_pdf(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    chunks = splitter.split_documents(docs)

    filename = os.path.basename(filepath)

    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["id"] = str(uuid.uuid4())

    Pinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print(f"✅ {filename} indexed successfully!")


if __name__ == "__main__":
    ingest_all_documents()