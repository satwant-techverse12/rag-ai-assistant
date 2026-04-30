import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import uuid

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = "pdf-openai"

# init pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

# create index if not exists
existing_indexes = [index.name for index in pc.list_indexes().indexes]

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

# text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

#FUNCTION 1: Bulk load (initial setup)
def ingest_all_documents():
    loader = PyPDFDirectoryLoader("documents")
    docs = loader.load()

    chunks = splitter.split_documents(docs)

    # add metadata + unique id
    for chunk in chunks:
        chunk.metadata["source"] = "bulk_upload"
        chunk.metadata["id"] = str(uuid.uuid4())

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print("✅ All documents indexed successfully!")


#FUNCTION 2: Single file upload (
def process_pdf(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    chunks = splitter.split_documents(docs)

    filename = os.path.basename(filepath)

    # add metadata
    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["id"] = str(uuid.uuid4())

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print(f"✅ {filename} indexed successfully!")



if __name__ == "__main__":
    ingest_all_documents()