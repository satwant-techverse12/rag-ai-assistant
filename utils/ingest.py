from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_DB_PATH = "faiss_index"

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env")


# 🔥 Better text cleaning
def clean_text(text):
    return " ".join(text.replace("\n", " ").replace("\t", " ").split())


def process_pdf(filepath):
    print("Processing:", filepath)

    loader = PyPDFLoader(filepath)
    documents = loader.load()

    print("Pages loaded:", len(documents))

    # 🔥 Clean + preserve structure
    cleaned_docs = []
    for i, doc in enumerate(documents):
        text = clean_text(doc.page_content)

        # skip empty pages
        if len(text) < 50:
            continue

        doc.page_content = text
        doc.metadata["page"] = i
        cleaned_docs.append(doc)

    print("Valid pages:", len(cleaned_docs))

    # 🔥 MUCH BETTER chunking (KEY FIX)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # bigger chunks = more context
        chunk_overlap=200,        # overlap for continuity
        separators=["\n\n", "\n", ".", " ", ""]
    )

    docs = splitter.split_documents(cleaned_docs)

    print("Chunks created:", len(docs))

    # 🔥 Add metadata (IMPORTANT for debugging + future features)
    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = i
        doc.metadata["source"] = os.path.basename(filepath)

    # 🔥 Better embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 🔥 Create / update FAISS index
    if os.path.exists(VECTOR_DB_PATH):
        print("Loading existing index...")
        db = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        db.add_documents(docs)
        print("Added new documents")
    else:
        print("Creating new index...")
        db = FAISS.from_documents(docs, embeddings)

    db.save_local(VECTOR_DB_PATH)
    print("Index saved successfully")