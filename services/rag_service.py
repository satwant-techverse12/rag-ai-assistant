from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

VECTOR_DB_PATH = "faiss_index"

# API key check
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env")

# Models
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Load FAISS
def load_vectorstore():
    if not os.path.exists(VECTOR_DB_PATH):
        raise ValueError("NO_INDEX")

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# 🔥 IMPROVED PROMPT (VERY IMPORTANT)
prompt_template = """
You are a highly accurate document assistant.

Instructions:
- Answer ONLY using the provided context.
- Do NOT use outside knowledge.
- Extract exact information when possible.
- If the answer is partially available, explain clearly.
- If nothing relevant exists, say: "No relevant information found in document."

Make your answer:
- Clear
- Structured
- Concise

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Build QA chain
def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,          # increased recall
            "fetch_k": 30    # better diversity
        }
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,  # enable debugging
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa, retriever

# Main function
def ask_question(query):
    try:
        vectorstore = load_vectorstore()
        qa, retriever = build_chain(vectorstore)

        # Debug retrieval
        docs = retriever.get_relevant_documents(query)
        print(f"\nRetrieved {len(docs)} documents")

        if not docs:
            return "No relevant information found in document."

        for d in docs[:2]:
            print("----")
            print(d.page_content[:200])

        result = qa({"query": query})

        answer = result.get("result", "").strip()

        # 🔥 fallback improvement
        if not answer or "I don't know" in answer.lower():
            return "No relevant information found in document."

        return answer

    except ValueError as ve:
        if str(ve) == "NO_INDEX":
            return "Please upload a PDF first."

        print("VALUE ERROR:", str(ve))
        return "Configuration error."

    except Exception as e:
        print("ERROR:", str(e))
        return "Something went wrong. Check server logs."