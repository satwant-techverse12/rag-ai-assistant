import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

INDEX_NAME = "pdf-openai"

#embeddings (FIXED)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# connect index
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LLM (better)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ✅ strong prompt
template = """
You are a helpful assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""

prompt = PromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def ask_question(query):
    docs = retriever.invoke(query)

    # ✅ fallback safety
    if not docs:
        return "No relevant information found in documents."

    context = format_docs(docs)

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": query
    })