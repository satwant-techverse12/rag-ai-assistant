import os
from dotenv import load_dotenv
import pinecone

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ✅ INIT PINECONE (VERY IMPORTANT)
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

INDEX_NAME = "pdf-openai"

# embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# connect index
vectorstore = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# prompt
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
    return "\n\n".join(doc.page_content for doc in docs)


def ask_question(query):
    try:
        docs = retriever.invoke(query)

        # fallback if nothing found
        if not docs:
            return "No relevant information found in documents."

        context = format_docs(docs)

        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke({
            "context": context,
            "question": query
        })

        # ✅ ADD SOURCE CITATION (IMPORTANT FOR JOB)
        sources = set(doc.metadata.get("source", "unknown") for doc in docs)

        return f"{answer}\n\n📄 Sources: {', '.join(sources)}"

    except Exception as e:
        print("ERROR:", str(e))  # logs in Render
        return "❌ Something went wrong. Please try again."