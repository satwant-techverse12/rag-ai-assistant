 AI RAG Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDFs and ask questions based on document content.

---

 Features

* 📄 Upload PDF dynamically
* 🤖 Ask questions from documents
* 🔍 Semantic search using Pinecone
* 🧠 Context-aware responses using OpenAI
* 💬 Interactive chat UI (Flask)

---

 Tech Stack

* **Backend:** Flask (Python)
* **LLM:** OpenAI (GPT-4o-mini)
* **Embeddings:** OpenAI (text-embedding-3-small)
* **Vector DB:** Pinecone
* **Frontend:** HTML, CSS, JavaScript

---

##  How It Works

1. User uploads a PDF
2. PDF is split into chunks
3. Chunks are converted into embeddings
4. Stored in Pinecone (vector DB)
5. User asks a question
6. Relevant chunks are retrieved
7. AI generates answer based on context

---

##  Installation

```bash
git clone https://github.com/YOUR_USERNAME/rag-ai-assistant.git
cd rag-ai-assistant

pip install -r requirements.txt
```

---

##  Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_ENV=us-east-1
```

---

##  Run Project

```bash
python ingest.py
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 📸 Screenshots

(Add your screenshots here)

---

## 💼 Use Case

* Document Q&A system
* Knowledge base chatbot
* Research assistant

---

## 🚀 Future Improvements

* Source citation
* Chat memory
* Multi-document search
* Deployment (Render / Railway)

---

## 👨‍💻 Author

Your Name
