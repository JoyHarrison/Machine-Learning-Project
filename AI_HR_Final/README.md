# 🤖 AI-Powered HR Assistant

An end-to-end **AI-driven HR application** built using Machine Learning and Natural Language Processing (NLP).  
This project helps HR teams analyze documents and retrieve intelligent answers from uploaded files using AI.

The application is fully **deployed using Streamlit**, showcasing the complete ML workflow from development to production.

---

## 🚀 Live Demo
🔗 **Deployed App:** https://machine-learning-project-q9hqkxfvr8apkoukx2tsaq.streamlit.app/

---

## 📌 Problem Statement
HR professionals spend a significant amount of time manually reviewing resumes and documents.  
This process is repetitive, time-consuming, and prone to inconsistencies.

**Objective:**  
To build an AI-powered system that can:
- Process HR-related documents automatically
- Retrieve relevant information efficiently
- Generate intelligent answers using NLP models

---

## 🧠 Solution
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline:
- Documents are uploaded and converted into text
- Text is split into chunks and embedded using a transformer model
- Embeddings are stored in a vector database (FAISS)
- User queries retrieve relevant chunks
- A language model generates accurate, context-aware answers

---

## ✨ Features
- 📄 Upload PDF documents
- 🔍 Semantic search using vector embeddings
- 🤖 AI-generated responses based on document content
- 🖥️ Interactive Streamlit web interface
- ☁️ Deployed and accessible online

---

## 🛠️ Tech Stack
- **Python**
- **Streamlit** – UI & deployment
- **Sentence Transformers** – Text embeddings
- **FAISS** – Vector similarity search
- **Hugging Face Transformers** – Language model
- **PyPDF** – PDF text extraction

---

## 📂 Project Structure
AI_HR_Final/
│
├── app.py # Main Streamlit application
├── core/
│ └── rag.py # RAG pipeline logic
│
├── data/
│ ├── uploads/ # Uploaded documents
│ └── vectorstore/ # FAISS index storage
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## ▶️ How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/JoyHarrison/Machine-Learning-Project.git
cd Machine-Learning-Project/AI_HR_Final
2️⃣ Create a Virtual Environment
python -m venv venv
Activate it:

Windows

venv\Scripts\activate
Mac/Linux

source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the App
streamlit run app.py
📊 What This Project Demonstrates
Practical application of Machine Learning & NLP

Understanding of Retrieval-Augmented Generation (RAG)

Ability to build production-ready ML applications

Experience with deployment and real-world constraints

Clean and modular project structure

🔮 Future Improvements
User authentication and role management

Improved UI/UX design

Support for more document formats

Model evaluation metrics and analytics

Dockerization for scalable deployment

👤 Author
Joy Harrison
Aspiring Machine Learning / AI Engineer

🔗 GitHub: https://github.com/JoyHarrison

⭐ If you find this project useful, feel free to star the repository!


---

### ✅ Final checklist (do this once)
- Paste into **Notepad**
- Save as **README.md**
- Save inside **AI_HR_Final**
- Commit & push to GitHub

Once this is live, your repo jumps from *“student project”* to *“entry-level ML engineer portfolio”* energy 💼🔥  

If you want next:
- I can write **resume bullets**
- A **LinkedIn post**
- Or tailor this README for **FAANG / startup recruiters**

Just say the word 🚀


