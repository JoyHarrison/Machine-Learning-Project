import os
import uuid
import json
import shutil
from datetime import datetime

import streamlit as st
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ======================================================
# CONFIG
# ======================================================

BASE_DIR = "data"
COMPANIES_META = os.path.join(BASE_DIR, "companies.json")
USAGE_LOG = os.path.join(BASE_DIR, "usage_log.jsonl")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

DEMO_COMPANY_ID = "demo-company"
DEMO_COMPANY_NAME = "Demo Company (Portfolio)"
DEMO_DIR = "demo"
DEMO_PDF = os.path.join(DEMO_DIR, "demo_policy.pdf")
DEMO_QUESTIONS = os.path.join(DEMO_DIR, "demo_questions.json")

os.makedirs(BASE_DIR, exist_ok=True)

# ======================================================
# GREETINGS
# ======================================================

GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}

def is_greeting(text: str) -> bool:
    return text.lower().strip() in GREETINGS

# ======================================================
# COMPANY STORAGE
# ======================================================

def load_companies():
    companies = {}
    if os.path.exists(COMPANIES_META):
        with open(COMPANIES_META, "r", encoding="utf-8") as f:
            companies = json.load(f)
    companies.setdefault(DEMO_COMPANY_ID, {"name": DEMO_COMPANY_NAME})
    return companies

def save_companies(data):
    data = {k: v for k, v in data.items() if k != DEMO_COMPANY_ID}
    with open(COMPANIES_META, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def create_company(name):
    companies = load_companies()
    cid = str(uuid.uuid4())
    companies[cid] = {"name": name}
    save_companies(companies)
    os.makedirs(os.path.join(BASE_DIR, "uploads", cid), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "vectorstores", cid), exist_ok=True)

def delete_company(cid):
    if cid == DEMO_COMPANY_ID:
        return
    companies = load_companies()
    if cid not in companies:
        return
    companies.pop(cid)
    save_companies(companies)
    shutil.rmtree(os.path.join(BASE_DIR, "uploads", cid), ignore_errors=True)
    shutil.rmtree(os.path.join(BASE_DIR, "vectorstores", cid), ignore_errors=True)

def get_company_dirs(cid):
    uploads = os.path.join(BASE_DIR, "uploads", cid)
    vs = os.path.join(BASE_DIR, "vectorstores", cid)
    os.makedirs(uploads, exist_ok=True)
    os.makedirs(vs, exist_ok=True)
    return uploads, vs

# ======================================================
# MODELS
# ======================================================

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    return tokenizer, model

embedder = load_embedder()
tokenizer, model = load_llm()

# ======================================================
# PDF CHUNKING
# ======================================================

def extract_pdf_chunks(path, chunk_size=600, overlap=100):
    reader = PdfReader(path)
    chunks = []
    chunk_id = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if len(chunk) > 80:
                chunks.append({
                    "id": chunk_id,
                    "text": chunk,
                    "page": page_num
                })
                chunk_id += 1
            start = end - overlap
    return chunks

# ======================================================
# FAISS
# ======================================================

def build_index(chunks, vs_dir):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(vs_dir, "index.faiss"))
    with open(os.path.join(vs_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

def retrieve_top_chunk(vs_dir, query):
    index_path = os.path.join(vs_dir, "index.faiss")
    chunks_path = os.path.join(vs_dir, "chunks.json")
    if not os.path.exists(index_path):
        return None
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    q_emb = embedder.encode([query]).astype("float32")
    _, I = index.search(q_emb, 1)
    return chunks[I[0][0]]

# ======================================================
# DEMO HELPERS
# ======================================================

def ensure_demo_index():
    _, vs_dir = get_company_dirs(DEMO_COMPANY_ID)
    if os.path.exists(os.path.join(vs_dir, "index.faiss")):
        return
    if not os.path.exists(DEMO_PDF):
        return
    chunks = extract_pdf_chunks(DEMO_PDF)
    build_index(chunks, vs_dir)

def show_demo_document_status():
    if not os.path.exists(DEMO_PDF):
        st.warning("Demo policy document not found.")
        return
    reader = PdfReader(DEMO_PDF)
    st.markdown("### 📄 HR Policy Document")
    st.success("Demo policy document is loaded")
    st.markdown(f"**Pages:** {len(reader.pages)}")

# ======================================================
# ANSWER GENERATION
# ======================================================

def generate_answer(question, chunk):
    if not chunk:
        return "I do not have that information in the company documents."
    prompt = f"""
Answer the question using ONLY the HR policy text below.

HR policy text:
{chunk['text']}

Question: {question}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.0)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ======================================================
# UI: EMPLOYEE
# ======================================================

def employee_ui():
    st.title("🧑‍💼 AI HR Assistant")

    companies = load_companies()
    name_to_id = {v["name"]: k for k, v in companies.items()}
    company = st.selectbox("🏢 Select company", list(name_to_id.keys()))
    cid = name_to_id[company]
    uploads, vs_dir = get_company_dirs(cid)

    if cid == DEMO_COMPANY_ID:
        ensure_demo_index()
        show_demo_document_status()
        if os.path.exists(DEMO_QUESTIONS):
            with open(DEMO_QUESTIONS) as f:
                for q in json.load(f):
                    if st.button(q):
                        st.session_state["auto_question"] = q

    with st.expander("📄 Upload HR Policy PDFs"):
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process & Index") and uploaded_files:
            progress = st.progress(0)
            total_chunks = 0

            with st.spinner("📄 Processing and indexing documents..."):
                for i, uploaded in enumerate(uploaded_files, start=1):
                    path = os.path.join(uploads, uploaded.name)
                    with open(path, "wb") as f:
                        f.write(uploaded.read())

                    chunks = extract_pdf_chunks(path)
                    total_chunks += len(chunks)
                    build_index(chunks, vs_dir)

                    progress.progress(i / len(uploaded_files))

            st.success(
                f"✅ Indexed {len(uploaded_files)} file(s) "
                f"with {total_chunks} total sections"
            )

    st.divider()

    chat_key = f"chat_{cid}"
    st.session_state.setdefault(chat_key, [])

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed_question = st.chat_input("Ask a policy question")
    question = st.session_state.get("auto_question") or typed_question

    if not question:
        return

    if "auto_question" in st.session_state:
        del st.session_state["auto_question"]

    st.session_state[chat_key].append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        chunk = retrieve_top_chunk(vs_dir, question)
        answer = generate_answer(question, chunk)
        st.markdown(answer)
        if chunk:
            st.markdown(f"📌 Page {chunk['page']} · Chunk {chunk['id']}")

    st.session_state[chat_key].append({"role": "assistant", "content": answer})

# ======================================================
# UI: HISTORY
# ======================================================

def history_ui():
    st.subheader("📜 History")
    if not os.path.exists(USAGE_LOG):
        return
    with open(USAGE_LOG) as f:
        for l in reversed(f.readlines()[-20:]):
            entry = json.loads(l)
            st.markdown(f"**Q:** {entry['question']}\n\n**A:** {entry['answer']}")
            st.divider()

# ======================================================
# UI: ADMIN
# ======================================================

def admin_ui():
    st.subheader("🛠 Admin Panel")

    for cid, info in load_companies().items():
        if cid == DEMO_COMPANY_ID:
            continue
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{info['name']}**")
        with col2:
            if st.button("Delete", key=f"delete_{cid}"):
                delete_company(cid)
                st.rerun()

    st.divider()

    name = st.text_input("Create new company")
    if st.button("Create") and name.strip():
        create_company(name.strip())
        st.success("Company created")
        st.rerun()

# ======================================================
# MAIN
# ======================================================

def main():
    st.set_page_config(
        page_title="AI HR Assistant",
        page_icon="🧑‍💼",
        layout="wide"
    )
    tab1, tab2, tab3 = st.tabs(["Employee", "History", "Admin"])
    with tab1:
        employee_ui()
    with tab2:
        history_ui()
    with tab3:
        admin_ui()

if __name__ == "__main__":
    main()
