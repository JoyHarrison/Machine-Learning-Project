import os
import uuid
import json
import shutil

import streamlit as st
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ======================================================
# PATHS (ABSOLUTE — STREAMLIT SAFE)
# ======================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.join(APP_DIR, "data")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstores")

DEMO_DIR = os.path.join(APP_DIR, "demo")
DEMO_PDF = os.path.join(DEMO_DIR, "demo_policy.pdf")
DEMO_QUESTIONS = os.path.join(DEMO_DIR, "demo_questions.json")

COMPANIES_META = os.path.join(BASE_DIR, "companies.json")

DEMO_COMPANY_ID = "demo-company"
DEMO_COMPANY_NAME = "Demo Company (Portfolio)"

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ======================================================
# MODELS
# ======================================================

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

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
# COMPANY STORAGE
# ======================================================

def load_companies():
    if os.path.exists(COMPANIES_META):
        with open(COMPANIES_META, "r", encoding="utf-8") as f:
            companies = json.load(f)
    else:
        companies = {}

    companies.setdefault(DEMO_COMPANY_ID, {"name": DEMO_COMPANY_NAME})
    return companies

def save_companies(data):
    clean = {k: v for k, v in data.items() if k != DEMO_COMPANY_ID}
    with open(COMPANIES_META, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)

def get_company_dirs(cid):
    u = os.path.join(UPLOADS_DIR, cid)
    v = os.path.join(VECTOR_DIR, cid)
    os.makedirs(u, exist_ok=True)
    os.makedirs(v, exist_ok=True)
    return u, v

def create_company(name):
    companies = load_companies()
    cid = str(uuid.uuid4())
    companies[cid] = {"name": name}
    save_companies(companies)
    get_company_dirs(cid)

def delete_company(cid):
    if cid == DEMO_COMPANY_ID:
        return
    companies = load_companies()
    companies.pop(cid, None)
    save_companies(companies)
    shutil.rmtree(os.path.join(UPLOADS_DIR, cid), ignore_errors=True)
    shutil.rmtree(os.path.join(VECTOR_DIR, cid), ignore_errors=True)

# ======================================================
# PDF → CHUNKS
# ======================================================

def extract_pdf_chunks(path, chunk_size=600, overlap=100):
    reader = PdfReader(path)
    chunks = []
    cid = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if len(chunk) > 80:
                chunks.append({
                    "id": cid,
                    "text": chunk,
                    "page": page_num,
                    "source": os.path.basename(path)
                })
                cid += 1
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

def retrieve(vs_dir, query):
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
# DEMO AUTO-INDEX (CRITICAL FIX)
# ======================================================

def ensure_demo_index():
    uploads, vs = get_company_dirs(DEMO_COMPANY_ID)

    if os.path.exists(os.path.join(vs, "index.faiss")):
        return

    demo_copy = os.path.join(uploads, "demo_policy.pdf")
    if not os.path.exists(demo_copy):
        shutil.copy(DEMO_PDF, demo_copy)

    chunks = extract_pdf_chunks(demo_copy)
    build_index(chunks, vs)

# ======================================================
# ANSWERING
# ======================================================

def generate_answer(question, chunk):
    if not chunk:
        return "I could not find this information in the policy documents."

    prompt = f"""
Answer using ONLY the text below.

{chunk['text']}

Question: {question}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=120)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ======================================================
# UI — EMPLOYEE
# ======================================================

def employee_ui():
    st.title("🧑‍💼 AI HR Assistant")

    companies = load_companies()
    name_to_id = {v["name"]: k for k, v in companies.items()}
    company = st.selectbox("🏢 Company", list(name_to_id.keys()))
    cid = name_to_id[company]

    uploads, vs = get_company_dirs(cid)

    if cid == DEMO_COMPANY_ID:
        ensure_demo_index()
        st.info("📚 Demo policy indexed and ready")

    chat_key = f"chat_{cid}"
    st.session_state.setdefault(chat_key, [])

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.text_input("Ask a policy question")
    if st.button("Send") and q:
        st.session_state[chat_key].append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            chunk = retrieve(vs, q)
            answer = generate_answer(q, chunk)
            st.markdown(answer)
        st.session_state[chat_key].append({"role": "assistant", "content": answer})

# ======================================================
# UI — ADMIN (UPLOAD FEATURE)
# ======================================================

def admin_ui():
    st.subheader("🛠 Admin Panel")

    companies = load_companies()
    real = {k: v for k, v in companies.items() if k != DEMO_COMPANY_ID}

    if real:
        name = st.selectbox("Company", [v["name"] for v in real.values()])
        cid = next(k for k, v in real.items() if v["name"] == name)
        uploads, vs = get_company_dirs(cid)

        files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if files:
            for f in files:
                with open(os.path.join(uploads, f.name), "wb") as out:
                    out.write(f.getbuffer())
            st.success("Files uploaded")

        if st.button("Build Index"):
            chunks = []
            for f in os.listdir(uploads):
                if f.endswith(".pdf"):
                    chunks.extend(extract_pdf_chunks(os.path.join(uploads, f)))
            if chunks:
                build_index(chunks, vs)
                st.success("Index built")

    st.divider()
    name = st.text_input("New company name")
    if st.button("Create") and name:
        create_company(name)
        st.rerun()

# ======================================================
# MAIN
# ======================================================

def main():
    st.set_page_config("AI HR Assistant", "🧑‍💼", layout="wide")
    t1, t2 = st.tabs(["Employee", "Admin"])
    with t1:
        employee_ui()
    with t2:
        admin_ui()

if __name__ == "__main__":
    main()
