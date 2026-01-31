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
# CONFIG
# ======================================================

BASE_DIR = "data"
COMPANIES_META = os.path.join(BASE_DIR, "companies.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

DEMO_COMPANY_ID = "demo-company"
DEMO_COMPANY_NAME = "Demo Company (Portfolio)"
DEMO_DIR = "demo"
DEMO_PDF = os.path.join(DEMO_DIR, "demo_policy.pdf")
DEMO_QUESTIONS = os.path.join(DEMO_DIR, "demo_questions.json")

os.makedirs(BASE_DIR, exist_ok=True)

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
    companies.pop(cid, None)
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
    cid = 0
    filename = os.path.basename(path)

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
                    "source": filename
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

def retrieve_top_chunk(vs_dir, query):
    index_path = os.path.join(vs_dir, "index.faiss")
    chunks_path = os.path.join(vs_dir, "chunks.json")

    if not os.path.exists(index_path):
        return None

    index = faiss.read_index(index_path)
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    q_emb = embedder.encode([query]).astype("float32")
    _, I = index.search(q_emb, 1)

    return chunks[I[0][0]]

def index_company_pdfs(cid):
    uploads_dir, vs_dir = get_company_dirs(cid)
    all_chunks = []

    for file in os.listdir(uploads_dir):
        if file.lower().endswith(".pdf"):
            all_chunks.extend(
                extract_pdf_chunks(os.path.join(uploads_dir, file))
            )

    if not all_chunks:
        return False

    build_index(all_chunks, vs_dir)
    return True

# ======================================================
# DEMO INDEX (CRITICAL FIX)
# ======================================================

def ensure_demo_index():
    _, vs_dir = get_company_dirs(DEMO_COMPANY_ID)

    index_path = os.path.join(vs_dir, "index.faiss")
    if os.path.exists(index_path):
        return

    if not os.path.exists(DEMO_PDF):
        st.error("❌ demo_policy.pdf not found in deployment")
        return

    chunks = extract_pdf_chunks(DEMO_PDF)
    if not chunks:
        st.error("❌ Demo PDF loaded but no text extracted")
        return

    build_index(chunks, vs_dir)

# ======================================================
# PROMPT ENGINEERING
# ======================================================

def improve_prompt(question: str) -> str:
    prompt = f"""
Rewrite the question to be clear, specific, and suitable for HR policy Q&A.

Original:
{question}

Improved:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=80, temperature=0.3)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip() or question

# ======================================================
# ANSWER GENERATION
# ======================================================

def generate_answer(question, chunk):
    if not chunk:
        return "I do not have that information in the company documents."

    prompt = f"""
Answer using ONLY the policy text below.

Policy:
{chunk['text']}

Question:
{question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=120)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ======================================================
# DEMO QUESTIONS
# ======================================================

def load_demo_questions():
    if not os.path.exists(DEMO_QUESTIONS):
        return []
    with open(DEMO_QUESTIONS, encoding="utf-8") as f:
        return json.load(f)

# ======================================================
# UI: EMPLOYEE
# ======================================================

def employee_ui():
    st.title("🧑‍💼 AI HR Assistant")

    companies = load_companies()
    name_to_id = {v["name"]: k for k, v in companies.items()}
    company = st.selectbox("🏢 Select company", list(name_to_id.keys()))
    cid = name_to_id[company]

    uploads_dir, vs_dir = get_company_dirs(cid)

    if cid == DEMO_COMPANY_ID:
        ensure_demo_index()

    indexed = os.path.exists(os.path.join(vs_dir, "index.faiss"))

    if indexed:
        st.info("📚 Policy documents indexed and ready")
    else:
        st.warning("⚠️ No indexed documents available")

    st.divider()

    if cid == DEMO_COMPANY_ID:
        demo_qs = load_demo_questions()
        if demo_qs:
            st.markdown("### 💡 Try a demo question")
            question = st.selectbox("Demo questions", demo_qs)
        else:
            question = ""
    else:
        question = ""

    user_input = st.text_input("Ask a policy question", value=question)

    use_pe = st.checkbox("🛠 Improve my question")
    send = st.button("Send")

    if not send or not user_input.strip():
        return

    final_q = improve_prompt(user_input) if use_pe else user_input

    with st.chat_message("assistant"):
        chunk = retrieve_top_chunk(vs_dir, final_q)
        answer = generate_answer(final_q, chunk)
        st.markdown(answer)

        if chunk:
            st.caption(f"📄 {chunk['source']} — Page {chunk['page']}")

# ======================================================
# UI: ADMIN
# ======================================================

def admin_ui():
    st.subheader("🛠 Admin Panel")

    companies = load_companies()
    company_names = [v["name"] for k, v in companies.items() if k != DEMO_COMPANY_ID]

    if company_names:
        selected = st.selectbox("🏢 Select company", company_names)
        cid = next(k for k, v in companies.items() if v["name"] == selected)

        uploads_dir, _ = get_company_dirs(cid)

        files = st.file_uploader("Upload policy PDFs", type=["pdf"], accept_multiple_files=True)

        if files:
            for file in files:
                with open(os.path.join(uploads_dir, file.name), "wb") as f:
                    f.write(file.getbuffer())
            st.success("Files uploaded")

        if st.button("📦 Build / Rebuild Index"):
            with st.spinner("Indexing..."):
                if index_company_pdfs(cid):
                    st.success("Index built successfully")
                else:
                    st.warning("No PDFs found")

        if st.button(f"❌ Delete {selected}"):
            delete_company(cid)
            st.rerun()

    st.divider()

    name = st.text_input("New company name")
    if st.button("Create company") and name:
        create_company(name)
        st.success("Company created")
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
