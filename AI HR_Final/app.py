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
LOCAL_LLM_NAME = "google/flan-t5-base"

DEMO_COMPANY_ID = "demo-company"
DEMO_COMPANY_NAME = "Demo Company (Portfolio)"

# Demo files in root
DEMO_PDF = "demo_policy.pdf"
DEMO_QUESTIONS = "demo_questions.json"

os.makedirs(BASE_DIR, exist_ok=True)

# ======================================================
# MODELS
# ======================================================

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_NAME)
    return tokenizer, model

embedder = load_embedder()
local_tokenizer, local_model = load_local_llm()

# ======================================================
# LOCAL LLM
# ======================================================

def run_llm(prompt: str, max_new_tokens=120) -> str:
    inputs = local_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    outputs = local_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.0
    )
    return local_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ======================================================
# PROMPT ENGINEERING
# ======================================================

def improve_question(question: str) -> str:
    prompt = f"""
You are a professional HR policy language assistant.

Rewrite the employee question into formal HR language.
Do NOT answer it.
Return only the rewritten question.

Question: "{question}"

Rewritten question:
"""
    improved = run_llm(prompt, 60)
    return improved.strip().strip('"')

# ======================================================
# ANSWER GENERATION
# ======================================================

def generate_answer(question, chunk):
    if not chunk:
        return "I do not have that information in the company documents."

    prompt = f"""
Answer the question using ONLY the HR policy text below.

Text:
{chunk['text']}

Question:
{question}

Answer:
"""
    return run_llm(prompt)

# ======================================================
# COMPANY STORAGE
# ======================================================

def load_companies():
    companies = {}
    if os.path.exists(COMPANIES_META):
        with open(COMPANIES_META, "r") as f:
            companies = json.load(f)
    companies.setdefault(DEMO_COMPANY_ID, {"name": DEMO_COMPANY_NAME})
    return companies

def save_companies(data):
    data = {k: v for k, v in data.items() if k != DEMO_COMPANY_ID}
    with open(COMPANIES_META, "w") as f:
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
# PDF + FAISS
# ======================================================

def extract_pdf_chunks(path, chunk_size=600, overlap=100):
    reader = PdfReader(path)
    chunks = []
    cid = 0
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        i = 0
        while i < len(text):
            chunk = text[i:i+chunk_size].strip()
            if len(chunk) > 80:
                chunks.append({"id": cid, "text": chunk, "page": page_num})
                cid += 1
            i += chunk_size - overlap
    return chunks

def build_index(chunks, vs_dir):
    emb = embedder.encode([c["text"] for c in chunks]).astype("float32")
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, os.path.join(vs_dir, "index.faiss"))
    json.dump(chunks, open(os.path.join(vs_dir, "chunks.json"), "w"), indent=2)

def retrieve_top_chunk(vs_dir, query):
    ip = os.path.join(vs_dir, "index.faiss")
    if not os.path.exists(ip):
        return None
    index = faiss.read_index(ip)
    chunks = json.load(open(os.path.join(vs_dir, "chunks.json")))
    qe = embedder.encode([query]).astype("float32")
    _, I = index.search(qe, 1)
    return chunks[I[0][0]]

# ======================================================
# AUTO DEMO INDEX
# ======================================================

def ensure_demo_index():
    vs_dir = os.path.join(BASE_DIR, "vectorstores", DEMO_COMPANY_ID)
    os.makedirs(vs_dir, exist_ok=True)

    index_path = os.path.join(vs_dir, "index.faiss")

    if os.path.exists(index_path):
        return

    if not os.path.exists(DEMO_PDF):
        return

    chunks = extract_pdf_chunks(DEMO_PDF)
    build_index(chunks, vs_dir)

# ======================================================
# UI: EMPLOYEE
# ======================================================

def employee_ui():
    st.title("🧑‍💼 AI HR Assistant (Offline Mode)")

    companies = load_companies()
    name_to_id = {v["name"]: k for k, v in companies.items()}
    company = st.selectbox("🏢 Select company", list(name_to_id.keys()))
    cid = name_to_id[company]
    uploads, vs_dir = get_company_dirs(cid)

    # Demo questions
    if cid == DEMO_COMPANY_ID and os.path.exists(DEMO_QUESTIONS):
        st.markdown("### 💡 Example questions")
        for q in json.load(open(DEMO_QUESTIONS)):
            if st.button(q):
                st.session_state["auto_question"] = q

    with st.expander("📄 Upload HR Policy PDFs"):
        files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("Process & Index") and files:
            for f in files:
                path = os.path.join(uploads, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())

                chunks = extract_pdf_chunks(path)
                build_index(chunks, vs_dir)

            st.success("Documents indexed ✅")

    st.divider()

    st.markdown("### ✨ Question assistant")
    use_pe = st.toggle("Improve my question automatically")

    chat_key = f"chat_{cid}"
    st.session_state.setdefault(chat_key, [])

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.session_state.pop("auto_question", None) or st.chat_input("Ask a policy question")
    if not question:
        return

    final_question = question
    st.session_state[chat_key].append({"role": "user", "content": question})

    with st.chat_message("assistant"):

        if use_pe:
            improved = improve_question(question)
            st.markdown("#### ✍️ Improved question")
            st.info(improved)
            final_question = improved

        chunk = retrieve_top_chunk(vs_dir, final_question)
        answer = generate_answer(final_question, chunk)

        st.markdown("#### 📄 Answer")
        st.markdown(answer)

        if chunk:
            st.caption(f"📌 Source: Page {chunk['page']} · Chunk {chunk['id']}")

    st.session_state[chat_key].append({"role": "assistant", "content": answer})

# ======================================================
# UI: ADMIN
# ======================================================

def admin_ui():
    st.subheader("🛠 Admin Panel")

    for cid, info in load_companies().items():
        if cid == DEMO_COMPANY_ID:
            continue
        if st.button(f"Delete {info['name']}"):
            delete_company(cid)
            st.rerun()

    name = st.text_input("Create new company")
    if st.button("Create") and name.strip():
        create_company(name.strip())
        st.success("Company created")
        st.rerun()

# ======================================================
# MAIN
# ======================================================

def main():
    st.set_page_config("AI HR Assistant", "🧑‍💼", layout="wide")

    # ✅ Pre-index demo before UI loads
    ensure_demo_index()

    tab1, tab2 = st.tabs(["Employee", "Admin"])
    with tab1:
        employee_ui()
    with tab2:
        admin_ui()

if __name__ == "__main__":
    main()
