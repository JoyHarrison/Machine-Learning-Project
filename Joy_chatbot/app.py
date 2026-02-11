import streamlit as st
import pickle
import re
import os

# =========================
# Base directory (important for Streamlit Cloud)
# =========================
BASE_DIR = os.path.dirname(__file__)

CHUNK_FILE = os.path.join(BASE_DIR, "chunks.pkl")
PROFILE_IMG = os.path.join(BASE_DIR, "profile.png")
RESUME_ML = os.path.join(BASE_DIR, "resume_ml.pdf")
RESUME_DA = os.path.join(BASE_DIR, "resume_da.pdf")

# =========================
# Load memory
# =========================
@st.cache_resource
def load_chunks():
    if not os.path.exists(CHUNK_FILE):
        st.error(f"Missing file: {CHUNK_FILE}")
        return []

    with open(CHUNK_FILE, "rb") as f:
        return pickle.load(f)

chunks = load_chunks()

# =========================
# Extraction functions
# =========================
def extract_skills():
    skills = []
    for chunk in chunks:
        if "Programming:" in chunk or "Machine Learning:" in chunk:
            skills.append(chunk)
    return "\n".join(skills)


def extract_projects():
    projects = []

    for chunk in chunks:
        if "Project:" in chunk:
            name = re.search(r"Project:\s*(.+)", chunk)
            summary = re.search(r"Summary:\s*(.+)", chunk, re.S)
            impact = re.search(r"Impact:\s*(.+)", chunk, re.S)

            proj = {
                "name": name.group(1).strip() if name else "",
                "summary": summary.group(1).strip().split("\n")[0] if summary else "",
                "impact": impact.group(1).strip().split("\n")[0] if impact else ""
            }

            projects.append(proj)

    return projects


def extract_experience():
    internships = []

    for chunk in chunks:
        if "Internship:" in chunk:
            company = re.search(r"Company:\s*(.+)", chunk)
            role = re.search(r"Role:\s*(.+)", chunk)
            impact = re.search(r"Impact:\s*(.+)", chunk, re.S)

            item = {
                "company": company.group(1).strip() if company else "",
                "role": role.group(1).strip() if role else "",
                "impact": impact.group(1).strip().split("\n")[0] if impact else ""
            }

            internships.append(item)

    return internships


def extract_email():
    for chunk in chunks:
        match = re.search(r"Email:\s*(.+)", chunk)
        if match:
            return match.group(1).strip()
    return None


# =========================
# Answer engine
# =========================
def answer_projects():
    projects = extract_projects()
    text = "📌 **Projects:**\n"

    for p in projects:
        text += f"\n🔹 **{p['name']}**\n"
        text += f"Summary: {p['summary']}\n"
        text += f"Impact: {p['impact']}\n"

    return text


def answer_experience():
    internships = extract_experience()
    text = "📌 **Experience:**\n"

    for i in internships:
        text += f"\n🔹 **{i['company']} — {i['role']}**\n"
        text += f"Impact: {i['impact']}\n"

    return text


def answer_skills():
    return f"📌 **Skills:**\n{extract_skills()}"


def answer_email():
    email = extract_email()
    return f"📧 Contact: **{email}**" if email else "Email not found."


def route_question(q):
    q = q.lower()

    if "skill" in q:
        return answer_skills()

    if "project" in q:
        return answer_projects()

    if "experience" in q or "intern" in q:
        return answer_experience()

    if "email" in q or "contact" in q:
        return answer_email()

    return "Try asking about skills, projects, or experience."


# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="Joy Harrison AI Assistant", layout="wide")

# =========================
# Sidebar branding panel
# =========================
with st.sidebar:

    if os.path.exists(PROFILE_IMG):
        st.image(PROFILE_IMG, width=120)

    st.title("Joy Harrison")

    st.markdown("""
🟢 **Open to Opportunities**

Machine Learning Engineer  
Business Analytics & AI Graduate
""")

    st.divider()

    st.markdown("### 🚀 Core Strengths")
    st.markdown("""
✔ Machine Learning Engineering  
✔ NLP & RAG Systems  
✔ Deep Learning  
✔ Analytics Dashboards  
""")

    st.markdown("### 🧠 Skill Stack")
    st.markdown("""
`Python` `ML` `NLP` `TensorFlow`  
`Scikit-learn` `FAISS` `Streamlit`
""")

    st.divider()

    if os.path.exists(RESUME_ML):
        with open(RESUME_ML, "rb") as f:
            st.download_button(
                label="📄 ML Resume",
                data=f,
                file_name="Joy_Harrison_ML_Resume.pdf",
                mime="application/pdf"
            )

    if os.path.exists(RESUME_DA):
        with open(RESUME_DA, "rb") as f:
            st.download_button(
                label="📄 Data Analyst Resume",
                data=f,
                file_name="Joy_Harrison_DA_Resume.pdf",
                mime="application/pdf"
            )

    st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/joy-harrison/")
    st.link_button("💻 GitHub", "https://github.com/JoyHarrison")

# =========================
# Main header
# =========================
st.title("📚 Joy Harrison AI Portfolio Assistant")
st.caption("Explore skills, experience, and projects")

col1, col2 = st.columns(2)

with col1:
    st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/joy-harrison/")

with col2:
    st.link_button("💻 GitHub", "https://github.com/JoyHarrison")

st.divider()

st.subheader("💡 Suggested Questions")

suggested = [
    "What skills does Joy have?",
    "What internships has Joy completed?",
    "What projects has Joy built?",
    "How can I contact Joy?"
]

cols = st.columns(2)

for i, q in enumerate(suggested):
    if cols[i % 2].button(q):
        st.session_state["quick_question"] = q

st.divider()

# =========================
# Chat memory
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask something about Joy...")

if "quick_question" in st.session_state:
    user_input = st.session_state.pop("quick_question")

if user_input:
    answer = route_question(user_input)

    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("assistant", answer))

for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)
