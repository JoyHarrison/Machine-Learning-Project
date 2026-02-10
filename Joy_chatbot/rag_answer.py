import pickle
import re

CHUNK_FILE = "chunks.pkl"

# =========================
# Load memory
# =========================
with open(CHUNK_FILE, "rb") as f:
    chunks = pickle.load(f)


# =========================
# Structured extraction
# =========================
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


def extract_internships():
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

    text = "\n📌 Projects:\n"
    for p in projects:
        text += f"\n🔹 {p['name']}\n"
        text += f"Summary: {p['summary']}\n"
        text += f"Impact: {p['impact']}\n"

    return text


def answer_experience():
    internships = extract_internships()

    text = "\n📌 Experience:\n"
    for i in internships:
        text += f"\n🔹 {i['company']} — {i['role']}\n"
        text += f"Impact: {i['impact']}\n"

    return text


def answer_email():
    email = extract_email()
    return f"\n📧 Contact: {email}" if email else "Email not found."


# =========================
# Interactive loop
# =========================
while True:
    q = input("\nAsk a question (exit to quit): ").lower()

    if q == "exit":
        break

    if "project" in q:
        print(answer_projects())

    elif "experience" in q or "intern" in q:
        print(answer_experience())

    elif "email" in q or "contact" in q:
        print(answer_email())

    else:
        print("\nI can answer about projects, experience, or contact info.")
