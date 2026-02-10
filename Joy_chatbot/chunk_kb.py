import re
import pickle

INPUT_FILE = "knowledge_base.txt"
OUTPUT_FILE = "chunks.pkl"

print("Loading knowledge base...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    kb_text = f.read()

chunks = []

# Split by main sections
sections = re.split(r"(?=== [A-Z].+ ===)", kb_text)

for section in sections:
    section = section.strip()
    if not section:
        continue

    # EXPERIENCE → split by Internship
    if "Internship:" in section:
        sub = re.split(r"(?=Internship:)", section)
        chunks.extend([s.strip() for s in sub if s.strip()])

    # PROJECTS → split by Project
    elif "Project:" in section:
        sub = re.split(r"(?=Project:)", section)
        chunks.extend([s.strip() for s in sub if s.strip()])

    # PROFILE / SKILLS → keep whole
    else:
        chunks.append(section)

print(f"Total chunks created: {len(chunks)}")

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(chunks, f)

print("Chunks saved to chunks.pkl")

print("\nPreview:\n")
for i, c in enumerate(chunks[:6]):
    print(f"--- Chunk {i+1} ---")
    print(c[:300])
    print()
