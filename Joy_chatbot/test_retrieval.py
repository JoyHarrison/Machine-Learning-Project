import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_FILE = "chunks.pkl"
INDEX_FILE = "vector.index"

# Load chunks
with open(CHUNK_FILE, "rb") as f:
    chunks = pickle.load(f)

index = faiss.read_index(INDEX_FILE)
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

while True:
    query = input("\nAsk a question (type exit to quit): ")
    if query.lower() == "exit":
        break

    q_embedding = embed_model.encode([query])
    D, I = index.search(np.array(q_embedding), 6)

    print("\nTop retrieved chunks:\n")

    for i in I[0]:
        print("-----")
        print(chunks[i][:400])
        print()
