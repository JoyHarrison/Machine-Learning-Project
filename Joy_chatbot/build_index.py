import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_FILE = "chunks.pkl"
INDEX_FILE = "vector.index"

print("Loading chunks...")

with open(CHUNK_FILE, "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks")

# Better embedding model
print("Loading embedding model...")
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

print("Generating embeddings...")
embeddings = embed_model.encode(chunks, show_progress_bar=True)

dimension = embeddings.shape[1]
print("Embedding dimension:", dimension)

print("Building FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, INDEX_FILE)

print("✅ Index built successfully!")
