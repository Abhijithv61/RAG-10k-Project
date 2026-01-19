import pickle
import os
from rank_bm25 import BM25Okapi

CHUNKS_DIR = "data/chunks"
BM25_DIR = "data/bm25"
os.makedirs(BM25_DIR, exist_ok=True)

documents = []
metadatas = []

for file in os.listdir(CHUNKS_DIR):
    if file.endswith("_chunks.pkl"):
        with open(os.path.join(CHUNKS_DIR, file), "rb") as f:
            docs = pickle.load(f)
            for d in docs:
                documents.append(d.page_content)
                metadatas.append(d.metadata)

tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

with open(os.path.join(BM25_DIR, "bm25.pkl"), "wb") as f:
    pickle.dump((bm25, documents, metadatas), f)

print("BM25 index built successfully.")
