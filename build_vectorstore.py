import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHUNKS_DIR = "data/chunks"
VECTORSTORE_DIR = "data/vectorstore/10k_faiss_index"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

all_docs = []

for file in os.listdir(CHUNKS_DIR):
    if file.endswith("_chunks.pkl"):
        with open(os.path.join(CHUNKS_DIR, file), "rb") as f:
            docs = pickle.load(f)
            all_docs.extend(docs)

print(f"Total chunks loaded: {len(all_docs)}")

print("Building FAISS index...")
vectorstore = FAISS.from_documents(all_docs, embedding_model)

vectorstore.save_local(VECTORSTORE_DIR)

print("FAISS vector store saved successfully.")
