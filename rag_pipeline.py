import os
from dotenv import load_dotenv
from groq import Groq

import pickle
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

with open("data/bm25/bm25.pkl", "rb") as f:
    bm25, bm25_docs, bm25_metadata = pickle.load(f)

VECTORSTORE_DIR = "data/vectorstore/10k_faiss_index"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

vectorstore = FAISS.load_local(
    VECTORSTORE_DIR,
    embedding_model,
    allow_dangerous_deserialization=True
)

def retrieve_rerank(query, k=5):
    # ----- Vector retrieval -----
    vector_docs = vectorstore.similarity_search(query, k=8)

    # ----- BM25 retrieval -----
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:8]

    bm25_docs_selected = [
        Document(
            page_content=bm25_docs[i],
            metadata=bm25_metadata[i]
        )
        for i in top_bm25_indices
    ]

    # ----- Merge & deduplicate -----
    combined = {doc.page_content: doc for doc in vector_docs + bm25_docs_selected}
    combined_docs = list(combined.values())

    # ----- Re-rank -----
    pairs = [(query, doc.page_content) for doc in combined_docs]
    scores = reranker.predict(pairs, batch_size=16)

    reranked = sorted(
        zip(combined_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return reranked[:k]


def build_prompt(query, top_chunks):
    context_blocks = []

    for i, (doc, _) in enumerate(top_chunks, 1):
        document = doc.metadata.get("document", "Unknown Document")
        section = doc.metadata.get("section", "UNKNOWN")
        page = doc.metadata.get("page", "N/A")

        citation = f'["{document}", "{section}", "p. {page}"]'

        block = f"""
SOURCE {i}:
Document: {document}
Section: {section}
Page: p. {page}
Citation Format: {citation}

Content:
{doc.page_content[:800]}
"""
        context_blocks.append(block)

    context_text = "\n".join(context_blocks)

    prompt = f"""
You are a financial and legal analysis assistant.

You must strictly follow these rules:

1. Answer ONLY using the information provided in the sources below.
2. Do not round up numerical values
3. Each factual statement MUST be followed by its citation in this exact format:
   ["Apple 10-K", "Item X", "p. Y"]
4. If the answer is NOT found in the sources, reply exactly:
   "Not specified in the document."
5. If the question is outside the scope of Apple 2024 10-K or Tesla 2023 10-K, reply exactly:
   "This question cannot be answered based on the provided documents."

Question:
{query}

Sources:
{context_text}

Provide your final answer below:
"""

    return prompt



def answer_question(query: str) -> dict:
    """
    Answers a question using the RAG pipeline.

    Args:
        query (str): The user question about Apple or Tesla 10-K filings.

    Returns:
        dict: {
            "answer": str,
            "sources": list
        }
    """
    top_chunks = retrieve_rerank(query)
    prompt = build_prompt(query, top_chunks)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    answer = response.choices[0].message.content.strip()

    sources = []
    for doc, _ in top_chunks:
        sources.append([
            doc.metadata.get("document", "Unknown"),
            doc.metadata.get("section", "UNKNOWN"),
            f"p. {doc.metadata.get('page', 'N/A')}"
        ])

    # Guardrail: refuse out-of-scope
    if answer == "This question cannot be answered based on the provided documents.":
        sources = []

    return {
        "answer": answer,
        "sources": sources
    }



def ask(query):
    top_chunks = retrieve_rerank(query)
    prompt = build_prompt(query, top_chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        # model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    answer = response.choices[0].message.content

    # Adding a post-generation check for absolute enforcement(extra-strong guardrail)
    if "Not specified in the document." in answer:
        answer = "Not specified in the document."


    print("\nAnswer:\n", answer)

if __name__ == "__main__":
    q = input("Ask your question: ")
    ask(q)
