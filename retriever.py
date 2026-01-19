from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

VECTORSTORE_DIR = "data/vectorstore/10k_faiss_index"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

vectorstore = FAISS.load_local(
    VECTORSTORE_DIR,
    embedding_model,
    allow_dangerous_deserialization=True
)

def retrieve_top_k(query: str, k: int = 5):
    initial_docs = vectorstore.similarity_search(query, k=15)

    pairs = [(query, doc.page_content) for doc in initial_docs]
    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(initial_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in reranked[:k]]

if __name__ == "__main__":
    query = "How does Tesla recognize revenue from regulatory credits?"
    results = retrieve_top_k(query)

    for i, doc in enumerate(results, 1):
        document = doc.metadata.get("document", "Unknown Document")
        section = doc.metadata.get("section", "UNKNOWN")
        page = doc.metadata.get("page", "N/A")

        print(f"\n--- Result {i} ---")
        print(f"Source: {document}")
        print(f"Section: {section}")
        print(f"Page: p. {page}")
        print(doc.page_content[:500])
