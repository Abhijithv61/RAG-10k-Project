# 📄 Advanced Retrieval-Augmented Generation (RAG) System for Financial & Legal Analysis  
### Apple 2024 10-K & Tesla 2023 10-K

## 📌 Project Overview

This project implements an **end-to-end Retrieval-Augmented Generation (RAG) system** that answers complex **financial and legal questions** by grounding responses in **Apple’s 2024 Form 10-K** and **Tesla’s 2023 Form 10-K**.

The system retrieves relevant document sections using **hybrid retrieval (vector + BM25)**, re-ranks results for precision, and generates **accurate, well-cited answers** using a **low-latency open-access LLM**.  
Special emphasis is placed on **faithfulness, citation correctness, and hallucination avoidance**, which are critical in financial and legal domains.

---

## 🎯 Objectives

- Build a RAG pipeline that answers questions **only using provided documents**
- Support **numeric, factual, and explanatory queries**
- Enforce **strict citations** with document, section, and page metadata
- Handle **unanswerable and out-of-scope questions safely**
- Provide a **reproducible evaluation framework** with meaningful metrics

---

## 🏗️ System Architecture

- **User Query**
  - **Hybrid Retrieval**
    - FAISS Vector Search (semantic recall)
    - BM25 Keyword Search (lexical precision)
  - Merge & De-duplicate
  - Cross-Encoder Re-Ranking
  - Context-Aware Prompt Construction
  - LLM Generation (Groq – LLaMA 3.1)
    - Final Answer with Citations



---

## 🧩 Key Components

### 1️⃣ Document Parsing & Chunking
- PDFs parsed into text with **page markers**
- Chunked using `RecursiveCharacterTextSplitter`
- Metadata preserved per chunk:
  - `document` (Apple 10-K / Tesla 10-K)
  - `section` (e.g., Item 8)
  - `page`
  - `chunk_id`

### 2️⃣ Embeddings & Vector Store
- Embeddings: `all-MiniLM-L6-v2`
- Vector index: **FAISS (local)**

### 3️⃣ Hybrid Retrieval
- **FAISS** → semantic similarity
- **BM25** → exact keyword matching
- Results merged and de-duplicated

### 4️⃣ Re-Ranking
- Cross-encoder: `ms-marco-MiniLM-L-6-v2`
- Improves precision for financial/legal phrasing

### 5️⃣ Generation
- LLM: **Groq – LLaMA-3.1-8B-Instant**
- Low temperature (`0.0`) for deterministic outputs
- Strict prompt rules to:
  - Use only retrieved context
  - Cite every factual claim
  - Refuse out-of-scope questions

---

## 🛡️ Guardrails & Safety

| Scenario | Response |
|--------|---------|
| Answer not present in documents | `"Not specified in the document."` |
| Question outside provided documents | `"This question cannot be answered based on the provided documents."` |
| No citation available | Answer rejected |
| Hallucination risk | Strict prompting + evaluation |

---

## 📊 Evaluation Framework

Evaluation uses a **hybrid strategy**:

### 🔹 Rule-Based Metrics
- **Recall@5** → retrieval quality
- **Citation Accuracy** → document + section + page alignment
- **Guardrail Accuracy** → correct handling of unanswerable queries

### 🔹 LLM-as-Judge
- Used for **answer correctness**
- Handles numeric answers, dates, and semantic equivalence
- Reduces brittleness of string-based evaluation

---

## 📈 Example Metrics

| Metric | Description |
|------|------------|
| Recall@5 | Correct document section retrieved |
| Answer Accuracy | LLM-judged factual correctness |
| Citation Accuracy | Correct document & section cited |
| Guardrail Accuracy | Safe refusal on out-of-scope questions |


---
## 📎 Runnable Notebook (Google Colab)

A fully reproducible Google Colab notebook is provided to demonstrate the complete end-to-end RAG pipeline.  
The notebook performs the following steps:

- Clones this GitHub repository
- Uploads and parses Apple 2024 and Tesla 2023 10-K PDFs
- Builds hybrid retrieval indices (FAISS + BM25)
- Runs RAG-based question answering with citations
- Optionally evaluates predictions against ground truth

🔗 **Colab Notebook:**  
https://colab.research.google.com/drive/1kmvJPrgQnOSPBJcpfuPTwyODElQXasY-?usp=sharing

> **Note:** Users must provide their own Groq API key to run inference.
---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

git clone https://github.com/Abhijithv61/RAG-10k-Project

cd RAG_10k_Project

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Set Environment Variable

GROQ_API_KEY=your_api_key_here

## ▶️ How to Run

### Parse PDFs
python parse_pdfs.py

### Chunk Documents
python chunk_documents.py

### Build Indices
python build_vectorstore.py
python bm25_index.py

### Run RAG Pipeline
python rag_pipeline.py

### Generate Predictions
python generate_predictions.py

### Evaluate
python evaluate_rag.py

