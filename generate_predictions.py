import json
from rag_pipeline import retrieve_rerank, build_prompt
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Load evaluation questions
# -----------------------------
with open("evaluation_questions.json", "r") as f:
    questions = json.load(f)

predictions = []

# -----------------------------
# Run RAG for each question
# -----------------------------
for q in questions:
    qid = q["question_id"]
    query = q["question"]

    top_chunks = retrieve_rerank(query)
    prompt = build_prompt(query, top_chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        # model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    answer = response.choices[0].message.content.strip()

    # -------- Sources (for citation validation) --------
    sources = []
    retrieved_chunks = []

    for doc, _ in top_chunks:
        document = doc.metadata.get("document")
        section = doc.metadata.get("section")
        page = doc.metadata.get("page")

        sources.append([document, section, f"p. {page}"])
        retrieved_chunks.append(
            {
                "document": document,
                "section": section,
                "page": page
            }
        )

    predictions.append(
        {
            "question_id": qid,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks
        }
    )

# -----------------------------
# Save predictions.json
# -----------------------------
with open("predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

print("predictions.json generated successfully.")
