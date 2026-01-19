import json
import re
from difflib import SequenceMatcher

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def llm_judge_answer(question, pred_answer, gt_answer, answerable):
    prompt = f"""
You are evaluating a Retrieval-Augmented Generation (RAG) system.

Question:
{question}

Model Answer:
{pred_answer}

Ground Truth Answer:
{gt_answer}

Answerable: {answerable}

Evaluation Rules:
- If Answerable is false, the model answer must be EXACTLY:
  "This question cannot be answered based on the provided documents."
- If Answerable is true, the model answer is CORRECT if it conveys the same factual
  information as the ground truth, even if phrasing differs.
- Additional explanation is allowed if it does NOT contradict the ground truth.
- Any factual contradiction makes the answer INCORRECT.

Respond with ONLY one word:
CORRECT or INCORRECT
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    verdict = response.choices[0].message.content.strip().upper()
    return verdict == "CORRECT"


# -----------------------------
# Utility helpers
# -----------------------------

def normalize(text):
    if text is None:
        return ""
    return text.lower().strip()

def strip_citations(text):
    # Remove anything after first citation bracket
    return text.split("[")[0].strip()

def fuzzy_match(a, b, threshold=0.85):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio() >= threshold

def extract_number(text):
    if text is None:
        return None

    # Percentage case
    if "%" in text:
        m = re.search(r"(\d+(\.\d+)?)\s*%", text)
        if m:
            return float(m.group(1))

    # Integer / monetary case
    matches = re.findall(r"\d[\d,]*", text)
    if not matches:
        return None

    return int(matches[0].replace(",", ""))


def numeric_match(pred, gt, tolerance=0):
    p = extract_number(pred)
    g = extract_number(gt)
    if p is None or g is None:
        return False
    return abs(p - g) <= tolerance

def section_match(gt_section, pred_section):
    if not gt_section or not pred_section:
        return False
    # Match ITEM number only (e.g., ITEM 8)
    return gt_section.split()[0] in pred_section

def recall_match(gt, chunk):
    return (
        gt["document"] == chunk["document"] and
        section_match(gt["section"], chunk["section"])
    )

def citation_match(gt, citation):
    doc_ok = gt["document"] in citation[0]
    section_ok = section_match(gt["section"], citation[1])
    page_ok = (
        gt.get("page") is None or
        str(gt["page"]) in citation[2]
    )
    return doc_ok and section_ok and page_ok

def normalize_guardrail(text):
    return normalize(text).replace(".", "")

# -----------------------------
# Load data
# -----------------------------

with open("predictions.json", "r") as f:
    predictions = {p["question_id"]: p for p in json.load(f)}

with open("ground_truth.json", "r") as f:
    ground_truth = {g["question_id"]: g for g in json.load(f)}

# -----------------------------
# Evaluation
# -----------------------------

results = []

for qid, gt in ground_truth.items():
    pred = predictions.get(qid)
    row = {"question_id": qid}

    if pred is None:
        row.update({
            "recall_at_5": 0,
            "answer_correct": 0,
            "citation_correct": 0,
            "guardrail_pass": 0
        })
        results.append(row)
        continue

    answer = pred.get("answer", "")
    retrieved_chunks = pred.get("retrieved_chunks", [])
    citations = pred.get("sources", [])

    # -----------------------------
    # Guardrail evaluation
    # -----------------------------
    if not gt["answerable"]:
        expected = normalize_guardrail(
            "This question cannot be answered based on the provided documents."
        )
        actual = normalize_guardrail(answer)

        row["guardrail_pass"] = int(actual == expected)
        row["recall_at_5"] = None
        row["answer_correct"] = None
        row["citation_correct"] = None
        results.append(row)
        continue

    row["guardrail_pass"] = None

    # -----------------------------
    # Recall@5 (retrieval quality)
    # -----------------------------
    recall = any(recall_match(gt, c) for c in retrieved_chunks)
    row["recall_at_5"] = int(recall)

    # -----------------------------
    # Answer correctness (LLM-as-Judge)
    # -----------------------------
    row["answer_correct"] = int(
        llm_judge_answer(
            question=pred.get("question", ""),
            pred_answer=answer,
            gt_answer=gt["answer"],
            answerable=gt["answerable"]
        )
    )


    # -----------------------------
    # Citation correctness
    # -----------------------------
    citation_ok = any(citation_match(gt, c) for c in citations)
    row["citation_correct"] = int(citation_ok)

    results.append(row)

# -----------------------------
# Aggregate metrics
# -----------------------------

answerable_rows = [r for r in results if r["recall_at_5"] is not None]
guardrail_rows = [r for r in results if r["guardrail_pass"] is not None]

metrics = {
    "Recall@5": round(
        sum(r["recall_at_5"] for r in answerable_rows) / len(answerable_rows), 3
    ),
    "Answer Accuracy": round(
        sum(r["answer_correct"] for r in answerable_rows) / len(answerable_rows), 3
    ),
    "Citation Accuracy": round(
        sum(r["citation_correct"] for r in answerable_rows) / len(answerable_rows), 3
    ),
    "Guardrail Accuracy": round(
        sum(r["guardrail_pass"] for r in guardrail_rows) / len(guardrail_rows), 3
    )
}

# -----------------------------
# Output
# -----------------------------

print("\nPer-question results:")
for r in results:
    print(r)

print("\nAggregate Metrics")
for k, v in metrics.items():
    print(f"{k}: {v}")
