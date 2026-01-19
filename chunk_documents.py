import os
import pickle
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

PARSED_TEXT_DIR = "data/parsed_text"
CHUNKS_DIR = "data/chunks"
os.makedirs(CHUNKS_DIR, exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

SECTION_PATTERN = re.compile(r"(ITEM\s+\d+[A-Z]?\.*\s+[A-Z ,’\-]+)", re.IGNORECASE)
PAGE_PATTERN = re.compile(r"<<<PAGE (\d+)>>>")

def detect_sections(text):
    return [(m.start(), m.group().strip()) for m in SECTION_PATTERN.finditer(text)]

def detect_pages(text):
    return [(m.start(), int(m.group(1))) for m in PAGE_PATTERN.finditer(text)]

def assign_value(chunk_start, positions, default):
    current = default
    for pos, value in positions:
        if chunk_start >= pos:
            current = value
        else:
            break
    return current

def get_document_name(file_name):
    if "apple" in file_name.lower():
        return "Apple 10-K"
    if "tesla" in file_name.lower():
        return "Tesla 10-K"
    return "Unknown Document"

def chunk_file(file_name):
    with open(os.path.join(PARSED_TEXT_DIR, file_name), "r", encoding="utf-8") as f:
        text = f.read()

    section_positions = detect_sections(text)
    page_positions = detect_pages(text)

    chunks = text_splitter.split_text(text)
    documents = []

    pointer = 0
    for i, chunk in enumerate(chunks):
        start_idx = text.find(chunk, pointer)
        pointer = start_idx + len(chunk)

        section = assign_value(start_idx, section_positions, "UNKNOWN")
        page = assign_value(start_idx, page_positions, -1)

        documents.append(
            Document(
                page_content=chunk.replace(PAGE_PATTERN.pattern, ""),
                metadata={
                    "document": get_document_name(file_name),
                    "section": section.upper(),
                    "page": page,
                    "chunk_id": i
                }
            )
        )

    return documents

def main():
    for file in os.listdir(PARSED_TEXT_DIR):
        if file.endswith(".txt"):
            print(f"Chunking with sections & pages: {file}")
            docs = chunk_file(file)

            out_file = file.replace(".txt", "_chunks.pkl")
            with open(os.path.join(CHUNKS_DIR, out_file), "wb") as f:
                pickle.dump(docs, f)

            print(f"Saved {len(docs)} chunks → {out_file}")

if __name__ == "__main__":
    main()
