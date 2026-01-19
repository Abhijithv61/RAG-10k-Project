from pypdf import PdfReader
from tqdm import tqdm
import re
import os

RAW_PDF_DIR = "data/raw_pdfs"
PARSED_TEXT_DIR = "data/parsed_text"
os.makedirs(PARSED_TEXT_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    # Remove page numbers and repeated headers/footers
    text = re.sub(r"\n\d+\s*\n", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"-\n", "", text)  # remove hyphenated line breaks
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def parse_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    full_text = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            full_text.append(f"\n\n<<<PAGE {page_num}>>>\n\n")
            full_text.append(text)

    return clean_text("\n".join(full_text))

def main():
    for file in os.listdir(RAW_PDF_DIR):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(RAW_PDF_DIR, file)
            cleaned_text = parse_pdf(pdf_path)

            output_file = file.replace(".pdf", ".txt")
            output_path = os.path.join(PARSED_TEXT_DIR, output_file)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"Saved parsed text → {output_path}")

if __name__ == "__main__":
    main()
