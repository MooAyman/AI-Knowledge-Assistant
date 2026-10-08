import os
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_TEXT_PATH, RAW_DOC_PATH

def extract_docx(input_path, output_path):
    print(f"Extracting text from {input_path}...")
    doc = Document(input_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write paragraph texts separated by newline
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(full_text))
        
    print(f"Saved extracted text to {output_path}")

if __name__ == "__main__":
    extract_docx(RAW_DOC_PATH, PROCESSED_TEXT_PATH)
