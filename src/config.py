import os

from dotenv import load_dotenv

load_dotenv()

RAW_DOC_PATH = os.getenv("RAW_DOC_PATH", "data/raw/knowledge_base.docx")
PROCESSED_TEXT_PATH = os.getenv("PROCESSED_TEXT_PATH", "data/processed/knowledge_base.txt")
