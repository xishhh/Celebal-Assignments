import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "rag_docs"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 3

LLM_PROVIDER = "groq"
GROQ_MODEL = "llama-3.1-8b-instant"
OPENAI_MODEL = "gpt-4o-mini"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def validate_config():
    missing = []
    if not GROQ_API_KEY and LLM_PROVIDER == "groq":
        missing.append("GROQ_API_KEY")
    if not OPENAI_API_KEY and LLM_PROVIDER == "openai":
        missing.append("OPENAI_API_KEY")

    if missing:
        print(f"Warning: Missing environment variable(s): {', '.join(missing)}")
        print(f"Set them before running the app.")
        return False
    return True


validate_config()
