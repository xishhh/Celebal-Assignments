import logging
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_pdf(filepath: Path) -> List[Dict]:
    reader = PdfReader(str(filepath))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages.append({
                "text": text.strip(),
                "source": filepath.name,
                "page": i + 1,
            })
    return pages


def load_txt(filepath: Path) -> List[Dict]:
    text = filepath.read_text(encoding="utf-8").strip()
    if text:
        return [{"text": text, "source": filepath.name, "page": None}]
    return []


def load_documents(data_dir: str) -> List[Dict]:
    dir_path = Path(data_dir)

    if not dir_path.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist. "
            f"Please create it or set DATA_DIR in config.py to an existing path."
        )
    if not dir_path.is_dir():
        raise NotADirectoryError(
            f"'{data_dir}' is not a directory. "
            f"Please provide a valid directory path."
        )

    supported_extensions = {".pdf", ".txt", ".md"}
    files = sorted(
        [f for f in dir_path.iterdir() if f.suffix.lower() in supported_extensions]
    )
    if not files:
        raise ValueError(
            f"No supported files (.pdf, .txt, .md) found in '{data_dir}'. "
            f"Add at least one document to ingest."
        )

    all_docs: List[Dict] = []
    for filepath in files:
        try:
            if filepath.suffix.lower() == ".pdf":
                docs = load_pdf(filepath)
            else:
                docs = load_txt(filepath)
            all_docs.extend(docs)
            logger.info("Loaded %d document(s) from %s", len(docs), filepath.name)
        except Exception as e:
            logger.warning("Skipping %s: %s", filepath.name, e)

    if not all_docs:
        logger.warning("No documents could be loaded from any file.")

    return all_docs
