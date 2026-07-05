import logging
from typing import Dict, List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

    def add_chunks(self, chunks: List[Dict]) -> None:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            ids = [c["chunk_id"] for c in batch]
            documents = [c["text"] for c in batch]
            metadatas = [
                {"source": c["source"], "page": str(c["page"]) if c["page"] else ""}
                for c in batch
            ]
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            logger.info(
                "Added batch %d–%d (%d chunks)",
                i,
                min(i + BATCH_SIZE, len(chunks)) - 1,
                len(batch),
            )

    def query(self, query_text: str, top_k: int = 3) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        out = []
        for idx, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][idx]
            page = metadata.get("page", "")
            out.append({
                "text": results["documents"][0][idx],
                "source": metadata.get("source", ""),
                "page": int(page) if page.isdigit() else None,
                "distance": results["distances"][0][idx],
                "chunk_id": doc_id,
            })
        return out

    def reset(self) -> None:
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except ValueError:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

    def count(self) -> int:
        return self.collection.count()
