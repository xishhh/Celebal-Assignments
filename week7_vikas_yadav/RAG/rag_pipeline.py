import logging

from config import TOP_K
from loader import load_documents
from chunker import chunk_documents
from vectorstore import VectorStore
from llm import generate_answer

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self.vectorstore = VectorStore()

    def ingest(self, data_dir: str, reset: bool = False) -> int:
        if reset:
            self.vectorstore.reset()
            print("Vector store reset.")

        print(f"Ingesting documents from {data_dir} ...")
        docs = load_documents(data_dir)
        print(f"Loaded {len(docs)} document(s).")

        print("Chunking documents ...")
        chunks = chunk_documents(docs)
        print(f"Created {len(chunks)} chunk(s).")

        print("Indexing into vector store ...")
        self.vectorstore.add_chunks(chunks)
        count = self.vectorstore.count()
        print(f"Done. Total indexed: {count} chunk(s).")
        return count

    def query(self, question: str, top_k: int = TOP_K) -> dict:
        retrieved = self.vectorstore.query(question, top_k=top_k)
        answer = generate_answer(question, retrieved)
        sources = list({r["source"] for r in retrieved})
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved,
        }



