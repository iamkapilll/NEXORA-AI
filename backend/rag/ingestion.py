"""PDF ingestion and thread-scoped FAISS index access."""
import hashlib
import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from backend.config import THREADS_DB_PATH
from backend.conversations import get_thread_documents
from backend.llm import embeddings
from backend.memory import document_conn, document_lock


def get_thread_index_path(thread_id: str) -> str:
    thread_key = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
    return os.path.join(THREADS_DB_PATH, thread_key)

def ingest_rag_document(
    file_path: str,
    thread_id: str,
    source_name: str | None = None
) -> str:
    if not thread_id:
        raise ValueError("A conversation thread ID is required to index a PDF.")

    filename = source_name or os.path.basename(file_path)
    content_hash = hashlib.sha256()
    with open(file_path, "rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            content_hash.update(block)
    file_hash = content_hash.hexdigest()

    with document_lock:
        document_conn.execute("BEGIN IMMEDIATE")
        try:
            existing_document = document_conn.execute(
                """
                SELECT document_id
                FROM conversation_documents
                WHERE thread_id = ? AND content_hash = ?
                """,
                (thread_id, file_hash)
            ).fetchone()
            if existing_document:
                document_conn.commit()
                print(
                    f"[RAG] Duplicate upload skipped: thread_id={thread_id}, "
                    f"document_id={existing_document[0]}, filename={filename}",
                    flush=True
                )
                return existing_document[0]

            document_id = str(uuid.uuid4())
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            for document in docs:
                document.metadata["document_id"] = document_id
                document.metadata["filename"] = filename
                document.metadata["source"] = filename
                document.metadata["page_number"] = (
                    int(document.metadata.get("page", 0)) + 1
                )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            if not chunks:
                raise ValueError(f"No text could be extracted from {filename}.")

            for chunk in chunks:
                chunk.metadata["document_id"] = document_id
                chunk.metadata["filename"] = filename

            index_path = get_thread_index_path(thread_id)
            os.makedirs(index_path, exist_ok=True)
            index_file = os.path.join(index_path, "index.faiss")
            pickle_file = os.path.join(index_path, "index.pkl")

            if os.path.isfile(index_file) and os.path.isfile(pickle_file):
                vector_store = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
                vector_store.add_documents(chunks)
            else:
                vector_store = FAISS.from_documents(chunks, embeddings)

            vector_store.save_local(index_path)
            document_conn.execute(
                """
                INSERT INTO conversation_documents
                    (thread_id, document_id, filename, content_hash)
                VALUES (?, ?, ?, ?)
                """,
                (thread_id, document_id, filename, file_hash)
            )
            document_conn.commit()

            print(
                f"[RAG] Indexed document: thread_id={thread_id}, "
                f"document_id={document_id}, filename={filename}, "
                f"chunks={len(chunks)}",
                flush=True
            )
            return document_id
        except Exception:
            document_conn.rollback()
            raise

def get_retriever(thread_id: str):
    associated_documents = get_thread_documents(thread_id)
    if not associated_documents:
        return None

    index_path = get_thread_index_path(thread_id)
    index_file = os.path.join(index_path, "index.faiss")
    pickle_file = os.path.join(index_path, "index.pkl")
    if not os.path.isfile(index_file) or not os.path.isfile(pickle_file):
        raise FileNotFoundError(
            f"FAISS index is missing for conversation {thread_id}."
        )
    vector_store = FAISS.load_local(
        folder_path=index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4,
            "fetch_k": vector_store.index.ntotal,
            "filter": {
                "document_id": {
                    "$in": [
                        document["document_id"]
                        for document in associated_documents
                    ]
                }
            }
        }
    )
