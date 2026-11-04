"""Conversation-isolated retrieval tool for attached PDFs."""
from langchain_core.tools import tool
from backend.conversations import get_thread_documents
from backend.rag.ingestion import get_retriever


@tool
def rag_tool(query: str, thread_id: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
        thread_id: The conversation whose attached PDFs may be searched.
    """
    associated_documents = get_thread_documents(thread_id)
    if not associated_documents:
        print(
            f"[RAG] No documents attached: thread_id={thread_id}",
            flush=True
        )
        return "No document is attached to this conversation."

    document_ids = {
        document["document_id"] for document in associated_documents
    }
    filenames = {
        document["document_id"]: document["filename"]
        for document in associated_documents
    }
    print(
        f"[RAG] Retrieving: thread_id={thread_id}, "
        f"documents={[(document['document_id'], document['filename']) for document in associated_documents]}",
        flush=True
    )
    retriever = get_retriever(thread_id)
    if retriever is None:
        return "No document is attached to this conversation."
    documents = retriever.invoke(query)

    if not documents:
        print(
            f"[RAG] No matching chunks: thread_id={thread_id}, "
            f"document_ids={sorted(document_ids)}",
            flush=True
        )
        return "No relevant information was found in this conversation's PDFs."

    retrieved_document_ids = {
        document.metadata.get("document_id") for document in documents
    }
    print(
        f"[RAG] Retrieved {len(documents)} chunk(s): thread_id={thread_id}, "
        f"documents={[(document_id, filenames.get(document_id, 'Unknown filename')) for document_id in sorted(retrieved_document_ids) if document_id]}",
        flush=True
    )
    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        document_id = document.metadata.get("document_id")
        filename = filenames.get(document_id, "Unknown filename")
        page = document.metadata.get("page_number", "Unknown page")

        formatted_documents.append(
            f"### Retrieved document {index}\n"
            f"- **Document ID:** {document_id}\n"
            f"- **Filename:** {filename}\n"
            f"- **Page:** {page}\n\n"
            f"**Content**\n\n{document.page_content}"
        )

    return "\n\n".join(formatted_documents)
