"""Conversation-isolated retrieval tool for attached PDFs."""
import logging

from langchain_core.tools import tool
from backend.conversations import get_thread_documents
from backend.rag.ingestion import get_retriever

logger = logging.getLogger(__name__)


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
        logger.info("PDF retrieval skipped because no documents are attached.")
        return "No document is attached to this conversation."

    filenames = {
        document["document_id"]: document["filename"]
        for document in associated_documents
    }
    logger.info("Searching the PDFs attached to this conversation.")
    retriever = get_retriever(thread_id)
    if retriever is None:
        return "No document is attached to this conversation."
    documents = retriever.invoke(query)

    if not documents:
        logger.info("PDF retrieval found no matching chunks.")
        return "No relevant information was found in this conversation's PDFs."

    logger.info("Retrieved %d PDF chunk(s).", len(documents))
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
