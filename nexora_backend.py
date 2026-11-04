"""Compatibility exports for the original Nexora backend module."""
from backend.conversations import (
    build_conversation_title,
    get_all_threads,
    get_conversation_title,
    get_thread_documents,
    save_conversation_title,
    sync_conversation_titles,
)
from backend.graph import chatbot, graph
from backend.llm import embeddings, llm
from backend.rag.ingestion import (
    get_retriever,
    get_thread_index_path,
    ingest_rag_document,
)
from backend.rag.retrieval import rag_tool
from backend.state import (
    ChatState,
    build_retrieval_query,
    is_document_question,
    should_retrieve,
)
from backend.tools import (
    calculator,
    chat_tools,
    get_current_weather,
    get_stock_price,
    llm_with_tools,
    search_tool,
    tools,
)

__all__ = [
    "build_conversation_title",
    "build_retrieval_query",
    "calculator",
    "chat_tools",
    "chatbot",
    "ChatState",
    "embeddings",
    "get_all_threads",
    "get_conversation_title",
    "get_current_weather",
    "get_retriever",
    "get_stock_price",
    "get_thread_documents",
    "get_thread_index_path",
    "graph",
    "ingest_rag_document",
    "is_document_question",
    "llm",
    "llm_with_tools",
    "rag_tool",
    "save_conversation_title",
    "search_tool",
    "should_retrieve",
    "sync_conversation_titles",
    "tools",
]
