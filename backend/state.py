"""LangGraph state and routing predicates."""
import re
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from backend.conversations import get_thread_documents


class ChatState(TypedDict, total=False):

    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_context: str

DOCUMENT_QUESTION_PATTERN = re.compile(
    r"\b(?:resume|résumé|cv|curriculum vitae|pdf|document|"
    r"experienc\w*|profession\w*|educat\w*|degree\w*|"
    r"qualific\w*|skill\w*|employ\w*|work history|"
    r"career\w*|project\w*|certificat\w*|achievement\w*)\b",
    re.IGNORECASE
)

TOOL_QUESTION_PATTERN = re.compile(
    r"\b(?:weather|forecast|temperature|stock|stocks|share price|"
    r"calculate|calculator|compute|latest news|current events|"
    r"search the web|web search)\b",
    re.IGNORECASE
)

DOCUMENT_FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:he|she|they|his|her|their|him|them|it|its|this|that|"
    r"these|those|there|former|latter)\b|\bwhat about\b|\btell me more\b",
    re.IGNORECASE
)

def is_document_question(query: str) -> bool:
    return DOCUMENT_QUESTION_PATTERN.search(query) is not None

def should_retrieve(
    state: ChatState,
    config: RunnableConfig
) -> str:
    messages = state.get("messages", [])
    thread_id = config.get("configurable", {}).get("thread_id")
    last_human_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
        ),
        None
    )

    if (
        last_human_index is None
        or not isinstance(thread_id, str)
        or not get_thread_documents(thread_id)
    ):
        return "chat_node"

    query = messages[last_human_index].content
    if not isinstance(query, str):
        return "chat_node"

    if TOOL_QUESTION_PATTERN.search(query):
        return "chat_node"

    if is_document_question(query):
        return "retrieve"

    if DOCUMENT_FOLLOW_UP_PATTERN.search(query):
        previous_question = next(
            (
                message.content
                for message in reversed(messages[:last_human_index])
                if isinstance(message, HumanMessage)
                and isinstance(message.content, str)
            ),
            None
        )
        if previous_question and is_document_question(previous_question):
            return "retrieve"

    return "chat_node"

def build_retrieval_query(messages: list[BaseMessage]) -> str:
    last_human_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
        ),
        None
    )
    if last_human_index is None:
        return ""

    query = messages[last_human_index].content
    if not isinstance(query, str):
        return ""

    if DOCUMENT_FOLLOW_UP_PATTERN.search(query):
        previous_question = next(
            (
                message.content
                for message in reversed(messages[:last_human_index])
                if isinstance(message, HumanMessage)
                and isinstance(message.content, str)
            ),
            None
        )
        if previous_question and is_document_question(previous_question):
            return f"{previous_question}\nFollow-up question: {query}"

    return query
