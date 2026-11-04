"""Conversation navigation and history loading."""
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from backend.conversations import (
    get_all_threads,
    get_conversation_title,
    save_conversation_title,
)


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str) -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat() -> None:
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])
    save_conversation_title(st.session_state["thread_id"], [])


def load_conversation(chatbot, thread_id: str) -> list[dict[str, str]]:
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    converted = []
    for message in state.values.get("messages", []):
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        converted.append({"role": role, "content": message.content})
    return converted


def initialize_conversation_state() -> None:
    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []

    if "conversation_titles_initialized" not in st.session_state:
        from backend.conversations import sync_conversation_titles

        sync_conversation_titles()
        st.session_state["conversation_titles_initialized"] = True

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = generate_thread_id()
        save_conversation_title(st.session_state["thread_id"], [])

    st.session_state["chat_threads"] = get_all_threads()
    add_thread(st.session_state["thread_id"])


def render_sidebar(chatbot) -> None:
    st.sidebar.title("My Conversations")
    if st.sidebar.button("New Chat"):
        reset_chat()
        st.rerun()

    for thread_id in st.session_state["chat_threads"]:
        if st.sidebar.button(get_conversation_title(thread_id), key=thread_id):
            st.session_state["thread_id"] = thread_id
            st.session_state["message_history"] = load_conversation(
                chatbot, thread_id
            )
            st.rerun()
