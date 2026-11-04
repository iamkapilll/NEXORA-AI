"""Nexora AI Streamlit application entry point."""
import streamlit as st

from backend.graph import chatbot
from frontend.chat_ui import render_chat_history, respond_to_user
from frontend.file_upload import (
    process_uploaded_files,
    render_chat_input,
    render_thread_documents,
)
from frontend.sidebar import initialize_conversation_state, render_sidebar

st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖")
st.title("Nexora AI — Your Intelligent Agentic Assistant")

initialize_conversation_state()
render_sidebar(chatbot)

thread_id = st.session_state["thread_id"]
render_chat_history(st.session_state["message_history"])
submission = render_chat_input()

if submission:
    if submission.files:
        process_uploaded_files(submission.files, thread_id)

render_thread_documents(thread_id)

if submission and submission.text:
    respond_to_user(submission.text, thread_id)
