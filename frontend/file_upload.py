"""PDF attachment handling for chat conversations."""
import os
import tempfile
from collections.abc import Sequence

import streamlit as st

from backend.conversations import get_thread_documents
from backend.rag.ingestion import ingest_rag_document


def render_chat_input():
    return st.chat_input(
        "Type here",
        accept_file=True,
        file_type=["pdf"],
    )


def process_uploaded_files(uploaded_files: Sequence, thread_id: str) -> None:
    for uploaded_pdf in uploaded_files:
        temporary_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temporary_file:
                temporary_file.write(uploaded_pdf.getvalue())
                temporary_file_path = temporary_file.name

            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                document_id = ingest_rag_document(
                    temporary_file_path,
                    thread_id=thread_id,
                    source_name=uploaded_pdf.name,
                )

            st.toast(
                f"{uploaded_pdf.name} is ready in this conversation "
                f"({document_id[:8]}).",
                icon="✅",
            )
        except Exception as error:
            st.error(f"PDF processing failed: {error}")
        finally:
            if temporary_file_path and os.path.exists(temporary_file_path):
                os.remove(temporary_file_path)


def render_thread_documents(thread_id: str) -> None:
    attached_documents = get_thread_documents(thread_id)
    if attached_documents:
        with st.sidebar.expander("Documents in this chat"):
            for document in attached_documents:
                st.caption(document["filename"])
