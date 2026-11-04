"""Chat history rendering and streamed assistant responses."""
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.conversations import save_conversation_title
from backend.graph import chatbot


def render_chat_history(message_history: list[dict[str, str]]) -> None:
    for message in message_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
            else:
                st.text(message["content"])


def respond_to_user(user_input: str, thread_id: str) -> None:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    save_conversation_title(
        thread_id,
        [
            HumanMessage(content=message["content"])
            for message in st.session_state["message_history"]
            if message["role"] == "user"
        ],
    )

    with st.chat_message("user"):
        st.text(user_input)

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_trace",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True,
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if (
                    isinstance(message_chunk, AIMessage)
                    and isinstance(message_chunk.content, str)
                ):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
    st.rerun()
