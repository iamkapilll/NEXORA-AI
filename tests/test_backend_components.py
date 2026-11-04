"""Focused tests for API-free Nexora backend behavior."""
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
import requests

from backend.conversations import build_conversation_title
from backend.rag.ingestion import get_thread_index_path
from backend.state import build_retrieval_query, should_retrieve
from backend.tools.calculator import calculator
from backend.tools.stocks import get_stock_price


class ConversationTests(unittest.TestCase):
    def test_title_skips_greeting_and_uses_first_question(self):
        messages = [
            HumanMessage(content="Hello!"),
            HumanMessage(content="Can you tell me about LangGraph?"),
        ]
        self.assertEqual(build_conversation_title(messages), "About Langgraph")

    def test_thread_index_paths_are_stable_and_isolated(self):
        first = get_thread_index_path("conversation-one")
        self.assertEqual(first, get_thread_index_path("conversation-one"))
        self.assertNotEqual(first, get_thread_index_path("conversation-two"))


class RoutingTests(unittest.TestCase):
    def test_document_question_retrieves_for_attached_thread(self):
        state = {"messages": [HumanMessage(content="Summarize my resume")]}
        config = {"configurable": {"thread_id": "thread-a"}}
        with patch("backend.state.get_thread_documents", return_value=[{}]):
            self.assertEqual(should_retrieve(state, config), "retrieve")

    def test_tool_question_takes_priority_over_document_route(self):
        state = {
            "messages": [
                HumanMessage(content="Calculate my work experience in years")
            ]
        }
        config = {"configurable": {"thread_id": "thread-a"}}
        with patch("backend.state.get_thread_documents", return_value=[{}]):
            self.assertEqual(should_retrieve(state, config), "chat_node")

    def test_unattached_conversation_cannot_retrieve_other_thread_documents(self):
        state = {"messages": [HumanMessage(content="Summarize my resume")]}
        config = {"configurable": {"thread_id": "thread-b"}}
        with patch("backend.state.get_thread_documents", return_value=[]):
            self.assertEqual(should_retrieve(state, config), "chat_node")

    def test_document_follow_up_includes_prior_question(self):
        messages = [
            HumanMessage(content="Tell me about my project experience"),
            AIMessage(content="The project details are..."),
            HumanMessage(content="What about the timeline?"),
        ]
        self.assertEqual(
            build_retrieval_query(messages),
            "Tell me about my project experience\n"
            "Follow-up question: What about the timeline?",
        )


class ToolTests(unittest.TestCase):
    def test_calculator_supports_arithmetic_without_external_services(self):
        self.assertEqual(calculator.invoke({"expression": "2 + 3 * 4"}), "14")

    def test_stock_service_failure_does_not_expose_request_details(self):
        with (
            patch("backend.tools.stocks.os.getenv", return_value="test-key"),
            patch(
                "backend.tools.stocks.requests.get",
                side_effect=requests.ConnectionError(
                    "request failed with test-key"
                ),
            ),
        ):
            result = get_stock_price.invoke({"symbol": "TEST"})
        self.assertEqual(
            result,
            {"Error": "Could not connect to the stock service."},
        )


if __name__ == "__main__":
    unittest.main()
