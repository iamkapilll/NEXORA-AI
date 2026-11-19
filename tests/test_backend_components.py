"""Focused tests for API-free Nexora backend behavior."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
import requests

from backend.conversations import build_conversation_title
from backend.rag.ingestion import get_thread_index_path
from backend.state import build_retrieval_query, should_retrieve
from backend.tools.calculator import calculator
from backend.tools.stocks import get_stock_price
from backend.tools.weather import get_current_weather
from frontend.file_upload import process_uploaded_files


class ConversationTests(unittest.TestCase):
    def test_title_skips_greeting_and_uses_first_question(self):
        messages = [
            HumanMessage(content="Hello!"),
            HumanMessage(content="Can you tell me about LangGraph?"),
        ]
        self.assertEqual(build_conversation_title(messages), "About LangGraph")

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
        self.assertEqual(calculator.invoke({"expression": "math.sqrt(16)"}), "4.0")

    def test_calculator_rejects_unbounded_power(self):
        result = calculator.invoke({"expression": "2 ** 101"})
        self.assertIn("Power exponent is too large.", result)

    def test_calculator_rejects_sequence_multiplication(self):
        result = calculator.invoke({"expression": "sum([1] * 100000)"})
        self.assertIn("Only numeric arithmetic is allowed.", result)

    def test_calculator_rejects_oversized_expressions(self):
        result = calculator.invoke({"expression": "1" * 257})
        self.assertEqual(result, "Calculation error: Expression is too long.")

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

    def test_stock_service_http_error_returns_a_clear_error(self):
        with (
            patch("backend.tools.stocks.os.getenv", return_value="test-key"),
            patch("backend.tools.stocks.requests.get") as get,
        ):
            get.return_value.raise_for_status.side_effect = requests.HTTPError(
                "request failed with test-key"
            )
            result = get_stock_price.invoke({"symbol": "TEST"})

        self.assertEqual(
            result,
            {"Error": "The stock service returned an HTTP error."},
        )

    def test_stock_symbol_is_normalized_before_lookup(self):
        with (
            patch("backend.tools.stocks.os.getenv", return_value="test-key"),
            patch("backend.tools.stocks.requests.get") as get,
        ):
            get.return_value.json.return_value = {"price": "123.45"}
            result = get_stock_price.invoke({"symbol": "  aapl "})

        self.assertEqual(result, {"price": "123.45"})
        self.assertEqual(get.call_args.kwargs["params"]["symbol"], "AAPL")

    def test_empty_stock_symbol_does_not_call_service(self):
        with (
            patch("backend.tools.stocks.os.getenv", return_value="test-key"),
            patch("backend.tools.stocks.requests.get") as get,
        ):
            result = get_stock_price.invoke({"symbol": "  "})

        self.assertEqual(result, {"Error": "A stock symbol is required."})
        get.assert_not_called()

    def test_weather_connection_error_does_not_expose_request_details(self):
        with (
            patch("backend.tools.weather.os.getenv", return_value="test-key"),
            patch(
                "backend.tools.weather.requests.get",
                side_effect=requests.ConnectionError(
                    "request failed with test-key"
                ),
            ),
        ):
            result = get_current_weather.invoke({"location": "London"})

        self.assertEqual(
            result,
            "Could not connect to the weather service. Please try again.",
        )


class FileUploadTests(unittest.TestCase):
    def test_processing_error_does_not_expose_exception_details(self):
        uploaded_pdf = SimpleNamespace(
            name="resume.pdf",
            getvalue=lambda: b"not important to the mocked ingestion",
        )
        with (
            patch(
                "frontend.file_upload.ingest_rag_document",
                side_effect=RuntimeError("provider request included private data"),
            ),
            patch("frontend.file_upload.st.spinner", return_value=nullcontext()),
            patch("frontend.file_upload.st.error") as display_error,
        ):
            process_uploaded_files([uploaded_pdf], "thread-a")

        display_error.assert_called_once_with(
            "Could not process resume.pdf. Verify that the PDF is readable "
            "and the embedding service is configured, then retry."
        )


if __name__ == "__main__":
    unittest.main()
