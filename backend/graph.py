"""LangGraph construction and compiled chatbot."""
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from backend.memory import checkpoint
from backend.nodes import chat_node, retrieve_node
from backend.state import ChatState, should_retrieve
from backend.tools import tools

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("tools", tool_node)
graph.add_conditional_edges(
    START,
    should_retrieve,
    {"retrieve": "retrieve", "chat_node": "chat_node"},
)
graph.add_edge("retrieve", "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)
