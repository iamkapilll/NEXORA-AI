"""Agent tools and their model binding."""
from backend.llm import llm
from backend.tools.calculator import calculator
from backend.tools.search import search_tool
from backend.tools.stocks import get_stock_price
from backend.tools.weather import get_current_weather
from backend.rag.retrieval import rag_tool

chat_tools = [search_tool, calculator, get_stock_price, get_current_weather]
tools = [*chat_tools, rag_tool]
llm_with_tools = llm.bind_tools(chat_tools)
