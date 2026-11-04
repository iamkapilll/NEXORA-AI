"""Stock quote lookup tool."""
import os
import requests
from langchain_core.tools import tool


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {
            "Note": "Stock API key is missing. "
            "Set the ALPHAVANTAGE_API_KEY environment variable."
        }

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": api_key,
            },
            timeout=10,
        )
        return response.json()
    except requests.RequestException:
        return {"Error": "Could not connect to the stock service."}
