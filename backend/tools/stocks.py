"""Stock quote lookup tool."""
import os
import requests
from langchain_core.tools import tool


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Look up a stock ticker such as 'AAPL' or 'TSLA' using Alpha Vantage.

    Whitespace is trimmed and the ticker is uppercased before the request.
    Returns the provider quote or an Error/Note message if unavailable.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {
            "Note": "Stock API key is missing. "
            "Set the ALPHAVANTAGE_API_KEY environment variable."
        }

    symbol = symbol.strip().upper()
    if not symbol:
        return {"Error": "A stock symbol is required."}

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
        response.raise_for_status()
        return response.json()
    except requests.HTTPError:
        return {"Error": "The stock service returned an HTTP error."}
    except requests.RequestException:
        return {"Error": "Could not connect to the stock service."}
