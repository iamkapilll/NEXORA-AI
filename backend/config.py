"""Project paths and environment initialization.

All paths are resolved relative to ``BASE_DIR`` rather than the process working
directory, so the app behaves the same whether it is started from the project
root or from another location.
"""
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "faiss_db"
THREADS_DB_PATH = DB_PATH / "threads"
DOCUMENT_DB_PATH = BASE_DIR / "chatbot.db"


def load_environment() -> None:
    """Load variables from the project-level ``.env`` file, if it exists.

    Called at import time so every module sees the same configuration. Values
    already present in the process environment take precedence, and a missing
    ``.env`` file is not an error.
    """
    load_dotenv(BASE_DIR / ".env")


load_environment()
