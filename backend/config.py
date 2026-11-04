"""Project paths and environment initialization."""
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "faiss_db"
THREADS_DB_PATH = DB_PATH / "threads"
DOCUMENT_DB_PATH = BASE_DIR / "chatbot.db"


def load_environment() -> None:
    load_dotenv(BASE_DIR / ".env")


load_environment()
