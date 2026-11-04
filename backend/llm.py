"""Provider clients configured for Nexora AI."""
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.config import load_environment

load_environment()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=512
)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
