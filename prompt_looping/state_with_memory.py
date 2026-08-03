from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()
GROQ_API_KEY = os.getenv("API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing groq api key in .env file")

llm = ChatGroq(
    model = "llama3-8b-8192",
    temperature = 0.3 
)
