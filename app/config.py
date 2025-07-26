import os
from dotenv import load_dotenv

if not os.environ.get("RENDER") and os.path.exists(".env"):
    load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not set.")
