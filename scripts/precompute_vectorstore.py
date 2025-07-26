import os
from dotenv import load_dotenv

load_dotenv()  # 👈 This loads your OPENAI_API_KEY from .env

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # ✅ Updated import
from app.pdf_loader import extract_metadata_from_folder

# 🗂️ Paths
pdf_folder = Path("pdfs/IISERMohali")
output_path = "vectorstores/iiserm_faiss"

print(f"📄 Reading PDFs from {pdf_folder}")
blocks = extract_metadata_from_folder(pdf_folder)

# 📎 Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
texts = splitter.split_text("\n\n".join(blocks))

# 🔐 Use OpenAI embeddings
print("🔍 Creating embeddings...")
embeddings = OpenAIEmbeddings()  # Will use the OPENAI_API_KEY from env
vectorstore = FAISS.from_texts(texts, embeddings)
vectorstore.save_local(output_path)

print("✅ Vectorstore saved to", output_path)
