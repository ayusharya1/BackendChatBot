
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os



# Set up OpenAI API key from environment variable (Render sets this in the dashboard)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set. Please set it in your Render dashboard.")




# Step 1: Read the PDF
# provide the path of  pdf file/files.
pdfreader = PdfReader('IISER Mohali.pdf')





from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content






raw_text




# Step 2: Split into chunks

# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)





len(texts)




# Step 3: Create embeddings and vector DB
# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)





document_search





from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI




# Step 4: Load QA chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Step 5: Setup FastAPI
# --- FastAPI app setup ---
app = FastAPI()

# Allow CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response models ---
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

# Step 6: API Endpoint
# --- API endpoint ---
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    query = request.question
    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    return {"answer": result}

@app.get("/")
def health_check():
    return {"status": "ok"}

# Step 7: Local development server
# --- For local dev: run with `python app.py` ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)