from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

# --- Load .env for local development ---
if not os.environ.get("RENDER") and os.path.exists(".env"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv is optional for cloud, required for local

# --- Set your OpenAI API key ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set. Please set it in your Render dashboard or in a .env file for local development.")
# --- Step 1: Load and extract PDF content ---
pdfreader = PdfReader('IISER Mohali.pdf')
raw_text = ''
for page in pdfreader.pages:
    content = page.extract_text()
    if content:
        raw_text += content

# --- Step 2: Split text into chunks ---
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# --- Step 3: Create embeddings and vector store ---
embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)

# --- Step 4: Set up memory and prompt ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt_template = """
You are Ridan, the friendly AI face of the IISER Mohali Library. Always answer in a warm, supportive, and concise tone—like a helpful friend.

INSTRUCTIONS:
- Always use the document below to answer.
- Do NOT say \"Based on the document...\" or mention any file.
- If the answer isn't in the document, answer using general knowledge politely.
- Keep answers short unless clarification is needed.
- Maintain continuity using chat history when follow-up is asked.

--- DOCUMENT CONTENT ---
{context}
--- END DOCUMENT ---

Chat History: {chat_history}
Question: {question}

Helpful Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=prompt_template
)

# --- Step 5: LLM and chain ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=document_search.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# --- Step 6: FastAPI setup ---
app = FastAPI()
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

# --- Step 7: Main Ask Endpoint ---
@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    query = request.question

    try:
        relevant_docs = document_search.similarity_search(query, k=4)
        has_relevant_context = any(doc.page_content.strip() for doc in relevant_docs)

        if has_relevant_context:
            result = chain.invoke({"question": query})
            return {"answer": result["answer"]}
        else:
            fallback_prompt = f"""
You are a helpful and witty AI assistant. Stay supportive, concise, and crack jokes occasionally.
If the user asks something outside library documents, use general knowledge.

User: {query}
Answer:
"""
            fallback_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
            fallback_response = fallback_llm.invoke([HumanMessage(content=fallback_prompt)])
            return {"answer": fallback_response.content}
    except Exception as e:
        return {"answer": f"Oops! Something went wrong: {str(e)}"}

# --- Step 8: Run server locally ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
