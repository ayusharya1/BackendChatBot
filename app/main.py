import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from .prompts import fallback_prompt_template, prompt
from langchain_openai import OpenAIEmbeddings
from .config import OPENAI_API_KEY # ✅ Load from centralized config

# ✅ FastAPI App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# 🧠 Vectorstore memory (reuse per access code)
vectorstore_cache = {}

# 🔁 Load saved FAISS vectorstore
def get_chain_from_saved_vectorstore(path, model="gpt-4.1-nano-2025-04-14"):
    vectorstore = FAISS.load_local(
        folder_path=path,
        embeddings=OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model=model, temperature=0.7)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# 📥 Request / Response Models
class AskRequest(BaseModel):
    question: str
    mode: Literal["normal", "professional"] = "normal"
    access_code: Optional[str] = None

class AskResponse(BaseModel):
    answer: str

# 🔄 Main API Endpoint
@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    query = request.question.strip()
    mode = request.mode
    code = (request.access_code or "").strip()

    try:
        use_fallback = True
        answer = ""
        if mode == "professional" and code:
            # 🔐 Map access code to vectorstore directory
            vectorstore_path_map = {
                "IISERM": "vectorstores/iiserm_faiss",
                # Add more codes → folder paths if needed
            }
            vs_path = vectorstore_path_map.get(code)

            if not vs_path or not Path(vs_path).exists():
                return {"answer": f"⚠️ Invalid or missing access code."}

            if code not in vectorstore_cache:
                try:
                    vectorstore_cache[code] = get_chain_from_saved_vectorstore(vs_path)
                except Exception as e:
                    return {"answer": f"⚠️ Failed to load vectorstore: {str(e)}"}

            chain = vectorstore_cache[code]
            try:
                result = chain.invoke({"question": query})
                rag_answer = result.get("answer", "").strip()
                if rag_answer and not any(
                    bad_phrase in rag_answer.lower()
                    for bad_phrase in [
                        "document doesn't specify",
                        "not found in the document",
                        "document does not",
                        "not mentioned in the document"
                    ]
                ):
                    return {"answer": rag_answer}
            except Exception as e:
                # fall through to fallback
                pass

        # 🌐 Normal Mode or Fallback
        fallback_prompt = fallback_prompt_template.replace("{query}", query)
        fallback_llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0.7)
        response = fallback_llm.invoke([HumanMessage(content=fallback_prompt)])
        return {"answer": response.content}

    except Exception as e:
        return {"answer": f"⚠️ Unexpected error: {str(e)}"}

# 🔃 Server startup configuration
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (for Render) or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False  # Disable reload in production
    )
