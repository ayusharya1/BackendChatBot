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
    try:
        vectorstore = FAISS.load_local(
            folder_path=path,
            embeddings=OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"⚠️ Failed to load vectorstore from {path}: {e}")
        # Try to regenerate if files don't exist
        if not Path(path).exists():
            print(f"🔄 Attempting to regenerate vectorstore at {path}")
            regenerate_vectorstore(path)
            vectorstore = FAISS.load_local(
                folder_path=path,
                embeddings=OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )
        else:
            raise e
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model=model, temperature=0.7)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

def regenerate_vectorstore(output_path):
    """Regenerate vectorstore from PDFs"""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from app.pdf_loader import extract_metadata_from_folder
        
        pdf_folder = Path("pdfs/IISERMohali")
        if not pdf_folder.exists():
            print(f"⚠️ PDF folder {pdf_folder} not found")
            return False
            
        print(f"📄 Reading PDFs from {pdf_folder}")
        blocks = extract_metadata_from_folder(pdf_folder)
        
        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        texts = splitter.split_text("\n\n".join(blocks))
        
        # Create embeddings
        print("🔍 Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts, embeddings)
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(output_path)
        
        print("✅ Vectorstore regenerated at", output_path)
        return True
    except Exception as e:
        print(f"❌ Failed to regenerate vectorstore: {e}")
        return False

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

            if not vs_path:
                return {"answer": f"⚠️ Invalid access code."}

            if code not in vectorstore_cache:
                try:
                    vectorstore_cache[code] = get_chain_from_saved_vectorstore(vs_path)
                except Exception as e:
                    print(f"⚠️ Failed to load vectorstore: {str(e)}")
                    # Try to regenerate if it doesn't exist
                    if not Path(vs_path).exists():
                        print(f"🔄 Attempting to regenerate vectorstore at {vs_path}")
                        if regenerate_vectorstore(vs_path):
                            vectorstore_cache[code] = get_chain_from_saved_vectorstore(vs_path)
                        else:
                            return {"answer": f"⚠️ Failed to regenerate vectorstore. Please try again."}
                    else:
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
