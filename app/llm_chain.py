# app/llm_chain.py
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .prompts import prompt

vectorstore_cache = {}

def get_chain_from_saved_vectorstore(path: str, model="gpt-4.1-nano-2025-04-14"):
    try:
        vectorstore = FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"⚠️ Failed to load vectorstore from {path}: {e}")
        # Try to regenerate if files don't exist
        if not Path(path).exists():
            print(f"🔄 Attempting to regenerate vectorstore at {path}")
            if regenerate_vectorstore(path):
                vectorstore = FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            else:
                raise Exception("Failed to regenerate vectorstore")
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

def regenerate_vectorstore(output_path: str):
    """Regenerate vectorstore from PDFs"""
    try:
        from .pdf_loader import extract_metadata_from_folder
        
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
