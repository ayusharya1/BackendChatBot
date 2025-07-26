# app/llm_chain.py
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from .prompts import prompt

vectorstore_cache = {}

def get_chain_from_saved_vectorstore(path: str, model="gpt-4.1-nano-2025-04-14"):
    vectorstore = FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model=model, temperature=0.7)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
