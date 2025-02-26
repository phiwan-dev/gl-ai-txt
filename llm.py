# based on and modified from the wiki over at https://python.langchain.com/v0.1/docs/get_started/quickstart/ 

import time
start_time = time.time()

from langchain_ollama import OllamaLLM
MODEL_NAME: str = "llama3.2:1b"
llm: OllamaLLM = OllamaLLM(model=MODEL_NAME)

# get raw docs
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
print(f"[{time.time() - start_time}]\tfound {len(docs)} documents")

# load embedding model
from langchain_ollama import OllamaEmbeddings
embeddings: OllamaEmbeddings = OllamaEmbeddings(model=MODEL_NAME)
print(f"[{time.time() - start_time}]\tloaded embeddings model")

# parse raw docs to vector store
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
text_splitter = RecursiveCharacterTextSplitter()
documents: List[Document] = text_splitter.split_documents(docs)
#print(f"{docs=}")
#print("#####################")
#print(f"{documents=}")  
vector: FAISS = FAISS.from_documents(documents, embeddings)
print(f"[{time.time() - start_time}]\tcreated vector store")

# history aware retriever
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])
retriever: VectorStoreRetriever = vector.as_retriever()
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
print(f"[{time.time() - start_time}]\tcreated history aware retriever")

# create history aware document chain
from langchain.chains.combine_documents import create_stuff_documents_chain # type: ignore
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)
print(f"[{time.time() - start_time}]\tcreated history aware document chain")

# history aware retrieval chain
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
print(f"[{time.time() - start_time}]\tcreated history aware retrieval chain")

# create history
from langchain_core.messages import HumanMessage, AIMessage
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]

# get output
response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
print(f"[{time.time() - start_time}]\t{response["answer"]}")




