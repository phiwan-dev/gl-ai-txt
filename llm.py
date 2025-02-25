# based on and modified from the wiki over at https://python.langchain.com/v0.1/docs/get_started/quickstart/ 
from typing import List
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

MODEL_NAME: str = "llama3.2:1b"
llm: OllamaLLM = OllamaLLM(model=MODEL_NAME)

# get raw docs
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# load embedding model
from langchain_ollama import OllamaEmbeddings
embeddings: OllamaEmbeddings = OllamaEmbeddings(model=MODEL_NAME)

# parse raw docs to vector store
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
text_splitter = RecursiveCharacterTextSplitter()
documents: List[Document] = text_splitter.split_documents(docs)
#print(f"{docs=}")
#print("#####################")
#print(f"{documents=}")  
vector: FAISS = FAISS.from_documents(documents, embeddings)

# create document chain
from langchain.chains.combine_documents import create_stuff_documents_chain # type: ignore
# prompt = ChatPromptTemplate.from_template(
# """Answer the following question based only on the provided context:
# <context>
# {context}
# </context>
# Question: {input}""")
# document_chain = create_stuff_documents_chain(llm, prompt)

# # create retrieval chain
from langchain.chains import create_retrieval_chain
from langchain_core.vectorstores import VectorStoreRetriever
retriever: VectorStoreRetriever = vector.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# history aware retriever
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# create history aware document chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)
# history aware retrieval chain
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# test
from langchain_core.messages import HumanMessage, AIMessage
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
print(response["answer"])

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Answer the user's questions based on the below context:\n\n{context}"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
# ])
# document_chain = create_stuff_documents_chain(llm, prompt)
# retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# get output
#response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
#print(response["answer"])



