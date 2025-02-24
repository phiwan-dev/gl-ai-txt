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
prompt = ChatPromptTemplate.from_template(
"""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)

# create retrieval chain
from langchain.chains import create_retrieval_chain
from langchain_core.vectorstores import VectorStoreRetriever
retriever: VectorStoreRetriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# get output
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])



