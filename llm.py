# based on the tutorial at https://python.langchain.com/v0.1/docs/get_started/quickstart/ and modified
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL_NAME: str = "llama3.2:1b"

# create simple chain
prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are an expert on the documentation of the game Galaxy Life. Be friendly and answer any questions of the user according to your context. You can also tell jokes."),
    ("user", "{input}")
])
llm: OllamaLLM = OllamaLLM(model=MODEL_NAME)
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

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
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
#print(f"{docs=}")
#print("#####################")
#print(f"{documents=}")  
vector = FAISS.from_documents(documents, embeddings)



# output
print(
chain.invoke({"input": "Name 3 units and explain their abilities."})
)

