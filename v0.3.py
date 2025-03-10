

MODEL_NAME = "qwen2.5:7b"


# init chat model
from langchain_ollama import ChatOllama
llm = ChatOllama(model=MODEL_NAME)


# get raw docs
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
doc_loader = DirectoryLoader("testdata", glob="**/*.txt", show_progress=True, use_multithreading=False, loader_cls=TextLoader)
raw_documents = doc_loader.load()


# create embeddings
from typing import List
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
embeddings: OllamaEmbeddings = OllamaEmbeddings(model=MODEL_NAME)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=500,length_function=len)
documents: List[Document] = text_splitter.split_documents(raw_documents)
#vector_store = Chroma(embedding_function=embeddings, persist_directory="./cache")
vector_store = Chroma(embedding_function=embeddings)
#print("  [raw docs]")
#for doc in raw_documents:
#    print(doc.page_content)
print(f"  [processed docs] ({len(documents)})")
for doc in documents:
    _ = vector_store.add_documents(documents)
    print("#", end="", flush=True)
print("")


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


# Define prompt for question-answering
from langchain_core.prompts import PromptTemplate
raw_response_prompt = """Answer the question at the end using the given context below.
Do not make up an answer.

{context}

Question: {question}"""
response_prompt = PromptTemplate.from_template(raw_response_prompt)



# Define state for application
class State(TypedDict):
    question: str
    query: str
    context: List[Document]
    answer: str


raw_query_prompt = """You are a chat bot to answer user questions about the game galaxy life.
Generate a short RAG query for the given question.
Do not ask for further information.
Question: {question}"""
query_prompt = PromptTemplate.from_template(raw_query_prompt)

def generate_query(state: State):
    messages = query_prompt.invoke({"question": state["question"]})
    response = llm.invoke(messages)
    return {"query": response.content}


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["query"])
    return {"context": retrieved_docs}


def generate_response(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = response_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile graph
source_graph = StateGraph(State).add_sequence([generate_query, retrieve, generate_response])
source_graph.add_edge(START, "generate_query")
graph = source_graph.compile()


# print output
question = "What buildings can i build"
for message, metadata in graph.stream({"question": question}, stream_mode="messages"):
    if metadata["langgraph_node"] == "generate_response":
        print(f"{message.content}", end="", flush=True)
print("")