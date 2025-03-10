

MODEL_NAME = "llama3.2:1b"


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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20,length_function=len)
documents: List[Document] = text_splitter.split_documents(raw_documents)
vector_store = Chroma(embedding_function=embeddings, persist_directory="./cache")
print(f"{len(documents)}")
for doc in documents:
    _ = vector_store.add_documents([doc])
    print("#", end="", flush=True)



from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()



for message, _ in graph.stream({"question": "What npc are there"}, stream_mode="messages"):
    print(f"{message.content}", end="", flush=True)