

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100,length_function=len)
documents: List[Document] = text_splitter.split_documents(raw_documents)
vector_store = Chroma(embedding_function=embeddings, persist_directory="./cache")
#print("  [raw docs]")
#for doc in raw_documents:
#    print(doc.page_content)
#print(f"  [processed docs] ({len(documents)})")
#for doc in documents:
#    #_ = vector_store.add_documents(documents)
#    print("#", end="", flush=True)
#print("")


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


# Define prompt for question-answering
from langchain_core.prompts import PromptTemplate
raw_prompt = """Answer the question at the end using the given context below.
Do not make up an answer.

{context}

Question: {question}"""
generation_prompt = PromptTemplate.from_template(raw_prompt)


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
    messages = generation_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


# print output
print("\n\n   [answer]")
for message, metadata in graph.stream({"question": "Who is Sparragon"}, stream_mode="messages"):
    print(f"{message.content}", end="", flush=True)