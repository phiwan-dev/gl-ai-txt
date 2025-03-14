

MODEL_NAME = "qwen2.5:7b"
#MODEL_NAME = "phi4"


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
#vector_store = Chroma(embedding_function=embeddings, persist_directory="./cache", )
#vector_store = Chroma(embedding_function=embeddings)
#vector_store = Chroma.from_documents(documents, embeddings, persist_directory="./cache")

from langchain_community.vectorstores import FAISS
# build vs load vectorstore. True=build, False=load
if True:
    vector_store = FAISS.from_documents(documents[:1], embeddings)
    print(f"  [processed docs] ({len(documents)})")
    for doc in documents:
        _ = vector_store.add_documents([doc])
        print("#", end="", flush=True)
    print("")
    vector_store.save_local("cache/faiss")
else:
    # load vector store
    vector_store = FAISS.load_local("cache/faiss", embeddings, allow_dangerous_deserialization=True)
print("vector store is ready")


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


# Define prompt for question-answering
from langchain_core.prompts import PromptTemplate
raw_response_prompt = """{chat_history}

{context}

IMPORTANT QUESTION: {question}

You are a chat bot to answer user questions about the game galaxy life.
Answer the user question at the above using the given context and the chat message history.
Do not make up an answer."""
response_prompt = PromptTemplate.from_template(raw_response_prompt)



# Define state for application
class State(TypedDict):
    question: str
    query: str
    context: List[Document]
    answer: str
    chat_history: List[str]


raw_rephrase_prompt: str = """Last Response: {last_response}

Question: {question}

You are a chat bot to answer user questions about the game galaxy life.
Rephrase the users question to make it independent of the last response.
Substitue any relative references to it with the actual content.
Only respond with a rephrased question."""
rephrase_prompt: PromptTemplate = PromptTemplate.from_template(raw_rephrase_prompt)


def analyze_question(state: State):
    try:
        last_response: str = state["last_response"]
        print("last_response exists")
    except KeyError:
        last_response = "This is the start of the conversation. Please ask your question."
        print("last_response does NOT exist")
    prompt = rephrase_prompt.invoke({"question": state["question"], "last_response": last_response})
    response = llm.invoke(prompt)
    return {"rephrased_question": response.content, "last_response": last_response}


def generate_query(state: State):
    prompt = query_prompt.invoke({"question": state["rephrased_question"], "last_response": state["last_response"]})
    print("REPHRASED_QUESTION:")
    print(state["rephrased_question"])
    response = llm.invoke(prompt)
    return {"query": response.content}


def retrieve(state: State):
    #retrieved_docs = vector_store.search(state["query"], search_type="mmr")
    retrieved_docs = vector_store.similarity_search(state["query"], k=4)
    return {"context": retrieved_docs}


def generate_response(state: State):
    docs_content = "\n------\n".join(doc.page_content for doc in state["context"])
    messages = response_prompt.invoke({
        "chat_history": "\n".join(state["chat_history"]), 
        "context": docs_content, 
        "question": state["question"]
    })
    print("QUERY:")
    print(state["query"])
    print("history:")
    print(state["chat_history"])
    print("CONTEXT:")
    print(docs_content)
    print("ANSWER:")
    response = llm.invoke(messages)
    new_chat_history = state["chat_history"] + [f"AIMessage: {response.content}"]
    return {"answer": response.content, "chat_history": new_chat_history}


# Compile graph using memory checkpointer for message history persistance across prompts
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
source_graph = StateGraph(State).add_sequence([analyze_question, generate_query, retrieve, generate_response])
source_graph.add_edge(START, "analyze_question")
graph = source_graph.compile(checkpointer=memory)


# print output
config = {"configurable": {"thread_id": "1"}}
question = "What NPC are there"
while True:
    for message, metadata in graph.stream({"question": question}, config=config, stream_mode="messages"):
        if metadata["langgraph_node"] == "generate_response":
            print(f"{message.content}", end="", flush=True)
    question = input("\n> ")
    if question == "exit":
        break