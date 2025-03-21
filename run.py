from typing import Any, List
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver


class GlBot():

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str


    def __init__(self) -> None:
        #self.MODEL_NAME = "gemma3:12b"
        #self.MODEL_NAME = "phi4"
        self.MODEL_NAME = "qwen2.5:7b"
        self.EMBEDDINGS_MODEL_NAME = "nomic-embed-text:latest"

        self.vector_store = self.embed_documents()
        self.llm = ChatOllama(model=self.MODEL_NAME)
        
        raw_rephrase_prompt: str = """Last Response: {last_response}

        Question: {question}

        You are a chat bot to answer user questions about the game galaxy life.
        Rephrase the users question to make it independent of the last response.
        Substitue any relative references to it with the actual content.
        Only respond with a rephrased question."""
        self.rephrase_prompt: PromptTemplate = PromptTemplate.from_template(raw_rephrase_prompt)

        raw_response_prompt: str = """context: {context}

        IMPORTANT QUESTION: {rephrased_question}

        You are a chat bot to answer user questions about the game galaxy life.
        Only answer the user question above using the given context.
        Do not make up an answer."""
        self.response_prompt: PromptTemplate = PromptTemplate.from_template(raw_response_prompt)

        self.run()


    def embed_documents(self) -> FAISS:
        # get raw docs
        doc_loader = DirectoryLoader("testdata", glob="**/*.txt", show_progress=True, use_multithreading=False, loader_cls=TextLoader)  
        raw_documents = doc_loader.load()

        # create embeddings
        embeddings: OllamaEmbeddings = OllamaEmbeddings(model=self.EMBEDDINGS_MODEL_NAME)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=500,length_function=len)
        documents: List[Document] = text_splitter.split_documents(raw_documents)

        # build vs load vectorstore. True=build, False=load
        if True:
            vector_store = FAISS.from_documents(documents[:1], embeddings)
            print(f"\t[PROCESS DOCS] ({len(documents)})")
            for doc in documents:
                _ = vector_store.add_documents([doc])
                print("#", end="", flush=True)
            print("")
            vector_store.save_local("cache/faiss")
        else:
            vector_store = FAISS.load_local("cache/faiss", embeddings, allow_dangerous_deserialization=True)
        return vector_store


    def analyze_question(self, state: State) -> dict[str, Any]:
        print("\t[ANALYZE QUESTION]")
        assert "question" in state, "No question was provided."

        try:
            last_response: str = state["answer"]
        except KeyError:
            last_response = "This is the start of the conversation. Please ask your question."
        prompt = self.rephrase_prompt.invoke({"question": state["question"], "last_response": last_response})
        response = self.llm.invoke(prompt)
        return {"question": response.content, "answer": last_response}


    def retrieve(self, state: State):
        print("\t[RETRIEVE DOCUMENTS]")
        assert "question" in state, "No query/rephrased question was provided. Call analyze_question before!"
        print("QUERY:")
        print(state["question"])

        #retrieved_docs = vector_store.search(state["query"], search_type="mmr")
        retrieved_docs = self.vector_store.similarity_search(state["question"], k=4)
        return {"context": retrieved_docs}


    def generate_response(self, state: State):
        print("\t[GENERATE RESPONSE]")
        assert "context" in state,          "No context provided. Likely unwanted! Call retrieve before!"
        assert "question" in state,         "No rephrased question found! Call analyze_question before!"

        docs_content = "\n------\n".join(doc.page_content for doc in state["context"])
        prompt = self.response_prompt.invoke({
            "context": docs_content,
            "rephrased_question": state["question"]
        })
        #print("CONTEXT:")
        #print(docs_content)
        print("ANSWER:")
        response = self.llm.invoke(prompt)
        return {"answer": response.content}


    def run(self):
        # Compile graph using memory checkpointer for message history persistance across prompts
        memory = MemorySaver()
        source_graph = StateGraph(self.State).add_sequence([self.analyze_question, self.retrieve, self.generate_response])
        source_graph.add_edge(START, "analyze_question")
        graph = source_graph.compile(checkpointer=memory)

        # main loop which prints output
        config: RunnableConfig = {"configurable": {"thread_id": "1"}}
        question = "What NPC are there"
        while True:
            for message, metadata in graph.stream({"question": question}, config=config, stream_mode="messages"):
                if metadata["langgraph_node"] == "generate_response":   # only print output from the final node
                    print(f"{message.content}", end="", flush=True)
            question = input("\n> ")
            if question == "exit":
                break


if __name__ == "__main__":
    GlBot()