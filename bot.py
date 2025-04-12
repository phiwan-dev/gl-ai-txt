"""
title: Galaxy Life Wiki Bot
author: phiwan-dev
date: 2025-03-24
version: 0.3
license: MIT
description: Galaxy Life chatbot which uses information from the wiki to answer as a RAG LLM
requirements: langchain_core, langchain-community, langchain-ollama, langchain-text-splitters, langgraph, faiss-cpu
"""

import os
from pydantic import BaseModel
from typing import Any, Generator, List
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
from langgraph.graph.state import CompiledStateGraph


class Pipeline:

    class Valves(BaseModel):
        model_name: str
        embeddings_model_name: str

    def __init__(self) -> None:
        self.name = "Galaxy Life Wiki Bot"
        
        self.valves = self.Valves(**{
            "model_name": "qwen2.5:7b",
            "embeddings_model_name": "nomic-embed-text:latest",
        })
        
        self.bot = GlBot(model_name=self.valves.model_name, embeddings_model_name=self.valves.embeddings_model_name)

    async def on_valves_updated(self):
        self.bot = GlBot(model_name=self.valves.model_name, embeddings_model_name=self.valves.embeddings_model_name)

    async def on_startup(self) -> None:
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    def pipe(self, user_message: str, *args: Any, **kwargs: Any) -> Generator[str, None, None]:
        
        print("PIPE EXECUTED")          # the pipeline is executed multiple times per user input
                                        # for things like title generation etc.
        if user_message[:3] == "###":   # implement a hacky workaround to only spend computation
            return                      # cost when user input (not starting with ###) is read.
        
        for message, metadata in self.bot.graph.stream(input={"question": user_message}, config=self.bot.config, stream_mode="messages"):
            if metadata["langgraph_node"] == "generate_response":   # only print output from the final node
                    yield message.content


class GlBot():

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str


    def __init__( self, 
                  model_name: str = "qwen2.5:7b", 
                  embeddings_model_name: str = "nomic-embed-text:latest",
                ) -> None:
        self.MODEL_NAME = model_name
        self.EMBEDDINGS_MODEL_NAME = embeddings_model_name

        self.vector_store = self.load_vector_store()
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

        self.graph: CompiledStateGraph = self.compile()
        self.config: RunnableConfig = {"configurable": {"thread_id": "1"}}


    def load_vector_store(self) -> FAISS:
        # make sure to use the same embeddings model as in the data preparation!
        embeddings: OllamaEmbeddings = OllamaEmbeddings(model=self.EMBEDDINGS_MODEL_NAME)       
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


    def retrieve(self, state: State) -> dict[str, List[Document]]:
        print("\t[RETRIEVE DOCUMENTS]")
        assert "question" in state, "No query/rephrased question was provided. Call analyze_question before!"
        print("QUERY:")
        print(state["question"])

        #retrieved_docs = vector_store.search(state["query"], search_type="mmr")
        retrieved_docs = self.vector_store.similarity_search(state["question"], k=4)
        return {"context": retrieved_docs}


    def generate_response(self, state: State) -> dict[str, str]:
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
        print("ANSWER...")
        response = self.llm.invoke(prompt)
        print("\nANSWER END")
        return {"answer": response.content}


    def compile(self) -> CompiledStateGraph:
        # Compile graph using memory checkpointer for message history persistance across prompts
        memory = MemorySaver()
        source_graph = StateGraph(self.State).add_sequence([self.analyze_question, self.retrieve, self.generate_response])
        source_graph.add_edge(START, "analyze_question")
        return source_graph.compile(checkpointer=memory)


    def cli_loop(self) -> None:
        # main loop which prints output for terminal use
        question = "What NPC are there"     # default question at the start
        while True:
            for message, metadata in self.graph.stream({"question": question}, config=self.config, stream_mode="messages"):
                if metadata["langgraph_node"] == "generate_response":   # only print output from the final node
                    print(f"{message.content}", end="", flush=True)
            question = input("\n> ")
            if question == "exit":
                break


if __name__ == "__main__":
    bot = GlBot()
    bot.cli_loop()