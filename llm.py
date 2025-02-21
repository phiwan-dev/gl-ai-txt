from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are an expert on the documentation of the game Galaxy Life. Be friendly and answer any questions of the user according to your context. You can also tell jokes."),
    ("user", "{input}")
])
llm: OllamaLLM = OllamaLLM(model="llama3.2:1b")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

print(
chain.invoke({"input": "Name 3 units and explain their abilities."})
)

