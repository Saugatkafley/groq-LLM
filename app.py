"API test for Groq"
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()
client = ChatGroq(
    temperature=0,
    model="mixtral-8x7b-32768",
)

SYSTEM = "You are a helpful assistant."
HUMAN = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])

chain = prompt | client
answer = chain.invoke({"text": "Explain the importance of low latency for LLMs."})
print(answer)
