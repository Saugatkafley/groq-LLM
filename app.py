# gradio app for groq-mistral langgraph demo:
from dotenv import load_dotenv

load_dotenv()

from agent import Agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import gradio as gr

MODEL_NAME = "mixtral-8x7b-32768"
SYSTEM_PROMT = "You are a helpful assistant."
model = ChatGroq(model=MODEL_NAME, temperature=0)  # reduce inference cost
agent = Agent(model, system=SYSTEM_PROMT)
import time


def get_response(message, history):
    return "", history + [[message, None]]


def get_streaming_response(message_content, chat_history):
    response = agent.graph.invoke(
        {"messages": [HumanMessage(content=message_content)]}
    )["messages"][-1].content
    # Append user message to chat history
    chat_history.append([message_content, None])

    # Variable to build the assistant's response incrementally
    assistant_response = ""

    for character in response:
        time.sleep(0.02)  # Adjusted sleep for smoother streaming
        assistant_response += character
        chat_history[-1][1] = assistant_response  # Update the assistant's message
        yield chat_history  # Yield the updated chat history


with gr.Blocks(theme="soft") as demo:
    chatbot = gr.Chatbot()
    message = gr.Textbox()

    submit = gr.Button("Submit", variant="primary")
    submit.click(
        get_streaming_response,
        inputs=[message, chatbot],
        outputs=[chatbot],
        # queue=False,
    )
    clear = gr.Button("Clear")
    clear.click(lambda: [], None, chatbot, queue=False)  # Clear the chat history
demo.launch()
