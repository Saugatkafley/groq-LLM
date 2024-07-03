# gradio app for groq-mistral langgraph demo:
import time
from dotenv import load_dotenv

load_dotenv()

from agent import Agent
from utils import convert_to_agent_state
import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

SYSTEM_PROMPT = "You are a helpful assistant."
MODEL_NAMES = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma-7b-it",
    "mixtral-8x7b-32768",
]


def get_streaming_response(
    model_name: str, message_content: str, chat_history: list = []
):
    model = ChatGroq(model=model_name, temperature=0)  # reduce inference cost
    agent = Agent(model, system=SYSTEM_PROMPT)
    agent_states = convert_to_agent_state(chat_history)

    response = agent.graph.invoke(
        {"messages": agent_states + [HumanMessage(content=message_content)]}
    )

    chat_history.append([message_content, response["messages"][-1].content])

    assistant_response = ""
    for character in response["messages"][-1].content:
        time.sleep(0.02)  # Adjusted sleep for smoother streaming
        assistant_response += character
        chat_history[-1][1] = assistant_response  # Update the assistant's message
        yield chat_history


with gr.Blocks(theme="soft") as demo:
    gr.Markdown(
        value="""
        <h1 align="center">Lang-graph Agent Groq</h1>
        <h4 align="center">Chatbot Demo</h4>
        <h3 align="center">Powered by Langchain and Groq</h3>
        """
    )
    model_name = gr.Dropdown(
        choices=MODEL_NAMES,
        value=MODEL_NAMES[0],
        label="Model Name",
        interactive=True,
    )
    chatbot = gr.Chatbot()
    message = gr.Textbox()
    submit = gr.Button("Submit", variant="primary")
    submit.click(
        get_streaming_response,
        inputs=[model_name, message, chatbot],
        outputs=[chatbot],
        # queue=False,
    )
    clear = gr.ClearButton()
    clear.click(lambda: [], None, chatbot, queue=False)
demo.launch()
