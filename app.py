# gradio app for groq-mistral langgraph demo:
import time
from dotenv import load_dotenv


load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
import gradio as gr
from agent import Agent


SYSTEM_PROMPT = "You are a helpful assistant."


# convert chat history to agent state
def convert_to_agent_state(chat_history):

    agent_states = [SystemMessage(content=SYSTEM_PROMPT)]
    if len(chat_history) > 0:
        for i, _ in enumerate(chat_history):
            print(chat_history[i])
            agent_states.append(HumanMessage(content=chat_history[i][0]))
            agent_states.append(AIMessage(content=chat_history[i][1]))

    return agent_states


def get_streaming_response(
    model_name: str, message_content: str, chat_history: list = []
):

    model = ChatGroq(model=model_name, temperature=0)  # reduce inference cost
    agent = Agent(model, system=SYSTEM_PROMPT)
    # convert chat history to agent state
    agent_states = convert_to_agent_state(chat_history)

    response = agent.graph.invoke(
        {"messages": agent_states + [HumanMessage(content=message_content)]}
    )

    # Append user message to chat history
    chat_history.append([message_content, response["messages"][-1].content])

    # Variable to build the assistant's response incrementally
    assistant_response = ""

    for character in response["messages"][-1].content:
        time.sleep(0.02)  # Adjusted sleep for smoother streaming
        assistant_response += character
        chat_history[-1][1] = assistant_response  # Update the assistant's message
        yield chat_history  # Yield the updated chat history


with gr.Blocks(theme="soft") as demo:
    gr.Markdown(
        value="""
        <h1 align="center">Lang-graph Agent Groq</h1>
        <h4 align="center">Chatbot Demo</h4>
        <h3 align="center">Powered by Langchain and Groq</h3>
        """
    )
    model_name = gr.Dropdown(
        choices=[
            "llama3-70b-8192",
            "llama3-8b-8192",
            "gemma-7b-it",
            "mixtral-8x7b-32768",
        ],
        value="llama3-70b-8192",
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
    # clear.click(lambda: [], None, chatbot, queue=False)  # Clear the chat history
demo.launch()
