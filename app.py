# gradio app for groq-mistral langgraph demo:
import time
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from agent import Agent
from utils import convert_to_agent_state
import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from tools import WikiInputs
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200, lang="en")
tool = WikipediaQueryRun(api_wrapper=api_wrapper)

tool = WikipediaQueryRun(
    name="wiki-tool",
    description="look up things in wikipedia",
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
    return_direct=True,
)

SYSTEM_PROMPT = """You are a smart research assistant that is specialized on solving complex Mechanical Engineering problems. Use the search engine from Wikipedia  to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
MODEL_NAMES = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma-7b-it",
    "mixtral-8x7b-32768",
]


def get_streaming_response(
    model_name_param: str,
    is_search_tools: bool = False,
    message_content: str = "",
    chat_history: list = [],
):
    model = ChatGroq(model=model_name_param, temperature=0)  # reduce inference cost
    if is_search_tools is True:
        agent = Agent(model, tools=[tool], system=SYSTEM_PROMPT)
    else:
        agent = Agent(model, system=SYSTEM_PROMPT)
    agent_states = convert_to_agent_state(SYSTEM_PROMPT, chat_history)

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


def draw_agent_graph():
    return gr.Image(
        "graph.png", type="filepath", label="Agent Graph", interactive=False
    )
    # return Image.open(
    #     Agent(model_name).graph.get_graph().draw_png(output_file_path="graph.png")
    # )


with gr.Blocks(theme="default") as demo:
    gr.Markdown(
        value="""
        <h1 align="center">Lang-graph Agent Groq</h1>
        <h4 align="center">Mechanical Chatbot Demo</h4>
        <h3 align="center">Powered by LangGraph,Groq and Wikipedia</h3>
        """
    )
    with gr.Row():
        model_name = gr.Dropdown(
            choices=MODEL_NAMES,
            value=MODEL_NAMES[0],
            label="Model Name",
            interactive=True,
        )
        is_agent_tools = gr.Radio(
            choices=[True, False],
            value=False,
            label="WikiPedia Tools",
            interactive=True,
        )
    system = gr.Textbox(
        value=SYSTEM_PROMPT, interactive=False, label="System Prompt Message"
    )
    chatbot = gr.Chatbot()
    message = gr.Textbox()
    submit = gr.Button("Submit", variant="primary")
    submit.click(
        get_streaming_response,
        inputs=[model_name, is_agent_tools, message, chatbot],
        outputs=[chatbot],
        queue=True,
    )
    clear = gr.ClearButton()
    clear.click(lambda: [], None, chatbot, queue=False)

    with gr.Row():
        generate_graph = gr.Button("Get Agent Graph", variant="secondary")
        image_graph = gr.Image()
        generate_graph.click(draw_agent_graph, None, outputs=[image_graph])
if __name__ == "__main__":
    demo.launch()
