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

from langchain.tools.retriever import create_retriever_tool
from rag import qdrant_retreiver


# ==================TOOLs section==================
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200, lang="en")


wiki_tool = WikipediaQueryRun(
    name="wiki-tool",
    description="look up things in wikipedia",
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
    return_direct=True,
)


SYSTEM_PROMPT = """You are a smart research assistant that is specialized on solving complex Mechanical Engineering problems. 
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
RETREIVER_PROMPT = """Use the best context from the extracted documents to answer the question.Don't make up answers."""
MODEL_NAMES = [
    "llama3-groq-70b-8192-tool-use-preview",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "gemma-7b-it",
    "mixtral-8x7b-32768",
]


def get_streaming_response(
    model_name_param: str,
    groq_api_key: str,
    search_tools: bool = False,
    uploaded_files: str | list = None,
    message_content: str = "",
    chat_history: list = [],
):
    """
    Generates a streaming response using a chatbot model.

    Args:
        model_name_param (str): The name of the chatbot model to use.
        groq_api_key (str): The API key for the Groq service.
        search_tools (bool, optional): Whether to enable search tools. Defaults to False.
        message_content (str, optional): The content of the initial message. Defaults to "".
        chat_history (list, optional): The history of previous chat messages. Defaults to [].

    Yields:
        list: The updated chat history at each iteration.

    """

    model = ChatGroq(model=model_name_param, api_key=groq_api_key, temperature=0)
    if search_tools == "RAG":

        retriever = qdrant_retreiver(uploaded_files)
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_pdf",
            RETREIVER_PROMPT,
        )

        agent = Agent(model, tools=[retriever_tool], system=SYSTEM_PROMPT)
    elif search_tools == "wiki":
        agent = Agent(model, tools=[wiki_tool], system=SYSTEM_PROMPT)
    else:
        agent = Agent(model, system=SYSTEM_PROMPT)
    agent_states = convert_to_agent_state(SYSTEM_PROMPT, chat_history)

    response = agent.graph.invoke(
        {"messages": agent_states + [HumanMessage(content=message_content)]}
    )

    chat_history.append([message_content, response["messages"][-1].content])

    assistant_response = ""
    for character in response["messages"][-1].content:
        time.sleep(0.01)  # Adjusted sleep for smoother streaming
        assistant_response += character
        chat_history[-1][1] = assistant_response  # Update the assistant's message
        yield chat_history


def draw_agent_graph():
    return gr.Image(
        "graph.png", type="filepath", label="Agent Graph", interactive=False
    )


def upload_files(f):
    # files
    return gr.Files(f)


CSS = """
#warning {background-color: #FFCCCB}
.clear-btn button {color:red !important }"""
with gr.Blocks(theme="default", css=CSS) as demo:
    gr.Markdown(
        value="""
        <h1 align="center">Lang-graph Agent Groq</h1>
        <h4 align="center">Mechanical Chatbot Demo</h4>
        <h3 align="center">Powered by LangGraph,Groq and Wikipedia</h3>
        """
    )
    with gr.Row(variant="compact"):
        groq_api_key = gr.Textbox(
            label="GROQ API Key", interactive=True, placeholder="Enter GROQ_API_KEY"
        )
        model_name = gr.Dropdown(
            choices=MODEL_NAMES,
            value=MODEL_NAMES[0],
            label="Model Name",
            interactive=True,
        )
        agent_tool = gr.Radio(
            choices=["RAG", "WikiPedia", "None"],
            value="None",
            label="Agent Tools",
            interactive=True,
        )
    with gr.Row(variant="panel"):
        with gr.Column():
            files_upload = gr.Files(
                file_count="multiple",
                file_types=[".pdf"],
                height=100,
            )
            terminal = gr.TextArea(
                value="Terminal Output", interactive=True, min_width=500, max_lines=5
            )
            files_upload.upload(
                upload_files, files_upload, files_upload, show_progress="full"
            )

        chatbot = gr.Chatbot(
            show_copy_button=True,
            min_width=900,
            avatar_images=["assets/user.jpg", "assets/bot.jpg"],
        )
    with gr.Row(variant="panel"):
        message = gr.Textbox(min_width=1200, max_lines=1, label="Message")
        submit = gr.Button("Submit", variant="primary", size="sm")
        # give red background to clear button
        clear = gr.ClearButton(elem_classes="clear-btn", variant="stop", size="sm")
        submit.click(
            get_streaming_response,
            inputs=[
                model_name,
                groq_api_key,
                agent_tool,
                files_upload,
                message,
                chatbot,
            ],
            outputs=[chatbot],
            queue=True,
        )
        clear.click(lambda: [], None, chatbot, queue=False)

    with gr.Row():
        generate_graph = gr.Button("Get Agent Graph", variant="secondary")
        image_graph = gr.Image()
        generate_graph.click(draw_agent_graph, None, outputs=[image_graph])
if __name__ == "__main__":
    demo.launch()
