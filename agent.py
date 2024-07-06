# Class for an agent that can be used in LangChain.
# Core :
import operator
from typing import Annotated, TypedDict, Sequence
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph

# langchain
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, BaseMessage


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools=None, system=""):
        """
        Initializes the Agent object with the specified model, tools, and system.

        Parameters:
            model: The model object to be used by the Agent.
            tools: (Optional) Dictionary of tools available to the Agent.
            system: (Optional) The system information for the Agent.
        """
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_groq)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools} if tools is not None else None
        self.model = model
        self.messages = []
        if self.tools is not None:
            self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState) -> bool:
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def call_groq(self, state: AgentState):
        messages = state["messages"]
        if self.system and len(messages) == 1:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        if self.tools is not None:
            for t in tool_calls:
                print(f"Calling: {t}")
                if not t["name"] in self.tools:  # check for bad tool name from LLM
                    print("\n ....bad tool name....")
                    result = "bad tool name, retry"  # instruct LLM to retry if bad
                else:
                    result = self.tools[t["name"]].invoke(t["args"])
                    print(result)
                results.append(
                    ToolMessage(
                        tool_call_id=t["id"], name=t["name"], content=str(result)
                    )
                )
            print("Back to the model!")
        return {"messages": results}


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode


class RAGAgent:

    def __init__(self, model, retriever_tool, system="") -> None:

        # Define a new graph
        self.graph = StateGraph(AgentState)

        # Define the nodes we will cycle between
        graph.add_node("llm", agent)  # agent
        retrieve = ToolNode([retriever_tool])
        graph.add_node("retrieve", retrieve)  # retrieval
        graph.add_node("rewrite", rewrite)  # Re-writing the question
        graph.add_node(
            "generate", generate
        )  # Generating a response after we know the documents are relevant
        # Call agent node to decide to retrieve or not
        graph.add_edge(START, "llm")

        # Decide whether to retrieve
        graph.add_conditional_edges(
            "llm",
            # Assess agent decision
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )

        # Edges taken after the `action` node is called.
        graph.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            grade_documents,
        )
        graph.add_edge("generate", END)
        graph.add_edge("rewrite", "llm")

        # Compile
        graph = graph.compile()
