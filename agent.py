# Class for an agent that can be used in LangChain.
# Core :
import operator
from typing import Annotated, TypedDict, Sequence
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig

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
        self.agent_log_file = open("files/agent_log.txt", "a", encoding="utf-8")
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
        print("TOOL CALLS: ", tool_calls)
        results = []
        if self.tools is not None:
            for t in tool_calls:
                print(f"Calling: {t}")
                # write to agent log file
                self.agent_log_file.write(f"Calling: {t}\n")
                if not t["name"] in self.tools:  # check for bad tool name from LLM
                    print("\n ....bad tool name....")
                    self.agent_log_file.write("\n ....bad tool name....\n")
                    result = "bad tool name, retry"  # instruct LLM to retry if bad
                else:
                    print("\n ....calling tool....")
                    self.agent_log_file.write("\n ....calling tool....\n")
                    self.agent_log_file.write(f"ARGS to tool: -----> {t['args']}\n")
                    print("ARGS to tool: ----->", t["args"])
                    result = self.tools[t["name"]].invoke(t["args"])
                    print("RESULT from tool -------->", result)
                    self.agent_log_file.write(f"RESULT from tool: -----> {result}\n")
                results.append(
                    ToolMessage(
                        tool_call_id=t["id"], name=t["name"], content=str(result)
                    )
                )
            print("Back to the model!")
            self.agent_log_file.write("Back to the model!\n")
        return {"messages": results}
