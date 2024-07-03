from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# convert chat history to agent state
def convert_to_agent_state(SYSTEM_PROMPT, chat_history):

    agent_states = [SystemMessage(content=SYSTEM_PROMPT)]
    if len(chat_history) > 0:
        for i, _ in enumerate(chat_history):
            print(chat_history[i])
            agent_states.append(HumanMessage(content=chat_history[i][0]))
            agent_states.append(AIMessage(content=chat_history[i][1]))

    return agent_states
