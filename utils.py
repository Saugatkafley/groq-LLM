import os
import re
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


def extract_python_code(text, output_directory="files", filename="python_code.py"):
    # Extract Python code from text
    print("text:", text)
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    print("matches:", matches)

    if matches:
        python_code = matches[0].strip()
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)
        # Define the complete file path
        file_path = os.path.join(output_directory, filename)
        # Write the extracted code to the file
        with open(file_path, "w") as file:
            file.write(python_code)
        print(f"Python code saved to {file_path}")
        return python_code , file_path
    else:
        print("No Python code found")
        return None
