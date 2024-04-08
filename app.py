"API test for Groq"
import os
from dotenv import load_dotenv

from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs",
        }
    ],
    model="mixtral-8x7b-32768",
)
print(chat_completion.choices[0].message.content)
