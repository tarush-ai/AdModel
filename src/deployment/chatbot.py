import json, openai
from dotenv import load_dotenv
from typing import Tuple
from src.config import config
from src.memory.memory import ConversationMemory
#from algorithm import basealg as algs

load_dotenv()

client = openai.OpenAI()

class Chatbot:
    def __init__(self): 
        self.client = client
        self.memory = ConversationMemory()

    def conversation(self):
        userinput = input("User: ")
        
        if userinput == "pause":
            return

        self.memory.add_conversation_memory("User", userinput)
        loaded_mem = self.memory.load_conversation_memory()

        messages = [
            {
                "role": "system",
                "content": str(loaded_mem) if loaded_mem else ""
            },
            {
                "role": "user",
                "content": userinput
            }
        ]

        response = self.client.chat.completions.create(
            model = config.TEST_MODEL,
            messages = messages
        )

        llm_response = response.choices[0].message.content
        print("Assistant: " +llm_response)
        self.memory.add_conversation_memory("LLM", llm_response)
        return self.conversation()
    


        
