import openai
from dotenv import load_dotenv
from src.config import config
from src.memory.memory import ConversationMemory
from datetime import datetime, timezone

load_dotenv()

client = openai.OpenAI()

class Chatbot:
    def __init__(self): 
        self.client = client
        self.memory = ConversationMemory()
        self.rate_limit = config.RATE_LIMIT

    def now_utc(self):
        dt = datetime.now(timezone.utc)
        return dt.timestamp(), dt.isoformat()   

    def conversation(self) -> str:
        counter = 0
        while True:
            userinput = input("User: ")
            
            if userinput.strip().lower() == "pause": break

            elif counter >= self.rate_limit:
                print("Rate limit reached. Please restart conversation.")
                break

            elif "convo id: " in userinput.strip().lower(): 
                self.memory.conversation_id = userinput.strip().lower().split("convo id: ")[1]
                print("Conversation ID set to: " + self.memory.conversation_id)
                continue

            ts_s, ts_iso = self.now_utc()
            self.memory.add_conversation_memory("User", userinput, ts_s, ts_iso)

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
            
            ts_s, ts_iso = self.now_utc()
            self.memory.add_conversation_memory("LLM", llm_response, ts_s, ts_iso)
            counter += 1
        return self.memory.conversation_id
        


        
