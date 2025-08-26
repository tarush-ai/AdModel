import os, json
from src.config import config
import ulid

class ConversationMemory:
    def __init__(self):
        self.numturns = config.NUMTURNS
        self.conversation_id = ulid.new()
        self.path = f"data/memory/conversation/{self.conversation_id}.json"

    def make_memory_file(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as file:
            json.dump([], file)

    def load_conversation_memory(self):
        try:
            with open(self.path, "r") as file:
                convo_memory = json.load(file)
                return convo_memory[-self.numturns:]
        except FileNotFoundError:
            self.make_memory_file()
            return []
        except json.JSONDecodeError:
            return []


    def add_conversation_memory(self, role: str, content: str):
        convo_memory = self.load_conversation_memory() 
        convo_memory.append({role: content})
        
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        with open(self.path, "w") as f:
            json.dump(convo_memory, f, indent=2)
    
    def get_conversation_id(self):
        return self.conversation_id