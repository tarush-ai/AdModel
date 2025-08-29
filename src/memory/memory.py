import os, json
from src.config import config
import ulid
from typing import List

class ConversationMemory:
    def __init__(self):
        self.numturns = config.NUMTURNS
        self.conversation_id = ulid.new()
        self._update_path()

    def _update_path(self):
        self.path = f"data/memory/conversation/{self.conversation_id}.json"

    def set_conversation_id(self, conversation_id: str):
        self.conversation_id = conversation_id
        self._update_path()

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

    def add_conversation_memory(self, role: str, content: str, ts_s: float, ts_iso: str):
        try:
            with open(self.path, "r") as file:
                convo_memory = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            convo_memory = []
        
        convo_memory.append({"role": role, "content": content, "ts_s": ts_s, "ts_iso": ts_iso})
        
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        with open(self.path, "w") as f:
            json.dump(convo_memory, f, indent=2)
    
    def get_conversation_id(self):
        return self.conversation_id

    def get_messages_for_llm(self):
        convo_memory = self.load_conversation_memory()
        messages = []
        
        for entry in convo_memory:
            if entry["role"] == "User":
                messages.append({"role": "user", "content": entry["content"]})
            elif entry["role"] == "LLM":
                messages.append({"role": "assistant", "content": entry["content"]})
        
        return messages

class ConsumerMemory:
    def __init__(self, db_path: str = "data/memory/consumer.db", provider_id: str = "voyage-3.5-lite"):
        from .topicmemory import ConsumerMemory as TopicMemory, encode_fn
        self.topic_memory = TopicMemory(db_path, provider_id, encode_fn)
        self.topic_memory.init_schema()
        self.conversation_links = {}

    def link_conversation_to_topics(self, conversation_id: str, user_id: str, messages: List[dict]):
        topic_ids = []
        for msg in messages:
            if msg.get("role") == "User":
                topic_id = self.topic_memory.ingest(user_id, f"{conversation_id}_{msg.get('ts_s', 0)}", 
                                                   msg.get("content", ""), msg.get("ts_s"))
                topic_ids.append(topic_id)
        
        self.conversation_links[conversation_id] = {
            "user_id": user_id,
            "topic_ids": topic_ids
        }
        return topic_ids

    def get_user_topic_scores(self, user_id: str):
        return self.topic_memory.scores(user_id)

    def get_conversation_relevance(self, conversation_id: str):
        if conversation_id not in self.conversation_links:
            return {}
        
        user_id = self.conversation_links[conversation_id]["user_id"]
        return self.get_user_topic_scores(user_id)

    def ingest_message(self, user_id: str, message_id: str, text: str, ts: float = None, dwell_seconds: float = None, session_freq: int = None, clicked: int = 0, provided: int = 0):
        return self.topic_memory.ingest(user_id, message_id, text, ts, dwell_seconds, session_freq, clicked, provided)

    def assign_topic(self, user_id: str, text: str):
        return self.topic_memory.assign(user_id, text)

