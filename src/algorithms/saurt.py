from typing import List, Dict, Optional
from src.memory.topicmemory import ConsumerMemory

class User_Topic_Relevance_Ad_Algorithm:
    def __init__(self, db_path: str = "data/memory/topics.db", provider_id: str = "voyage-3.5-lite"):
        from src.memory.topicmemory import encode_fn
        self.memory = ConsumerMemory(db_path, provider_id, encode_fn)
        self.memory.init_schema()

    def update_topic_score(self, user_id: str, text: str, 
                          dwell_seconds: Optional[float] = None,
                          session_freq: Optional[int] = None,
                          clicked: int = 0, provided: int = 0,