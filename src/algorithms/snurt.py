from typing import List, Dict, Optional
from src.memory.topicmemory import ConsumerMemory
from .snurt_engine import SnurtEngine

class User_Topic_Relevance_NoAd_Algorithm:
    def __init__(self, db_path: str = "data/memory/topics.db", provider_id: str = "voyage-3.5-lite"):
        from src.memory.topicmemory import encode_fn
        self.memory = ConsumerMemory(db_path, provider_id, encode_fn)
        self.memory.init_schema()
        self.engine = SnurtEngine()

    def update_topic_score(self, user_id: str, text: str, 
                          dwell_seconds: Optional[float] = None,
                          session_freq: Optional[int] = None,
                          clicked: int = 0, provided: int = 0,
                          ts: Optional[float] = None) -> int:
        msg_id = f"msg_{int(ts or self.memory._now())}"
        topic_id = self.memory.ingest(user_id, msg_id, text, ts, dwell_seconds, session_freq, clicked, provided)
        return topic_id

    def get_user_topic_scores(self, user_id: str) -> Dict[str, float]:
        return self.memory.scores(user_id)

    def get_top_topics(self, user_id: str, limit: int = 5) -> List[tuple]:
        scores = self.get_user_topic_scores(user_id)
        sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_topics[:limit]

