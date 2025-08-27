import statistics, math
from src.config import config
from typing import List
from src.memory.memory import ConsumerMemory
from common import CommonAlgorithm

class User_Topic_Relevance_NoAd_Algorithm:
    def __init__(self):
        self.h_nurt = config.HNURT
        self.w8 = config.w8
        self.w9 = config.w9
        self.η = config.LEARNING_RATE
        self.memory = ConsumerMemory()

    def set_data(data: List[float], confirm: bool = False)
        if data is not None and not confirm:
            print("Please confirm and add a confirm variable.")
            return
        self.data = data

    def external_factor_scoring(self, topic) -> float:

    def internal_factor_scoring(self, topic) -> float:

    def snurt_reg(self, topic) -> float:
        return w8 * self.external_factor_scoring(topic) + w9 * self.internal_factor_scoring(topic)

    def calculate(topic_id: str, t_k: float, last_updated: float) -> float:
        topic = self.memory.fetch_topic(topic_id)
        
        Δ = (t_k - last_updated) / date
        self.last_updated = t_k
        
        λ = math.log(2.00) / self.h_nurt        
        Snurt_raw *= math.exp(-λ * Δ)
        Snurt_raw += η * self.snurt_reg(topic)

        return Snurt_raw

    



