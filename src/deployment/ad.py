from typing import List

class Advertisement:
    def __init__(self, cpm: float, cpa: float, cpc: float, industry: str = None, description: str = None):
        self.cpm = cpm
        self.cpa = cpa
        self.cpc = cpc
        self.industry = industry
        self.description = description if description else ""
        self.overall_score = 0.0
        self.relevant_topics = List[str]

    
