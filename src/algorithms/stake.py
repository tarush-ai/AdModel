from typing import List
from src.deployment.ad import Advertisement as Ad
from common import CommonAlgorithm
from src.config import config

class StakeAlgorithm:
    def __init__(self):
        self.CTR = config.CTR
        self.CVR = config.CVR
        self.common = CommonAlgorithm
    
    def compute_stake_scores(self, ads: List[Ad]) -> List[Ad]:
        


    def compute_eCPM(self, ad: Ad) -> float:
        

