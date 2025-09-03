import math
from typing import List, Tuple
from common import CommonAlgorithm
from src.deployment.ad import Advertisement as Ad

##Relevance
from sqra import Query_Ad_Relevance_Algorithm           # Sqra
from snurt import User_Topic_Relevance_NoAd_Algorithm   # Snurt
from saurt import User_Topic_Relevance_Ad_Algorithm     # Saurt
from sart import Ad_Topic_Relevance_Algorithm           # Sart
from sqrt import Query_Topic_Relevance_Algorithm        # Sqrt

## Quality
from quality import QualityAlgorithm                    # Quality

## Stake
from stake import StakeAlgorithm                        # Stake

from src.config import config
from scipy.special import softmax

class BaseAlgorithm:
    def __init__(self):
        self.common = CommonAlgorithm()

        #Algorithms
        self.sqra = Query_Ad_Relevance_Algorithm()
        self.snurt = User_Topic_Relevance_NoAd_Algorithm()
        self.saurt = User_Topic_Relevance_Ad_Algorithm()
        self.sart = Ad_Topic_Relevance_Algorithm()
        self.sqrt = Query_Topic_Relevance_Algorithm()
    
        self.stake = StakeAlgorithm()
        
        self.quality = QualityAlgorithm()

        #Weights
        self.w1 = config.w1  # Relevance
        self.w2 = config.w2  # Quality
        self.w3 = config.w3  # Stake
        self.w4 = config.w4  # Surt (User Topic Relevance)
        self.w5 = config.w5  # Sart (Ad Topic Relevance)
        self.w6 = config.w6  # Sqrt (Query Topic Relevance)
        self.w7 = config.w7  # Sqra (Query Ad Relevance)
        self.w8 = config.w8  # Snurt (User Topic Relevance No Ad)
        self.w9 = config.w9  # Saurt (User Topic Relevance Ad)
        
        self.CTR = config.CTR
        self.CVR = config.CVR
        
    def compute(ads: List[Ad], topics: List[Topic], query: query) -> Dict[float: Tuple(Ad, List[Topic])]:
        
        returnval = {}

        stakes = self.stake_score_computation(ads)

        for i, ad in enumerate(ads):
            Relevance, RelTopics  = self.relevance_score_computation(ad, topics, query)
            Relevance *= self.w1
            Quality = self.w2 * self.quality_score_computation()
            Stake = w3 * math.log(1 + stakes[i])
            overall = Relevance + Quality + Stake
            returnval.append(overall: Tuple(ad, RelTopics))
        
        return returnval

    def relevance_score_computation(self, ad: Ad, topics: List[Topic], query: query) -> Tuple(float, List[Topic]):
        Sqra = config.w7 * self.sqra.compute(ad, query) 
        topic_factors = List[float]
        rel_topics = List[Topic]
        for topic in topics:
            Surt = self.w4(self.w8 * snurt.calculate(topic) + self.w9 * self.saurt.calculate(topic))
            Sart = self.w5 * self.sart.calculate(ad, topic)
            Sqrt = self.w6 * self.sqrt.calculate(query, topic)
            if Sqrt >= 0.8:
                rel_topics.append(topic)
            final = Surt + Sart + Sqrt
            topic_factors.append(final)
        αt = softmax(np.array(topic_factors) * self.tau)
        topic_factors_flt = αt * np.sum(topic_factors)
        relevance = Sqra + topic_factors
        return relevance, rel_topics

        
    def stake_score_computation(self, ads: List[Ad]) -> List[float]:
        eCPMs = List[float]
        for ad in ads:
            eCPMs.append(self.compute_eCPM(ad))
        eCPMs = self.common.normalize_function(eCPMs, robust=True)
        stakes = [min(max(stake, 0.05), 0.95) for stake in eCPMs]
        return stakes

    def compute_eCPM(self, ad):
        if ad.CPC is not None:
            eCPM = 1000 * ad.CPC * self.CTR
        elif ad.CPM is not None:
            eCPM = 1000 * ad.CPM
        elif ad.CPA is not None:
            eCPM = 1000 * ad.CPA * self.CVR * self.CTR
        else:
            eCPM = 0.0
        return eCPM

    def quality_score_computation(self) -> List[Ad]:


