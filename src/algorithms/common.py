from typing import Sequence, List
from src.config import config
import statistics as stats
import math

class CommonAlgorithm:
    def __init__(self):
        self.epsilon = config.EPSILON
        self.sigmoid_steepness = config.SIGMOID_STEEPNESS

    def simple_z_scaling(self, data: List[float], population: bool = True) -> List[float]:
        if not data:
            return []
        mean = stats.fmean(data)
        sd = stats.pstdev(data) if population else stats.stdev(data)
        if sd <= 0:
            return [0.0 for _ in data]
        return [(num - mean)/(sd + self.epsilon) for num in data]

    def robust_z_scaling(self, population: bool = True) -> List[float]:
        if not data:
            return []
        med = stats.median(data)
        mad = stats.median([abs(num - med) for num in data])
        
        c = 1.4826
        if not population:
            n = len(data)
            if n > 1:
                c *= (1 + 1/(4*n) + 7/(32*n**2))

        denom = c * mad
        if denom <= 0:
            return [0.0 for _ in data]
        return [(num - med)/(denom + self.epsilon) for num in data]

    def sigmoid(self, num: float) -> float:
        k = self.sigmoid_steepness
        if num >= 0:
            zscore = math.exp(-num * k)
            return 1.0 / (1.0 + zscore)
        else:
            zscore = math.exp(num * k)
            return zscore / (1 + zscore)
        
    def normalize_function(self, robust: bool = False) -> List[float]:
        zs = self.robust_z_scaling() if robust else self.simple_z_scaling()
        return [self.sigmoid(z) for z in zs]

    @staticmethod
    def l2_norm(v: List[float]) -> float:
        return math.sqrt(sum(x*x for x in v))

    @staticmethod
    def normalize_vector(v: List[float]) -> List[float]:
        n = CommonAlgorithm.l2_norm(v)
        return v if n <= 1e-12 else [x/n for x in v]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        da, db = CommonAlgorithm.l2_norm(a), CommonAlgorithm.l2_norm(b)
        if da <= 1e-12 or db <= 1e-12:
            return 0.0
        val = sum(x*y for x,y in zip(a,b)) / (da*db)
        return max(min(val, 1.0), -1.0)

    @staticmethod
    def exponential_decay(S: float, last_t: float, now_t: float, half_life_days: float) -> float:
        if last_t <= 0:
            return S
        lam = math.log(2) / max(half_life_days, 1e-6)
        return S * math.exp(-lam * (now_t - last_t) / 86400.0)

    