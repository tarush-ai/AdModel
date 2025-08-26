from typing import Sequence, List
from src.config import config
import statistics as stats
import math, json, os

class CommonAlgorithm:
    def __init__(self, data: Sequence[float]):
        self.data: List[float] = list(data) 
        self.epsilon = config.EPSILON
        self.sigmoid_steepness = config.SIGMOID_STEEPNESS

    def simple_z_scaling(self, population: bool = True) -> List[float]:
        mean = stats.fmean(self.data)
        sd = stats.pstdev(self.data) if population else stats.stdev(self.data)
        return [(num - mean)/(sd + self.epsilon) for num in self.data]

    def robust_z_scaling(self, population: bool = True) -> List[float]:
        med = stats.median(self.data)
        mad_calculate = [abs(num - med) for num in self.data]
        mad = stats.median(mad_calculate)
        denom = 1.4826 * mad
        return [(num - med)/(denom + self.epsilon) for num in self.data]

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

