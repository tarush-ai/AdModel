import statistics, math
from src.config import config

class User_Topic_Relevance_NoAd_Algorithm:
    def __init__(self, data: List[float] = None)
        self.data = data
        self.h_nurt = config.HNURT

    def replace_data(data: List[float], confirm: bool = False)
        if data is not None and not confirm:
            print("Please confirm and add a confirm variable.")
            return
        self.data = data

    def memory_loading():
        memory = 

    def external_factor_scoring(num: float) -> float:

    def internal_factor_scoring(num: float) -> float:

    def calculate() -> List[float]:
        final_scored = List[float] * data.size()
        for num in self.data:
            Snurt_reg = config.w10 * external_factor_scoring(num) + config.w11 * internal_factor_scoring(num)
            λ_nurt = math.log(2.00) / h_nurt
            algorithm = math.pow(λ_nurt)
            
            final_scored.append(algorithm)
        common = CommonAlgorithm(final_scored)
        return common.normalize_function()

