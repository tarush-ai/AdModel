import math, statistics as stats
from typing import List, Dict, Optional, Tuple
from src.config import config
from .common import CommonAlgorithm

class SnurtEngine:
    def __init__(self):
        self.h_nurt = config.HALF_LIFE_DAYS
        self.eta = config.ETA
        self.k = config.K
        self.w_ext = 0.30
        self.w_int = 0.70
        self.w12 = 0.50
        self.w13 = 0.30
        self.w14 = 0.20
        self.w15 = 0.60
        self.w16 = 0.30
        self.w17 = 0.10
        self.t_min = 5.0
        self.idle_cap = 180.0

    def dwell_score(self, dwell_seconds: float) -> float:
        if dwell_seconds < self.t_min:
            return 0.0
        d_eff = min(dwell_seconds, self.idle_cap)
        return math.tanh(math.log1p(d_eff) / math.log1p(self.idle_cap))

    def frequency_score(self, session_freq: int) -> float:
        freq_log = math.log1p(session_freq)
        return 1.0 / (1.0 + math.exp(-freq_log))

    def relevance_score(self, lifetime_relevance: float, r_event: float = 1.0, 
                       delta_days: float = 0.0, h_rel: float = 60.0) -> float:
        R = lifetime_relevance * math.exp(-math.log(2) * delta_days / h_rel) + r_event
        return 1.0 / (1.0 + math.exp(-math.log1p(R)))

    def prompt_length_score(self, text_length: int) -> float:
        return 1.0 / (1.0 + math.exp(-math.log1p(text_length)))

    def click_utility_score(self, clicked: int, provided: int) -> float:
        if provided == 0:
            return 0.0
        r_click = clicked / max(provided, 1)
        f_click = min(math.log1p(provided) * r_click, 1.0)
        return 1.0 / (1.0 + math.exp(-f_click))

    def gratitude_score(self, text: str) -> float:
        gratitude_words = ['thank', 'thanks', 'grateful', 'appreciate', 'awesome', 'great', 'excellent']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'stupid', 'wrong', 'useless']
        
        text_lower = text.lower()
        gratitude_count = sum(1 for word in gratitude_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        raw_score = gratitude_count - negative_count
        return max(min(raw_score / 5.0, 1.0), -1.0)

    def verbosity_multiplier(self, text: str) -> float:
        word_count = len(text.split())
        if word_count < 5:
            return 0.8
        elif word_count > 50:
            return 1.2
        else:
            return 0.8 + 0.4 * (word_count - 5) / 45

    def external_factor_scoring(self, dwell_seconds: Optional[float] = None,
                              session_freq: Optional[int] = None,
                              lifetime_relevance: float = 0.0,
                              delta_days: float = 0.0) -> float:
        s_dur = self.dwell_score(dwell_seconds) if dwell_seconds is not None else 0.0
        s_freq = self.frequency_score(session_freq) if session_freq is not None else 0.0
        s_rel = self.relevance_score(lifetime_relevance, delta_days=delta_days)
        
        return self.w12 * s_dur + self.w13 * s_freq + self.w14 * s_rel

    def internal_factor_scoring(self, text: str, clicked: int = 0, provided: int = 0) -> float:
        s_plen = self.prompt_length_score(len(text))
        s_click = self.click_utility_score(clicked, provided)
        s_grateful = self.gratitude_score(text)
        v = self.verbosity_multiplier(text)
        
        base = self.w15 * s_plen + self.w17 * s_grateful
        click_capped = min(self.w16 * s_click, 0.2 * max(base, 1e-6))
        
        return v * (base + click_capped)

    def snurt_reg(self, text: str, dwell_seconds: Optional[float] = None,
                  session_freq: Optional[int] = None, lifetime_relevance: float = 0.0,
                  delta_days: float = 0.0, clicked: int = 0, provided: int = 0) -> float:
        external = self.external_factor_scoring(dwell_seconds, session_freq, 
                                               lifetime_relevance, delta_days)
        internal = self.internal_factor_scoring(text, clicked, provided)
        
        return self.w_ext * external + self.w_int * internal

    def update_topic_score(self, S_prev: float, t_prev: float, t_now: float, 
                          event_score: float) -> float:
        decayed = CommonAlgorithm.exponential_decay(S_prev, t_prev, t_now, self.h_nurt)
        return decayed + self.eta * event_score

    def normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        
        if len(scores) >= 10:
            algo = CommonAlgorithm(scores)
            z_scores = algo.simple_z_scaling()
        else:
            algo = CommonAlgorithm(scores)
            z_scores = algo.robust_z_scaling()
        
        return [self.sigmoid_k(z) for z in z_scores]

    def sigmoid_k(self, z: float) -> float:
        if z >= 0:
            x = math.exp(-self.k * z)
            return 1.0 / (1.0 + x)
        else:
            x = math.exp(self.k * z)
            return x / (1.0 + x)