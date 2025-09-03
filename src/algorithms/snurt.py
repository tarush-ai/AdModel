# src/algorithms/snurt.py
import math
from typing import Dict, List, Optional, Tuple

from src.config import config
from .common import CommonAlgorithm


class SnurtEngine:
    def __init__(self) -> None:
        self.common = CommonAlgorithm()

        self.h_nurt_days = config.SNURT_HALF_LIFE_DAYS
        self.eta = config.SNURT_ETA

        self.w_ext = config.SNURT_W_EXT
        self.w_int = config.SNURT_W_INT

        self.w12 = config.SNURT_W12_DUR
        self.w13 = config.SNURT_W13_FREQ
        self.w14 = config.SNURT_W14_REL

        self.w15 = config.SNURT_W15_PLEN
        self.w16 = config.SNURT_W16_CLICK
        self.w17 = config.SNURT_W17_GRAT

        self.dwell_min_s = config.SNURT_T_MIN_SECONDS
        self.dwell_idle_cap_s = config.SNURT_IDLE_CAP_SECONDS

        self.h_rel_days = config.SNURT_REL_HALF_LIFE_DAYS
        self.r_event = config.SNURT_R_EVENT

        self.click_cap_frac = config.SNURT_CLICK_CAP_FRAC
        self.verb_lo = config.SNURT_VERBOSITY_MIN
        self.verb_hi = config.SNURT_VERBOSITY_MAX
        self.verb_low_words: int = config.SNURT_VERBOSITY_LOW_WORDS
        self.verb_high_words: int = config.SNURT_VERBOSITY_HIGH_WORDS

    def _sigmoid(self, x) -> float:
        return self.common.sigmoid(x)

    def dwell_score(self, dwell_seconds: Optional[float]) -> float:
        if dwell_seconds is None or dwell_seconds < self.dwell_min_s:
            return 0.0
        d_eff = min(dwell_seconds, self.dwell_idle_cap_s)
        r = math.log1p(d_eff) / max(math.log1p(self.dwell_idle_cap_s), 1e-9)
        return self._sigmoid((r * 6.0 - 3.0))

    def frequency_score(self, session_freq: Optional[int]) -> float:
        if session_freq is None or session_freq <= 0:
            return 0.0
        return self._sigmoid(math.log1p(session_freq))

    def relevance_score(self, lifetime_relevance, delta_days) -> float:
        prior = lifetime_relevance * math.exp(-math.log(2.0) * delta_days / max(self.h_rel_days, 1e-6))
        r = prior + self.r_event
        return self._sigmoid(math.log1p(max(r, 0.0)))

    def prompt_length_score(self, text: str) -> float:
        return self._sigmoid(math.log1p(len(text.split())))

    def click_utility_score(self, clicked: int, provided: int) -> float:
        if provided <= 0:
            return 0.0
        r_click = clicked / max(provided, 1)
        f_click = min(math.log1p(provided) * r_click, 1.0)
        return self._sigmoid(f_click)

    def gratitude_score(self, text: str) -> float:
        g_words = ("thank", "thanks", "grateful", "appreciate", "awesome", "great", "excellent")
        n_words = ("bad", "terrible", "awful", "hate", "stupid", "wrong", "useless")
        tl = text.lower()
        score = sum(w in tl for w in g_words) - sum(w in tl for w in n_words)
        score = max(min(score / 5.0, 1.0), -1.0)
        return self._sigmoid(score)

    def verbosity_multiplier(self, text: str) -> float:
        words = len(text.split())
        if words <= self.verb_low_words:
            return self.verb_lo
        if words >= self.verb_high_words:
            return self.verb_hi
        span = self.verb_high_words - self.verb_low_words
        return self.verb_lo + (self.verb_hi - self.verb_lo) * ((words - self.verb_low_words) / max(span, 1))

    def event_score(
        self,
        *,
        text: str,
        dwell_seconds: Optional[float] = None,
        session_freq: Optional[int] = None,
        lifetime_relevance = 0.0,
        delta_days = 0.0,
        clicked: int = 0,
        provided: int = 0,
    ) -> Tuple[float, Dict[str, float]]:
        s_dur = self.dwell_score(dwell_seconds)
        s_freq = self.frequency_score(session_freq)
        s_rel = self.relevance_score(lifetime_relevance, delta_days)
        s_plen = self.prompt_length_score(text)
        s_click = self.click_utility_score(clicked, provided)
        s_grat = self.gratitude_score(text)
        v = self.verbosity_multiplier(text)

        external = self.w12 * s_dur + self.w13 * s_freq + self.w14 * s_rel
        base_internal = self.w15 * s_plen + self.w17 * s_grat
        click_capped = min(self.w16 * s_click, self.click_cap_frac * max(base_internal, 1e-6))
        internal = v * (base_internal + click_capped)

        sconv = self.w_ext * external + self.w_int * internal
        sconv = max(min(sconv, 1.0), 0.0)

        parts = {
            "s_dur": s_dur,
            "s_freq": s_freq,
            "s_rel": s_rel,
            "s_plen": s_plen,
            "s_click": s_click,
            "s_grat": s_grat,
            "v_mult": v,
            "external": external,
            "internal": internal,
            "sconv": sconv,
        }
        return sconv, parts

    def update_raw(
        self,
        *,
        s_prev_raw,
        last_updated_ts,
        event_ts,
        sconv,
    ) -> float:
        decayed = CommonAlgorithm.exponential_decay(
            S=s_prev_raw,
            last_t=last_updated_ts,
            now_t=event_ts,
            half_life_days=self.h_nurt_days,
        )
        return decayed + self.eta * sconv

    def display_scores(
        self,
        *,
        now_ts,
        topic_raw: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        topics = list(topic_raw.keys())
        decayed_vals: List[float] = []
        for t in topics:
            s_raw, last_ts = topic_raw[t]
            decayed = CommonAlgorithm.exponential_decay(
                S=s_raw,
                last_t=last_ts,
                now_t=now_ts,
                half_life_days=self.h_nurt_days,
            )
            decayed_vals.append(decayed)
        if not decayed_vals:
            return {}
        z = self.common.simple_z_scaling(decayed_vals, population=True)
        mapped = [self.common.sigmoid(zi) for zi in z]
        return {topic: mapped[i] for i, topic in enumerate(topics)}


class User_Topic_Relevance_NoAd_Algorithm:
    def __init__(self) -> None:
        self.engine = SnurtEngine()

    def calculate(
        self,
        *,
        text: str,
        dwell_seconds: Optional[float] = None,
        session_freq: Optional[int] = None,
        lifetime_relevance = 0.0,
        delta_days = 0.0,
        clicked: int = 0,
        provided: int = 0,
    ) -> float:
        sconv, _ = self.engine.event_score(
            text=text,
            dwell_seconds=dwell_seconds,
            session_freq=session_freq,
            lifetime_relevance=lifetime_relevance,
            delta_days=delta_days,
            clicked=clicked,
            provided=provided,
        )
        return sconv

    def insight(
        self,
        *,
        text: str,
        dwell_seconds: Optional[float] = None,
        session_freq: Optional[int] = None,
        lifetime_relevance = 0.0,
        delta_days = 0.0,
        clicked: int = 0,
        provided: int = 0,
    ) -> Dict[str, float]:
        _, parts = self.engine.event_score(
            text=text,
            dwell_seconds=dwell_seconds,
            session_freq=session_freq,
            lifetime_relevance=lifetime_relevance,
            delta_days=delta_days,
            clicked=clicked,
            provided=provided,
        )
        return {
            "Sconv": round(parts["sconv"], 3),
            "External": round(parts["external"], 3),
            "Internal": round(parts["internal"], 3),
            "s_dur": round(parts["s_dur"], 3),
            "s_freq": round(parts["s_freq"], 3),
            "s_rel": round(parts["s_rel"], 3),
            "s_plen": round(parts["s_plen"], 3),
            "s_click": round(parts["s_click"], 3),
            "s_grat": round(parts["s_grat"], 3),
            "v_mult": round(parts["v_mult"], 3),
        }

    def update_raw(
        self,
        *,
        s_prev_raw,
        last_updated_ts,
        event_ts,
        sconv,
    ) -> float:
        return self.engine.update_raw(
            s_prev_raw=s_prev_raw,
            last_updated_ts=last_updated_ts,
            event_ts=event_ts,
            sconv=sconv,
        )

    def display_scores(
        self,
        *,
        now_ts,
        topic_raw: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        return self.engine.display_scores(now_ts=now_ts, topic_raw=topic_raw)
