from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    EPSILON: float = 1e-8
    SIGMOID_STEEPNESS: float = 1.0
    HNURT: float = 3.0

config = Config()