from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    EPSILON: float = 1e-8
    SIGMOID_STEEPNESS: float = 1.0
    HNURT: float = 3.0
    TEST_MODEL: str = "gpt-4o"
    REASONING_MODEL: str = "o3"
    NUMTURNS: int = 12     
    RATE_LIMIT: int = 10
    
    HALF_LIFE_DAYS: float = 21.0
    TAU_ABS: float = 0.80
    DELTA: float = 0.05
    ETA: float = 1.0
    K: float = 1.4
    LEARNING_RATE: float = 1.0
    w8: float = 0.6
    w9: float = 0.4


config = Config()