from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    EPSILON: float = 1e-8
    SIGMOID_STEEPNESS: float = 1.0
    HNURT: float = 3.0
    TEST_MODEL: str = "gpt-4o"
    REASONING_MODEL: str = "o3"
    NUMTURNS: int = 12     

config = Config()