from dataclasses import dataclass


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    total: int
    correct: int
