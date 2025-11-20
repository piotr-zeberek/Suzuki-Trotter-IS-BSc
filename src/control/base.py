from abc import ABC, abstractmethod


class TimeControl(ABC):
    def __init__(self):
        self.step: int = 0
        self.times: list[float] = [0.0]

    def time(self) -> float:
        return self.times[self.step]

    @abstractmethod
    def advance(self) -> tuple[bool, float]:
        pass
