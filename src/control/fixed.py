from .base import TimeControl

class FixedTimeControl(TimeControl):
    def __init__(self, times: list[float]):
        super().__init__()
        self.times = times
        self.num_steps = len(times) - 1
        
    def advance(self) -> tuple[bool, float]:
        if self.step < self.num_steps:
            self.step += 1
            dt = self.times[self.step] - self.times[self.step - 1]
            return True, dt
        else:
            return False, 0.0