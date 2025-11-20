from .base import TimeControl


# TODO
class AdaptiveTimeControl(TimeControl):
    def __init__(self, final_time: float, initial_dt: float):
        super().__init__()
        self.final_time = final_time
        self.initial_dt = initial_dt

    def advance(self) -> tuple[bool, float]:
        if self.time < self.final_time:
            dt = self.initial_dt  # Placeholder for adaptive logic
            self.times.append(self.time() + dt)
            self.step += 1
            return True, dt
        else:
            return False, 0.0
