from abc import ABC, abstractmethod

from ..definition import Problem


class Solver(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem

    @abstractmethod
    def solve(self):
        pass
