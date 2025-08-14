import itertools
from abc import ABC, abstractmethod

class SchedulingPolicy(ABC):
    """Abstract base class for scheduling policies"""
    
    @classmethod
    @abstractmethod
    def schedule(cls, cycler: itertools.cycle, sequence_length: int = None) -> str:
        raise NotImplementedError("Scheduling policy not implemented")
    