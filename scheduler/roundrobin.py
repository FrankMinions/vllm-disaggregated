import itertools
from base import SchedulingPolicy


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    """Round-robin scheduling policy (like dis_demo.py)"""
    
    def schedule(self, cycler: itertools.cycle, sequence_length: int = None, instance_type: str = None) -> str:
        return next(cycler)
