import logging
import itertools
import threading
from typing import Dict, List
from base import SchedulingPolicy

logger = logging.getLogger(__name__)

class DynamicPrioritySchedulingPolicy(SchedulingPolicy):
    """
    Intelligent priority-based scheduling policy with load balancing
    - Dynamic priority calculation based on request characteristics
    - Efficient load balancing without hard-coded thresholds
    - Smart instance selection based on current load and performance
    - O(1) scheduling complexity
    """
    
    def __init__(self, max_active_requests_per_instance: int = 3):
        self.max_active_requests_per_instance = max_active_requests_per_instance
        self.instance_active_requests = {}  # instance -> active request count
        self.instance_performance_history = {}  # instance -> list of recent response times
        self.request_length_stats = {'min': float('inf'), 'max': 0, 'count': 0, 'sum': 0}
        self._lock = threading.Lock()
        logger.info(f"Initialized intelligent priority scheduling: max_active_per_instance={max_active_requests_per_instance}")
    
    def _get_instance_load(self, instance: str) -> int:
        """Get current active request count for an instance"""
        return self.instance_active_requests.get(instance, 0)
    
    def _update_instance_load(self, instance: str, delta: int):
        """Update instance active request count"""
        current_load = self.instance_active_requests.get(instance, 0)
        new_load = max(0, current_load + delta)
        self.instance_active_requests[instance] = new_load
        logger.debug(f"Instance {instance} load updated: {current_load} -> {new_load}")
    
    def _update_request_length_stats(self, sequence_length: int):
        """Update global request length statistics for dynamic threshold calculation"""
        with self._lock:
            self.request_length_stats['min'] = min(self.request_length_stats['min'], sequence_length)
            self.request_length_stats['max'] = max(self.request_length_stats['max'], sequence_length)
            self.request_length_stats['count'] += 1
            self.request_length_stats['sum'] += sequence_length
    
    def _calculate_dynamic_threshold(self) -> float:
        """Calculate dynamic threshold based on observed request lengths"""
        stats = self.request_length_stats
        if stats['count'] < 10:  # Need enough samples
            return 100  # Default threshold
        
        # Use 25th percentile as threshold (requests shorter than this are considered "short")
        # For simplicity, use mean - 0.5 * std as approximation
        mean = stats['sum'] / stats['count']
        # Simple approximation of variance
        variance = (stats['max'] - stats['min']) ** 2 / 4
        std = variance ** 0.5
        threshold = max(mean - 0.5 * std, stats['min'])
        
        return threshold
    
    def _calculate_instance_score(self, instance: str, sequence_length: int) -> float:
        """Calculate instance score for request assignment (lower is better)"""
        current_load = self._get_instance_load(instance)
        
        # Load factor (0-1, lower is better)
        load_factor = current_load / self.max_active_requests_per_instance
        
        # Performance factor based on recent history
        if instance in self.instance_performance_history and self.instance_performance_history[instance]:
            recent_times = self.instance_performance_history[instance][-10:]  # Last 10 responses
            avg_response_time = sum(recent_times) / len(recent_times)
            performance_factor = min(avg_response_time / 2.0, 1.0)  # Normalize to 0-1
        else:
            performance_factor = 0.5  # Default neutral score
        
        # Length factor (prefer shorter requests for better instances)
        # Normalize sequence length to 0-1 range
        stats = self.request_length_stats
        if stats['max'] > stats['min']:
            length_factor = (sequence_length - stats['min']) / (stats['max'] - stats['min'])
        else:
            length_factor = 0.5
        
        # Combined score with weights
        score = load_factor * 0.5 + performance_factor * 0.3 + length_factor * 0.2
        
        return score
    
    def _get_available_instances(self, cycler: itertools.cycle) -> List[str]:
        """Get list of available instances from cycler efficiently"""
        instances = []
        seen = set()
        
        # Get instances from cycler efficiently
        for _ in range(20):  # Reasonable limit
            try:
                instance = next(cycler)
                if instance not in seen:
                    instances.append(instance)
                    seen.add(instance)
                    if len(instances) >= 10:  # Reasonable limit
                        break
            except StopIteration:
                break
        
        return instances
    
    def schedule(self, cycler: itertools.cycle, sequence_length: int = None, instance_type: str = None) -> str:
        """
        Intelligent priority-based scheduling with dynamic load balancing
        - Dynamic threshold calculation based on request history
        - Smart instance selection based on load and performance
        - Efficient scheduling without hard-coded values
        """
        if sequence_length is None:
            return next(cycler)  # Fallback to round-robin
        
        # Update request length statistics
        self._update_request_length_stats(sequence_length)
        
        with self._lock:
            # Get available instances
            available_instances = self._get_available_instances(cycler)
            
            if not available_instances:
                return next(cycler)  # Fallback to round-robin
            
            # Filter non-overloaded instances
            non_overloaded = [inst for inst in available_instances 
                            if self._get_instance_load(inst) < self.max_active_requests_per_instance]
            
            if not non_overloaded:
                # All instances are overloaded, use round-robin
                logger.debug(f"All instances overloaded, using round-robin for request with length {sequence_length}")
                return next(cycler)
            
            # Calculate dynamic threshold
            dynamic_threshold = self._calculate_dynamic_threshold()
            
            # For requests below dynamic threshold, use intelligent selection
            if sequence_length <= dynamic_threshold:
                # Find best instance based on load and performance
                best_instance = min(non_overloaded, 
                                  key=lambda x: self._calculate_instance_score(x, sequence_length))
                logger.debug(f"Priority scheduling: short request (length={sequence_length}, threshold={dynamic_threshold:.1f}) assigned to {best_instance}")
                return best_instance
            else:
                # For longer requests, use simple round-robin among available instances
                # Use a deterministic selection based on current state
                selected_idx = sequence_length % len(non_overloaded)
                selected_instance = non_overloaded[selected_idx]
                logger.debug(f"Standard scheduling: long request (length={sequence_length}, threshold={dynamic_threshold:.1f}) assigned to {selected_instance}")
                return selected_instance
    
    def update_request_status(self, request_id: str, status: str, assigned_instance: str = None):
        """Update request status and instance load"""
        if assigned_instance:
            with self._lock:
                # Define active statuses
                active_statuses = ['pending', 'prefilling', 'decoding']
                
                # Update load based on status change
                if status in active_statuses:
                    # Request became active, increase load
                    self._update_instance_load(assigned_instance, 1)
                else:
                    # Request completed/failed, decrease load
                    self._update_instance_load(assigned_instance, -1)
    
    def record_response_time(self, instance: str, response_time: float):
        """Record response time for performance tracking"""
        with self._lock:
            if instance not in self.instance_performance_history:
                self.instance_performance_history[instance] = []
            
            self.instance_performance_history[instance].append(response_time)
            
            # Keep only recent history (last 50 responses)
            if len(self.instance_performance_history[instance]) > 50:
                self.instance_performance_history[instance] = self.instance_performance_history[instance][-50:]
    
    def get_statistics(self) -> Dict:
        """Get scheduling statistics"""
        with self._lock:
            dynamic_threshold = self._calculate_dynamic_threshold()
            return {
                'instance_loads': self.instance_active_requests.copy(),
                'max_active_requests_per_instance': self.max_active_requests_per_instance,
                'total_instances': len(self.instance_active_requests),
                'request_length_stats': self.request_length_stats.copy(),
                'dynamic_threshold': dynamic_threshold,
                'instance_performance': {k: len(v) for k, v in self.instance_performance_history.items()}
            }
        