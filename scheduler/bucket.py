import time
import logging
import itertools
import threading
from typing import Dict, Optional
from base import SchedulingPolicy

logger = logging.getLogger(__name__)


class AdaptiveBucketingSchedulingPolicy(SchedulingPolicy):
    """
    Adaptive bucketing scheduling policy with optimized load balancing
    - Groups requests by sequence length into buckets
    - Schedules similar-length requests to same instances
    - Dynamically splits/merges buckets based on load
    - Bucket-level round-robin: each bucket gets a different instance from global cycler
    - Within bucket: all requests use the same assigned instance (no internal round-robin)
    - Priority scheduling: shorter requests get higher priority within buckets
    """
    
    def __init__(self, l_max: int = 1000, n_max: int = 100, theta: float = 0.5, 
                 max_active_requests_per_instance: int = 3, priority_threshold: float = 1.2):
        self.l_max = l_max
        self.n_max = n_max
        self.theta = theta
        self.max_active_requests_per_instance = max_active_requests_per_instance
        self.priority_threshold = priority_threshold
        # Initialize with single bucket covering [0, L_max)
        self.buckets = [{'low': 0, 'up': l_max, 'requests': {}, 'instance_mapping': {}}]
        self._lock = threading.Lock()
        self.request_bucket_map = {}  # request_id -> bucket_index
        self.instance_active_requests = {}  # instance -> active request count
        logger.info(f"Initialized adaptive bucketing with optimized load balancing: L_max={l_max}, N_max={n_max}, θ={theta}, max_active_per_instance={max_active_requests_per_instance}, priority_threshold={priority_threshold}")
    
    def _get_instance_load(self, instance: str) -> int:
        """Get current active request count for an instance"""
        return self.instance_active_requests.get(instance, 0)
    
    def _update_instance_load(self, instance: str, delta: int):
        """Update instance active request count"""
        current_load = self.instance_active_requests.get(instance, 0)
        new_load = max(0, current_load + delta)
        self.instance_active_requests[instance] = new_load
        logger.debug(f"Instance {instance} load updated: {current_load} -> {new_load}")

    def _get_shortest_pending_request(self, bucket) -> Optional[tuple]:
        """
        Get the shortest pending request from a bucket for priority scheduling
        Returns (request_id, sequence_length) or None if no pending requests
        """
        pending_requests = [
            (req_id, req_data['length']) 
            for req_id, req_data in bucket['requests'].items() 
            if req_data['status'] == 'pending'
        ]
        
        if not pending_requests:
            return None
        
        # Sort by sequence length (shorter first) and return the shortest
        pending_requests.sort(key=lambda x: x[1])
        return pending_requests[0]
    
    def _should_prioritize_request(self, bucket, sequence_length: int) -> bool:
        """
        Check if a request should be prioritized based on its length
        - Returns True if this is one of the shortest pending requests in the bucket
        - This helps prioritize shorter requests for better overall latency
        """
        shortest_pending = self._get_shortest_pending_request(bucket)
        if not shortest_pending:
            return True  # No pending requests, so this one gets priority
        
        # Consider requests within 20% of the shortest length as priority
        shortest_length = shortest_pending[1]
        priority_threshold = shortest_length * self.priority_threshold
        
        return sequence_length <= priority_threshold
    
    def _assign_instance_to_bucket(self, bucket, instance_type: str, cycler: itertools.cycle) -> str:
        """
        Assign an instance to a bucket using bucket-level round-robin
        - Each bucket gets a fresh instance assignment from the global cycler
        - No inheritance from parent buckets
        """
        # Get next instance from global cycler for this bucket
        assigned_instance = next(cycler)
        
        # Store the assignment
        instance_key = f"{instance_type}_instance" if instance_type else "assigned_instance"
        bucket['instance_mapping'][instance_key] = assigned_instance
        
        logger.debug(f"Assigned {instance_type} instance {assigned_instance} to bucket [{bucket['low']}, {bucket['up']})")
        return assigned_instance
    
    def add_request(self, sequence_length: int, request_id: str):
        """Add request to appropriate bucket (Algorithm 1 lines 2-6)"""
        with self._lock:
            for bucket_idx, bucket in enumerate(self.buckets):
                if bucket['low'] <= sequence_length < bucket['up']:
                    bucket['requests'][request_id] = {
                        'id': request_id, 
                        'length': sequence_length,
                        'timestamp': time.time(),
                        'status': 'pending'  # Track request status in bucket
                    }

                    self.request_bucket_map[request_id] = bucket_idx
                    logger.debug(f"Added request {request_id} (length={sequence_length}) to bucket [{bucket['low']}, {bucket['up']})")
                    break
            
            # Check if we need to adjust buckets (Algorithm 1 lines 10-31)
            self._adjust_buckets()
    
    def remove_request(self, request_id: str):
        """Remove completed/failed request from bucket (GC mechanism) - O(1) complexity"""
        with self._lock:
            if request_id in self.request_bucket_map:
                bucket_idx = self.request_bucket_map[request_id]
                if bucket_idx < len(self.buckets):
                    bucket = self.buckets[bucket_idx]
                    if request_id in bucket['requests']:
                        # Update instance load when removing request
                        request_data = bucket['requests'][request_id]
                        if 'assigned_instance' in request_data:
                            self._update_instance_load(request_data['assigned_instance'], -1)
                        
                        del bucket['requests'][request_id]
                    # Remove from mapping
                    del self.request_bucket_map[request_id]
                    logger.debug(f"Removed request {request_id} from bucket {bucket_idx}")
                
                # Check if we need to adjust buckets after removal
                self._adjust_buckets()
            else:
                logger.warning(f"Request {request_id} not found in bucket mapping")
    
    def update_request_status(self, request_id: str, status: str, assigned_instance: str = None):
        """Update request status in bucket"""
        with self._lock:
            for bucket in self.buckets:
                if request_id in bucket['requests']:
                    old_status = bucket['requests'][request_id]['status']
                    bucket['requests'][request_id]['status'] = status
                    
                    if assigned_instance:
                        bucket['requests'][request_id]['assigned_instance'] = assigned_instance
                    
                    # Update instance load when status changes
                    if assigned_instance:
                        # Define active statuses
                        active_statuses = ['pending', 'prefilling', 'decoding']
                        
                        # Special case: when request is first assigned to an instance (from pending to prefilling/decoding)
                        if old_status == 'pending' and status in ['prefilling', 'decoding']:
                            # This is the first time the request is assigned to an instance, increase load
                            self._update_instance_load(assigned_instance, 1)
                            logger.debug(f"Request {request_id} assigned to {assigned_instance}, increased load")
                        
                        # Check if status changed from active to inactive (completed/failed)
                        elif old_status in active_statuses and status not in active_statuses:
                            # Request became inactive, decrease load
                            self._update_instance_load(assigned_instance, -1)
                            logger.debug(f"Request {request_id} became inactive on {assigned_instance}, decreased load")
                    
                    break
    
    def _adjust_buckets(self):
        """Adjust buckets based on current request distribution"""
        # Count only active requests (pending, prefilling, decoding)
        total_active_requests = sum(
            sum(1 for req in b['requests'].values() 
                if req['status'] in ['pending', 'prefilling', 'decoding'])
            for b in self.buckets
        )
        
        # Reset condition: if total active < N_max, merge all buckets into one (Algorithm 1 lines 11-13)
        if total_active_requests < self.n_max:
            self._merge_all_buckets()
            return
        
        # Split condition: if total active >= N_max, check for buckets to split (Algorithm 1 lines 14-30)
        if total_active_requests >= self.n_max:
            self._split_buckets()
    
    def _split_bucket(self, bucket):
        """Split a single bucket into two sub-buckets (Algorithm 1 lines 24-28)"""
        mid = (bucket['low'] + bucket['up']) / 2
        
        # Create left and right buckets - NO inheritance of instance mapping
        b_l = {'low': bucket['low'], 'up': int(mid), 'requests': {}, 'instance_mapping': {}}
        b_r = {'low': int(mid), 'up': bucket['up'], 'requests': {}, 'instance_mapping': {}}
        
        # Partition all requests based on sequence length (including completed/failed for history)
        for request_id, request_data in bucket['requests'].items():
            if request_data['length'] < mid:
                b_l['requests'][request_id] = request_data
            else:
                b_r['requests'][request_id] = request_data
        
        # Find bucket index for updating mapping
        bucket_idx = None
        for idx, b in enumerate(self.buckets):
            if b == bucket:
                bucket_idx = idx
                break
        
        # Update bucket list
        self.buckets.remove(bucket)
        self.buckets.extend([b_l, b_r])
        
        # Update request-to-bucket mapping for requests in the split bucket
        if bucket_idx is not None:
            new_bucket_idx_l = bucket_idx
            new_bucket_idx_r = bucket_idx + 1
            
            # Update mapping for requests in left bucket
            for request_id, request_data in b_l['requests'].items():
                self.request_bucket_map[request_id] = new_bucket_idx_l
            
            # Update mapping for requests in right bucket
            for request_id, request_data in b_r['requests'].items():
                self.request_bucket_map[request_id] = new_bucket_idx_r
            
            # Update mapping for requests in buckets after the split point
            for req_id, old_idx in list(self.request_bucket_map.items()):
                if old_idx > bucket_idx:
                    self.request_bucket_map[req_id] = old_idx + 1
        
        logger.info(f"Split bucket [{bucket['low']}, {bucket['up']}) into "
                   f"[{b_l['low']}, {b_l['up']}) and [{b_r['low']}, {b_r['up']}) - "
                   f"new buckets will get fresh instance assignments")
    
    def _merge_all_buckets(self):
        """Merge all buckets into a single bucket covering the full range"""
        if len(self.buckets) == 1:
            return  # Already a single bucket
        
        all_requests = {}
        for bucket in self.buckets:
            all_requests.update(bucket['requests'])
        
        # Create a single bucket with all requests - fresh instance mapping
        self.buckets = [{'low': 0, 'up': self.l_max, 'requests': all_requests, 'instance_mapping': {}}]
        
        # Update all request mappings to point to bucket 0
        for req_id in self.request_bucket_map:
            self.request_bucket_map[req_id] = 0
        
        logger.info(f"Merged all buckets into single bucket. Total requests: {len(all_requests)}")
    
    def _split_buckets(self):
        """Split buckets that meet the splitting criteria (Algorithm 1 lines 16-22)"""
        split_list = []
        
        for bucket in self.buckets:
            # Count only active requests in this bucket
            active_requests = [req for req in bucket['requests'].values() 
                             if req['status'] in ['pending', 'prefilling', 'decoding']]
            
            if len(active_requests) <= self.n_max:
                continue
            
            # Calculate midpoint
            mid = (bucket['low'] + bucket['up']) / 2
            
            # Count active requests below midpoint
            c_s = sum(1 for req in active_requests if req['length'] < mid)
            
            # Check splitting conditions: C_s / |b.requests| > θ and |b.requests| > N_max
            if (c_s / len(active_requests) > self.theta and 
                len(active_requests) > self.n_max):
                split_list.append(bucket)
        
        # Perform splits (Algorithm 1 lines 23-29)
        for bucket in split_list:
            self._split_bucket(bucket)
    
    def schedule(self, cycler: itertools.cycle, sequence_length: int = None, instance_type: str = None) -> str:
        """
        Schedule request using bucket-aware strategy with priority scheduling
        - Find bucket for sequence length
        - Check if this request should get priority (shorter requests get higher priority)
        - Bucket-level round-robin: each bucket gets a different instance from global cycler
        - Within bucket: all requests use the same assigned instance (no internal round-robin)
        - Fall back to global round-robin if no bucket mapping
        """
        if sequence_length is None:
            return next(cycler)  # Fallback to round-robin
        
        with self._lock:
            # Find appropriate bucket for this sequence length
            for bucket in self.buckets:
                if bucket['low'] <= sequence_length < bucket['up']:
                    # Check if this request should get priority (shorter requests get higher priority)
                    is_priority = self._should_prioritize_request(bucket, sequence_length)
                    
                    # Use instance_type to distinguish between prefill and decode
                    instance_key = f"{instance_type}_instance" if instance_type else "assigned_instance"
                    
                    # If bucket has instance mapping for this type, use it (no internal round-robin)
                    if instance_key in bucket['instance_mapping']:
                        assigned_instance = bucket['instance_mapping'][instance_key]
                        current_load = self._get_instance_load(assigned_instance)
                        
                        # If instance is overloaded, use round-robin for this request
                        if current_load >= self.max_active_requests_per_instance:
                            logger.debug(f"Instance {assigned_instance} overloaded (load={current_load}), using round-robin for request with length {sequence_length}")
                            return next(cycler)
                        else:
                            # Instance is not overloaded, use bucket's assigned instance
                            if is_priority:
                                logger.debug(f"Priority scheduling: shorter request (length={sequence_length}) assigned to {assigned_instance}")
                            return assigned_instance
                    else:
                        # First request for this bucket and type, assign an instance using bucket-level round-robin
                        assigned_instance = self._assign_instance_to_bucket(bucket, instance_type, cycler)
                        if is_priority:
                            logger.debug(f"Priority scheduling: first request for bucket (length={sequence_length}) assigned to {assigned_instance}")
                        return assigned_instance
            
            # Fallback to global round-robin
            return next(cycler)
    
    def get_bucket_statistics(self) -> Dict:
        """Get statistics about current bucket state"""
        with self._lock:
            stats = {
                'total_buckets': len(self.buckets),
                'total_requests': sum(len(b['requests']) for b in self.buckets),
                'bucket_details': [],
                'instance_loads': self.instance_active_requests.copy(),
                'max_active_requests_per_instance': self.max_active_requests_per_instance
            }
            
            for i, bucket in enumerate(self.buckets):
                # Count requests by status
                pending_count = sum(1 for req in bucket['requests'].values() if req['status'] == 'pending')
                prefilling_count = sum(1 for req in bucket['requests'].values() if req['status'] == 'prefilling')
                decoding_count = sum(1 for req in bucket['requests'].values() if req['status'] == 'decoding')
                completed_count = sum(1 for req in bucket['requests'].values() if req['status'] == 'completed')
                failed_count = sum(1 for req in bucket['requests'].values() if req['status'] == 'failed')
                
                # Get shortest pending request for priority info
                shortest_pending = self._get_shortest_pending_request(bucket)
                shortest_length = shortest_pending[1] if shortest_pending else None
                
                # Get instance assignments for this bucket
                instance_assignments = {}
                for key, instance in bucket['instance_mapping'].items():
                    if isinstance(instance, str):
                        instance_assignments[key] = {
                            'instance': instance,
                            'current_load': self._get_instance_load(instance),
                            'is_overloaded': self._get_instance_load(instance) >= self.max_active_requests_per_instance
                        }
                
                stats['bucket_details'].append({
                    'bucket_id': i,
                    'range': f"[{bucket['low']}, {bucket['up']})",
                    'total_requests': len(bucket['requests']),
                    'pending': pending_count,
                    'prefilling': prefilling_count,
                    'decoding': decoding_count,
                    'completed': completed_count,
                    'failed': failed_count,
                    'active_requests': pending_count + prefilling_count + decoding_count,
                    'midpoint': (bucket['low'] + bucket['up']) / 2,
                    'shortest_pending_length': shortest_length,
                    'instance_assignments': instance_assignments
                })
            
            return stats
