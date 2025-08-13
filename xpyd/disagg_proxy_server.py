#!/usr/bin/env python3
"""
xPyD Disaggregated Proxy Server with Adaptive Bucketing

This implementation combines:
1. PD separation: prefill instances with max_tokens=1, decode instances handle full generation
2. Adaptive bucketing: schedule similar-length requests to same instances
"""

import argparse
import json
import logging
import os
import sys
import time
import threading
import itertools
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import aiohttp
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


class RequestStatus(Enum):
    PENDING = "pending"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    FAILED = "failed"


class SchedulingPolicy(ABC):
    """Abstract base class for scheduling policies"""
    
    @abstractmethod
    def schedule(self, cycler: itertools.cycle, sequence_length: int = None) -> str:
        raise NotImplementedError("Scheduling policy not implemented")


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    """Round-robin scheduling policy (like dis_demo.py)"""
    
    def schedule(self, cycler: itertools.cycle, sequence_length: int = None, instance_type: str = None) -> str:
        return next(cycler)


class AdaptiveBucketingSchedulingPolicy(SchedulingPolicy):
    """
    Adaptive bucketing scheduling policy with simple load balancing
    - Groups requests by sequence length into buckets
    - Schedules similar-length requests to same instances
    - Dynamically splits/merges buckets based on load
    - Simple load balancing: if instance active requests > threshold, use round-robin
    """
    
    def __init__(self, l_max: int = 1000, n_max: int = 100, theta: float = 0.5, 
                 max_active_requests_per_instance: int = 3):
        self.l_max = l_max
        self.n_max = n_max
        self.theta = theta
        self.max_active_requests_per_instance = max_active_requests_per_instance
        # Initialize with single bucket covering [0, L_max)
        self.buckets = [{'low': 0, 'up': l_max, 'requests': {}, 'instance_mapping': {}}]
        self._lock = threading.Lock()
        self.request_bucket_map = {}  # request_id -> bucket_index
        self.instance_active_requests = {}  # instance -> active request count
        logger.info(f"Initialized adaptive bucketing with load balancing: L_max={l_max}, N_max={n_max}, θ={theta}, max_active_per_instance={max_active_requests_per_instance}")
    
    def _get_instance_load(self, instance: str) -> int:
        """Get current active request count for an instance"""
        return self.instance_active_requests.get(instance, 0)
    
    def _update_instance_load(self, instance: str, delta: int):
        """Update instance active request count"""
        current_load = self.instance_active_requests.get(instance, 0)
        new_load = max(0, current_load + delta)
        self.instance_active_requests[instance] = new_load
        logger.debug(f"Instance {instance} load updated: {current_load} -> {new_load}")
    
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
        
        # Create left and right buckets
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
                   f"[{b_l['low']}, {b_l['up']}) and [{b_r['low']}, {b_r['up']})")
    
    def _merge_all_buckets(self):
        """Merge all buckets into a single bucket covering the full range"""
        if len(self.buckets) == 1:
            return  # Already a single bucket
        
        all_requests = {}
        for bucket in self.buckets:
            all_requests.update(bucket['requests'])
        
        # Create a single bucket with all requests
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
        Schedule request using bucket-aware strategy with simple load balancing
        - Find bucket for sequence length
        - If bucket's assigned instance is overloaded (> threshold), use round-robin
        - Otherwise, use consistent instance assignment per bucket
        - Fall back to global round-robin if no bucket mapping
        """
        if sequence_length is None:
            return next(cycler)  # Fallback to round-robin
        
        with self._lock:
            # Find appropriate bucket for this sequence length
            for bucket in self.buckets:
                if bucket['low'] <= sequence_length < bucket['up']:
                    # Use instance_type to distinguish between prefill and decode
                    instance_key = f"{instance_type}_instance" if instance_type else "assigned_instance"
                    
                    # If bucket has instance mapping for this type, check if it's overloaded
                    if instance_key in bucket['instance_mapping']:
                        assigned_instance = bucket['instance_mapping'][instance_key]
                        current_load = self._get_instance_load(assigned_instance)
                        
                        # If instance is overloaded, use round-robin for this request
                        if current_load >= self.max_active_requests_per_instance:
                            logger.debug(f"Instance {assigned_instance} overloaded (load={current_load}), using round-robin for request with length {sequence_length}")
                            return next(cycler)
                        else:
                            # Instance is not overloaded, use consistent assignment
                            return assigned_instance
                    else:
                        # First request for this bucket and type, assign an instance
                        assigned_instance = next(cycler)
                        bucket['instance_mapping'][instance_key] = assigned_instance
                        logger.debug(f"Assigned {instance_type} instance {assigned_instance} to bucket [{bucket['low']}, {bucket['up']}) for length {sequence_length}")
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
                    'instance_assignments': instance_assignments
                })
            
            return stats


@dataclass
class RequestInfo:
    """Request information for tracking"""
    request_id: str
    sequence_length: int
    status: RequestStatus
    prefill_instance: Optional[str] = None
    decode_instance: Optional[str] = None
    start_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0


class xPyDProxy:
    """
    xPyD Disaggregated Proxy with adaptive bucketing
    - PD separation: prefill with max_tokens=1, decode handles full generation
    - Adaptive bucketing: schedule similar-length requests to same instances
    """
    
    def __init__(
        self,
        prefill_instances: List[str],
        decode_instances: List[str],
        model: str,
        scheduling_policy: SchedulingPolicy,
    ):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances)
        self.model = model
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.scheduling_policy = scheduling_policy
        
        # Request tracking
        self.request_history: Dict[str, RequestInfo] = {}
        self._lock = threading.Lock()
        
        # Setup router
        self.router = APIRouter()
        self.setup_routes()
        
        logger.info(f"Initialized xPyD proxy with {len(prefill_instances)} prefill and {len(decode_instances)} decode instances")
    
    def _update_request_status_sync(self, request_id: str, status: RequestStatus):
        """Synchronous version of status update for background worker"""
        with self._lock:
            if request_id in self.request_history:
                req_info = self.request_history[request_id]
                req_info.status = status
                if status == RequestStatus.PREFILLING:
                    req_info.prefill_time = time.time()
                elif status == RequestStatus.DECODING:
                    req_info.decode_time = time.time()
                
                # Update status in bucket if using adaptive scheduling
                if isinstance(self.scheduling_policy, AdaptiveBucketingSchedulingPolicy):
                    # For completed/failed requests, we need to pass the assigned instance
                    assigned_instance = None
                    if status == RequestStatus.COMPLETED or status == RequestStatus.FAILED:
                        # Try to get the assigned instance from request history
                        if req_info.prefill_instance:
                            assigned_instance = req_info.prefill_instance
                        elif req_info.decode_instance:
                            assigned_instance = req_info.decode_instance
                    
                    self.scheduling_policy.update_request_status(request_id, status.value, assigned_instance)
                
                # Remove from bucket if request is completed or failed
                if status in [RequestStatus.COMPLETED, RequestStatus.FAILED]:
                    if isinstance(self.scheduling_policy, AdaptiveBucketingSchedulingPolicy):
                        self.scheduling_policy.remove_request(request_id)
                        logger.debug(f"Removed {status.value} request {request_id} from buckets")
    
    def setup_routes(self):
        """Setup API routes"""
        self.router.post("/v1/completions")(self.create_completion)
        self.router.post("/v1/chat/completions")(self.create_chat_completion)
        self.router.get("/status", response_class=JSONResponse)(self.get_status)
        self.router.get("/buckets", response_class=JSONResponse)(self.get_bucket_status)
    
    async def forward_request(self, url: str, data: Dict[str, Any], use_chunked: bool = True):
        """Forward request to instance"""
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            try:
                async with session.post(url=url, json=data, headers=headers) as response:
                    if 200 <= response.status < 300 or 400 <= response.status < 500:
                        if use_chunked:
                            async for chunk_bytes in response.content.iter_chunked(1024):
                                yield chunk_bytes
                        else:
                            content = await response.read()
                            yield content
                    else:
                        error_content = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_content}")
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Request failed with status {response.status}: {error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error(f"ClientError occurred: {str(e)}")
                raise HTTPException(
                    status_code=502,
                    detail="Bad Gateway: Error communicating with upstream server.",
                ) from e
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e)) from e
    
    def _estimate_sequence_length(self, request_data: Dict[str, Any]) -> int:
        """Estimate sequence length from request data"""
        if "prompt" in request_data:
            return len(request_data["prompt"].split())
        elif "messages" in request_data:
            total_length = 0
            for message in request_data["messages"]:
                if "content" in message:
                    total_length += len(message["content"].split())
            return total_length
        return 0  # Default for unknown formats
    
    def _track_request(self, request_id: str, sequence_length: int):
        """Track request in history and add to bucketing"""
        with self._lock:
            self.request_history[request_id] = RequestInfo(
                request_id=request_id,
                sequence_length=sequence_length,
                status=RequestStatus.PENDING,
                start_time=time.time()
            )
            
            # Add to bucketing if using adaptive scheduling
            if isinstance(self.scheduling_policy, AdaptiveBucketingSchedulingPolicy):
                self.scheduling_policy.add_request(sequence_length, request_id)
    
    def _update_request_status(self, request_id: str, status: RequestStatus, 
                             prefill_instance: str = None, decode_instance: str = None):
        """Update request status"""
        with self._lock:
            if request_id in self.request_history:
                req_info = self.request_history[request_id]
                req_info.status = status
                if prefill_instance:
                    req_info.prefill_instance = prefill_instance
                if decode_instance:
                    req_info.decode_instance = decode_instance
                if status == RequestStatus.PREFILLING:
                    req_info.prefill_time = time.time()
                elif status == RequestStatus.DECODING:
                    req_info.decode_time = time.time()
                
                # Update status in bucket if using adaptive scheduling
                if isinstance(self.scheduling_policy, AdaptiveBucketingSchedulingPolicy):
                    # Pass the assigned instance to the bucket
                    assigned_instance = prefill_instance if status == RequestStatus.PREFILLING else decode_instance
                    self.scheduling_policy.update_request_status(request_id, status.value, assigned_instance)
                
                # Remove from bucket if request is completed or failed
                if status in [RequestStatus.COMPLETED, RequestStatus.FAILED]:
                    if isinstance(self.scheduling_policy, AdaptiveBucketingSchedulingPolicy):
                        self.scheduling_policy.remove_request(request_id)
                        logger.debug(f"Removed {status.value} request {request_id} from buckets")
    
    async def get_status(self):
        """Get server status"""
        status_info = {
            "prefill_node_count": len(self.prefill_instances),
            "decode_node_count": len(self.decode_instances),
            "prefill_nodes": self.prefill_instances,
            "decode_nodes": self.decode_instances,
            "total_requests": len(self.request_history),
            "model": self.model,
            "scheduling_policy": type(self.scheduling_policy).__name__
        }
        
        # Add request status counts
        with self._lock:
            status_counts = {}
            for req_info in self.request_history.values():
                status = req_info.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            status_info["request_status"] = status_counts
        
        return status_info
    
    async def get_bucket_status(self):
        """Get bucket statistics"""
        if isinstance(self.scheduling_policy, AdaptiveBucketingSchedulingPolicy):
            return self.scheduling_policy.get_bucket_statistics()
        else:
            return {"message": "Not using adaptive bucketing policy"}
    
    async def create_completion(self, raw_request: Request):
        """Handle completion requests with PD separation and bucketing"""
        try:
            request_data = await raw_request.json()
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            sequence_length = self._estimate_sequence_length(request_data)
            
            # Validate sequence length
            if not sequence_length:
                raise HTTPException(status_code=400, detail="Request content cannot be empty")
            
            # Track request
            self._track_request(request_id, sequence_length)
            logger.info(f"Processing completion request {request_id} (length: {sequence_length})")
            
            # Create KV prepare request (prefill stage) - max_tokens=1 like dis_demo.py
            kv_prepare_request = request_data.copy()
            kv_prepare_request["max_tokens"] = 1
            
            # Schedule prefill instance using bucketing strategy
            prefill_instance = self.scheduling_policy.schedule(self.prefill_cycler, sequence_length, "prefill")
            self._update_request_status(request_id, RequestStatus.PREFILLING, prefill_instance)
            
            logger.info(f"Prefill stage for {request_id} (length={sequence_length}) on {prefill_instance}")
            
            # Execute prefill stage
            try:
                async for _ in self.forward_request(
                        f"http://{prefill_instance}/v1/completions", kv_prepare_request
                ):
                    continue
            except HTTPException as http_exc:
                self.remove_instance_endpoint("prefill", prefill_instance)
                self._update_request_status(request_id, RequestStatus.FAILED)
                raise http_exc
            
            # Schedule decode instance using bucketing strategy
            decode_instance = self.scheduling_policy.schedule(self.decode_cycler, sequence_length, "decode")
            self._update_request_status(request_id, RequestStatus.DECODING, decode_instance=decode_instance)
            
            logger.info(f"Decode stage for {request_id} (length={sequence_length}) on {decode_instance}")
            
            # Execute decode stage with full request
            try:
                generator = self.forward_request(
                    f"http://{decode_instance}/v1/completions", request_data
                )
                
                # Create streaming response with non-blocking completion callback
                async def streaming_generator():
                    try:
                        async for chunk in generator:
                            yield chunk
                        # Use direct synchronous call to avoid deadlock
                        self._update_request_status_sync(request_id, RequestStatus.COMPLETED)
                        logger.info(f"Completed streaming for request {request_id}")
                    except Exception as e:
                        # Use direct synchronous call to avoid deadlock
                        self._update_request_status_sync(request_id, RequestStatus.FAILED)
                        logger.error(f"Streaming failed for request {request_id}: {e}")
                        raise e
                
                return StreamingResponse(streaming_generator())
                
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                self._update_request_status(request_id, RequestStatus.FAILED)
                raise http_exc
                
        except Exception as e:
            logger.error(f"Error in create_completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def create_chat_completion(self, raw_request: Request):
        """Handle chat completion requests with PD separation and bucketing"""
        try:
            request_data = await raw_request.json()
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            sequence_length = self._estimate_sequence_length(request_data)
            
            # Validate sequence length
            if not sequence_length:
                raise HTTPException(status_code=400, detail="Request content cannot be empty")
            
            # Track request
            self._track_request(request_id, sequence_length)
            logger.info(f"Processing chat completion request {request_id} (length: {sequence_length})")
            
            # Create KV prepare request (prefill stage) - max_tokens=1 like dis_demo.py
            kv_prepare_request = request_data.copy()
            kv_prepare_request["max_tokens"] = 1
            if "max_completion_tokens" in kv_prepare_request:
                kv_prepare_request["max_completion_tokens"] = 1
            
            # Schedule prefill instance using bucketing strategy
            prefill_instance = self.scheduling_policy.schedule(self.prefill_cycler, sequence_length, "prefill")
            self._update_request_status(request_id, RequestStatus.PREFILLING, prefill_instance)
            
            logger.info(f"Prefill stage for {request_id} (length={sequence_length}) on {prefill_instance}")
            
            # Execute prefill stage
            try:
                async for _ in self.forward_request(
                        f"http://{prefill_instance}/v1/chat/completions", kv_prepare_request
                ):
                    continue
            except HTTPException as http_exc:
                self.remove_instance_endpoint("prefill", prefill_instance)
                self._update_request_status(request_id, RequestStatus.FAILED)
                raise http_exc
            
            # Schedule decode instance using bucketing strategy
            decode_instance = self.scheduling_policy.schedule(self.decode_cycler, sequence_length, "decode")
            self._update_request_status(request_id, RequestStatus.DECODING, decode_instance=decode_instance)
            
            logger.info(f"Decode stage for {request_id} (length={sequence_length}) on {decode_instance}")
            
            # Execute decode stage with full request - defer status update until streaming completes
            try:
                generator = self.forward_request(
                    f"http://{decode_instance}/v1/chat/completions", request_data
                )
                
                # Create streaming response with non-blocking completion callback
                async def streaming_generator():
                    try:
                        async for chunk in generator:
                            yield chunk
                        # Use direct synchronous call to avoid deadlock
                        self._update_request_status_sync(request_id, RequestStatus.COMPLETED)
                        logger.info(f"Completed streaming for chat request {request_id}")
                    except Exception as e:
                        # Use direct synchronous call to avoid deadlock
                        self._update_request_status_sync(request_id, RequestStatus.FAILED)
                        logger.error(f"Chat streaming failed for request {request_id}: {e}")
                        raise e
                
                return StreamingResponse(content=streaming_generator())
                
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                self._update_request_status(request_id, RequestStatus.FAILED)
                raise http_exc
                
        except Exception as e:
            logger.error(f"Error in create_chat_completion: {e}")
            return StreamingResponse(
                content=iter([str(e)]), media_type="text/event-stream"
            )
    
    def remove_instance_endpoint(self, instance_type: str, instance: str):
        """Remove failed instance"""
        if instance_type == "decode" and instance in self.decode_instances:
            self.decode_instances.remove(instance)
            self.decode_cycler = itertools.cycle(self.decode_instances)
            logger.warning(f"Removed failed decode instance: {instance}")
        if instance_type == "prefill" and instance in self.prefill_instances:
            self.prefill_instances.remove(instance)
            self.prefill_cycler = itertools.cycle(self.prefill_instances)
            logger.warning(f"Removed failed prefill instance: {instance}")


class xPyDProxyServer:
    """xPyD Proxy Server with PD separation and adaptive bucketing"""
    
    def __init__(
        self,
        args: argparse.Namespace,
        scheduling_policy: Optional[SchedulingPolicy] = None,
    ):
        self.validate_parsed_serve_args(args)
        self.port = args.port
        
        # Set scheduling policy
        if scheduling_policy is None:
            if args.scheduling == "adaptive_bucketing":
                scheduling_policy = AdaptiveBucketingSchedulingPolicy(
                    l_max=args.l_max, 
                    n_max=args.n_max, 
                    theta=args.theta,
                    max_active_requests_per_instance=args.max_active_requests_per_instance
                )
            else:
                scheduling_policy = RoundRobinSchedulingPolicy()
        
        self.proxy_instance = xPyDProxy(
            prefill_instances=[] if args.prefill is None else args.prefill,
            decode_instances=[] if args.decode is None else args.decode,
            model=args.model,
            scheduling_policy=scheduling_policy,
        )
    
    def validate_parsed_serve_args(self, args: argparse.Namespace):
        """Validate server arguments"""
        if not args.prefill:
            raise ValueError("Please specify at least one prefill node.")
        if not args.decode:
            raise ValueError("Please specify at least one decode node.")
        self.validate_instances(args.prefill)
        self.validate_instances(args.decode)
    
    def validate_instances(self, instances: List[str]):
        """Validate instance format"""
        for instance in instances:
            if len(instance.split(":")) != 2:
                raise ValueError(f"Invalid instance format: {instance}")
            host, port = instance.split(":")
            try:
                if host != "localhost":
                    import ipaddress
                    ipaddress.ip_address(host)
                port = int(port)
                if not (0 < port < 65536):
                    raise ValueError(f"Invalid port number in instance: {instance}")
            except Exception as e:
                raise ValueError(f"Invalid instance {instance}: {str(e)}") from e
    
    def run_server(self):
        """Run the proxy server"""
        app = FastAPI(title="xPyD Disaggregated Proxy Server with Adaptive Bucketing")
        app.include_router(self.proxy_instance.router)
        
        config = uvicorn.Config(app, port=self.port, loop="uvloop")
        server = uvicorn.Server(config)
        server.run()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser("xPyD Disaggregated Proxy Server with Adaptive Bucketing")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name")
    
    parser.add_argument(
        "--prefill",
        "-p",
        type=str,
        nargs="+",
        help="List of prefill node URLs (host:port)",
    )
    
    parser.add_argument(
        "--decode",
        "-d",
        type=str,
        nargs="+",
        help="List of decode node URLs (host:port)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number",
    )
    
    parser.add_argument(
        "--scheduling",
        type=str,
        choices=["round_robin", "adaptive_bucketing"],
        default="adaptive_bucketing",
        help="Scheduling policy",
    )
    
    # Bucketing parameters
    parser.add_argument(
        "--l-max",
        type=int,
        default=32768,
        help="Maximum sequence length for bucketing (L_max)",
    )
    
    parser.add_argument(
        "--n-max",
        type=int,
        default=10,
        help="Maximum requests before bucket operations (N_max)",
    )
    
    parser.add_argument(
        "--theta",
        type=float,
        default=0.5,
        help="Splitting threshold θ",
    )
    
    # Load balancing parameters
    parser.add_argument(
        "--max-active-requests-per-instance",
        type=int,
        default=30,
        help="Maximum active requests per instance before using round-robin",
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create and run server
    proxy_server = xPyDProxyServer(args=args)
    proxy_server.run_server()


if __name__ == "__main__":
    main() 
