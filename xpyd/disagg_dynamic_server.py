import os
import argparse
import logging
import time
import threading
import itertools
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import aiohttp
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..scheduler.base import SchedulingPolicy
from ..scheduler.dynamic import DynamicPrioritySchedulingPolicy
from ..scheduler.roundrobin import RoundRobinSchedulingPolicy
from ..scheduler.bucket import AdaptiveBucketingSchedulingPolicy

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

        logger.info(
            f"Initialized xPyD proxy with {len(prefill_instances)} prefill and {len(decode_instances)} decode instances")

    def _update_request_status_sync(self, request_id: str, status: RequestStatus):
        """Synchronous version of status update for background worker"""
        with self._lock:
            if request_id in self.request_history:
                req_info = self.request_history[request_id]
                old_status = req_info.status
                req_info.status = status

                current_time = time.time()
                if status == RequestStatus.PREFILLING:
                    req_info.prefill_time = current_time
                elif status == RequestStatus.DECODING:
                    req_info.decode_time = current_time
                elif status == RequestStatus.COMPLETED:
                    # Calculate total response time
                    total_time = current_time - req_info.start_time
                    # Record response time for the instance that handled the request
                    assigned_instance = req_info.decode_instance or req_info.prefill_instance
                    if assigned_instance and isinstance(self.scheduling_policy, DynamicPrioritySchedulingPolicy):
                        self.scheduling_policy.record_response_time(assigned_instance, total_time)
                        logger.info(f"Recorded response time {total_time:.3f}s for instance {assigned_instance}")

                # Update status for bucket scheduling policy
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

                # Update status for dynamic priority scheduling policy
                elif isinstance(self.scheduling_policy, DynamicPrioritySchedulingPolicy):
                    # Update status for simple priority scheduling
                    assigned_instance = None
                    if status in [RequestStatus.PREFILLING, RequestStatus.DECODING]:
                        if status == RequestStatus.PREFILLING:
                            assigned_instance = req_info.prefill_instance
                        else:
                            assigned_instance = req_info.decode_instance
                    elif status in [RequestStatus.COMPLETED, RequestStatus.FAILED]:
                        # Use decode instance if available, otherwise prefill instance
                        assigned_instance = req_info.decode_instance or req_info.prefill_instance

                    if assigned_instance:
                        self.scheduling_policy.update_request_status(request_id, status.value, assigned_instance)

                # Remove from bucket if request is completed or failed
                if status in [RequestStatus.COMPLETED, RequestStatus.FAILED]:
                    if isinstance(self.scheduling_policy, AdaptiveBucketingSchedulingPolicy):
                        self.scheduling_policy.remove_request(request_id)
                        logger.info(f"Removed {status.value} request {request_id} from buckets")

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

                # Update status for bucket scheduling policy
                if isinstance(self.scheduling_policy, AdaptiveBucketingSchedulingPolicy):
                    # Pass the assigned instance to the bucket
                    assigned_instance = prefill_instance if status == RequestStatus.PREFILLING else decode_instance
                    self.scheduling_policy.update_request_status(request_id, status.value, assigned_instance)

                elif isinstance(self.scheduling_policy, DynamicPrioritySchedulingPolicy):
                    # Update status for simple priority scheduling
                    assigned_instance = None
                    if status in [RequestStatus.PREFILLING, RequestStatus.DECODING]:
                        if status == RequestStatus.PREFILLING:
                            assigned_instance = prefill_instance
                        else:
                            assigned_instance = decode_instance
                    elif status in [RequestStatus.COMPLETED, RequestStatus.FAILED]:
                        # Use decode instance if available, otherwise prefill instance
                        assigned_instance = decode_instance or prefill_instance

                    if assigned_instance:
                        self.scheduling_policy.update_request_status(request_id, status.value, assigned_instance)

                # Remove from bucket if request is completed or failed
                if status in [RequestStatus.COMPLETED, RequestStatus.FAILED]:
                    if isinstance(self.scheduling_policy, AdaptiveBucketingSchedulingPolicy):
                        self.scheduling_policy.remove_request(request_id)
                        logger.info(f"Removed {status.value} request {request_id} from buckets")

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
        """Get scheduling statistics"""
        if isinstance(self.scheduling_policy, AdaptiveBucketingSchedulingPolicy):
            return self.scheduling_policy.get_bucket_statistics()
        elif isinstance(self.scheduling_policy, DynamicPrioritySchedulingPolicy):
            return self.scheduling_policy.get_statistics()
        else:
            return {"message": "Not using supported scheduling policy"}

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
                    max_active_requests_per_instance=args.max_active_requests_per_instance,
                    priority_threshold=args.priority_threshold
                )
            elif args.scheduling == "dynamic_priority":
                scheduling_policy = DynamicPrioritySchedulingPolicy(
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
        choices=["round_robin", "adaptive_bucketing", "dynamic_priority"],
        default="dynamic_priority",
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

    # Priority scheduling parameters
    parser.add_argument(
        "--priority-threshold",
        type=float,
        default=1.2,
        help="Priority threshold multiplier for shorter requests (default: 1.2 = 20% above shortest)",
    )

    # Dynamic priority scheduling parameters
    parser.add_argument(
        "--stats-max-count",
        type=int,
        default=100000,
        help="Maximum count for dynamic priority scheduling statistics before reset",
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
