#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

set -xe

echo "~_~Z~_~Z Warning: The usage of disaggregated prefill is experimental and subject to change ~_~Z~_~Z"
sleep 1

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

export VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1

# You can also adjust --kv-ip and --kv-port for distributed inference.

# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=9527 vllm serve Qwen/Qwen3-32B \
    --port 8100 \
    --enforce-eager \
    --served-model-name qwen3-32b \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' &

# decoding instance, which is the KV consumer
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=9627 vllm serve Qwen/Qwen3-32B \
    --port 8200 \
    --enforce-eager \
    --served-model-name qwen3-32b \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - send the request to prefill vLLM instance (port 8100), change max_tokens
#   to 1
# - after the prefill vLLM finishes prefill, send the request to decode vLLM
#   instance
# NOTE: the usage of this API is subject to change --- in the future we will
# introduce "vllm connect" to connect between prefill and decode instances
python3 disagg_proxy_server.py  \
  --model qwen3-32b  \
  --prefill localhost:8100 \
  --decode localhost:8200 \
  --port 9000