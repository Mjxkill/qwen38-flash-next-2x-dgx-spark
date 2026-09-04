#!/usr/bin/env bash
# SGLang node for Qwen3.8-Flash-Next-NVFP4 (TP2 across 2× DGX Spark).
# Recipe based on tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark, adapted for the
# rebuilt day-0 image (PR #36845 included, bundled NCCL 2.30.7 — no LD_PRELOAD).
#
# Usage: sglang-node.sh <0|1>   (0 = head, serves the API on :8000; 1 = worker)
# Start rank 1 first, then rank 0.
set -euo pipefail

NODE_RANK="${1:?usage: sglang-node.sh <0|1>}"

IMAGE="qwen38fn:sm121"
NAME="sglang_qwen38fn"
HEAD_IP="10.99.1.1"      # head node's IP on the 200G inter-node link
INIT_PORT="29511"
PORT="8000"

# Point this at your HF-cache copy of the model (the models--RadixArk--... dir).
REPO_HOST_PATH="$HOME/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
SNAP="$(ls "$REPO_HOST_PATH/snapshots" | head -1)"
MODEL_PATH="/models/repo/snapshots/$SNAP"
CACHE_HOST_PATH="/var/tmp/qwen38fn-sglang-cache"

test -f "$REPO_HOST_PATH/snapshots/$SNAP/config.json"
mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true

docker run --gpus all -d \
  --name "$NAME" --restart unless-stopped \
  --memory 110g --memory-swap 110g \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$REPO_HOST_PATH:/models/repo:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_CACHE=/cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/cache/huggingface/hub \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
  sglang serve \
    --model-path "$MODEL_PATH" \
    --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port "$PORT" \
    --tp-size 2 \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --dist-init-addr "$HEAD_IP:$INIT_PORT" \
    --quantization modelopt_fp4 \
    --fp4-gemm-backend flashinfer_cutlass \
    --page-size 64 \
    --mamba-scheduler-strategy extra_buffer \
    --mamba-track-interval 64 \
    --max-mamba-cache-size 97 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --enable-linear-replayssm-spec \
    --speculative-attention-mode decode \
    --chunked-prefill-size 4096 \
    --max-running-requests 6 \
    --context-length 262144 \
    --max-total-tokens 600000 \
    --mem-fraction-static 0.80 \
    --allow-auto-truncate \
    --reasoning-parser auto \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --trust-remote-code \
    --ple-offload-embedding \
    --cuda-graph-max-bs 8 \
    --disable-cuda-graph-padding \
    --disable-prefill-cuda-graph \
    --disable-radix-cache \
    --sampling-backend pytorch

echo "launched $NAME rank=$NODE_RANK"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited; docker logs $NAME" >&2
  exit 1
}
