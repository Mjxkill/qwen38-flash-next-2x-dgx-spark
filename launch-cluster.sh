#!/usr/bin/env bash
# Orchestrator: launches the 2-node SGLang cluster for Qwen3.8-Flash-Next.
# Run from the worker node; the head node is reached over SSH.
# GB10 note: purge the model blobs from the page cache before loading, or the
# unified-memory GPU allocator starves (NV_ERR_NO_MEMORY at ~8% load).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HEAD=user@10.99.1.1   # head node (rank 0, serves :8000)

echo "[qwen38] dropping HF blobs from page cache on both nodes..."
sync; echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
ssh "$HEAD" 'sync; echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null' || true

echo "[qwen38] rank 1 (worker, local)..."
bash "$HERE/sglang-node.sh" 1
echo "[qwen38] rank 0 (head)..."
ssh "$HEAD" 'bash ~/sglang-node.sh 0'
echo "[qwen38] follow with: docker logs -f sglang_qwen38fn (on the head for the API)"
