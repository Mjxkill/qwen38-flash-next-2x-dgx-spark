# Qwen3.8-Flash-Next (NVFP4) on 2× DGX Spark — the long-context-SAFE path

Running [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(180B hybrid MoE, 6B active, vision, native 262K context) on **two NVIDIA DGX Spark
(GB10, SM121)** linked by a 200G RoCE cable, TP=2, via **SGLang**.

This repo documents the **corruption-safe** kernel path and, as far as we know, the
first published benchmark numbers for it on this hardware. TL;DR: **it works, quality
is clean, but decode is 10–14 tok/s** — the price of correctness on SM121 today.

## Why "safe path"?

The fast FlashInfer TRT-LLM sparse-decode kernel **silently corrupts long contexts on
SM121/GB10** — token-ID-0 (`!`) corruption in 1/4 runs at 120K tokens, 4/4 at 210K,
with HTTP 200 throughout ([sglang #36806](https://github.com/sgl-project/sglang/pull/36806)).
Upstream's answer is a purpose-built Triton online-softmax packed-varlen QSA decode
kernel ([sglang #36845](https://github.com/sgl-project/sglang/pull/36845), merged
2026-08-30): **TRT-LLM is now refused on SM121 by design**, every decode goes through
the Triton kernel.

The widely-shared 50–70 tok/s dual-Spark numbers
([tonyd2wild's excellent day-0 repo](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark),
on which this recipe is based) were measured **on the corrupting kernel**. Fine below
~120K, unsafe above. If you need the model's full context (or YaRN toward 1M), you
need this path — and its cost.

## Measured results (2026-09-03, safe path)

| Metric | Value |
|---|---|
| Decode, single stream | **10–14 tok/s** (~310 ms/step) |
| MTP (NEXTN, 3 steps) accept length | 3.1–3.4 (rate 0.67–0.81) |
| CUDA graphs | active (decode) |
| Weight load | 135 GB NVFP4, ~5 min/node (sequential) |
| Free VRAM after load | ~37 GB/node (mem-fraction 0.80) |
| Inter-node RDMA during decode | ~724 MB/s (verified via port counters) |
| Quality smoke tests | clean (FR/EN prose, code, factual) |

Per-step time is ~6× the TRT-LLM path (their report: ~50 ms/step). The Triton kernel
itself is 1.6–4.5× *faster* than the old FA4-cute fallback (PR numbers) — it is not
badly written, SM121 simply has no validated fast path yet. Watch #36845 follow-ups.

## Files

- `launch-cluster.sh` — orchestrator: cleanup, then worker (rank 1) and head (rank 0)
- `sglang-node.sh` — per-node `docker run` with the full serve command
- `Dockerfile.sm121` — the image: day-0 base + the one patch still needed (see trap #2)

## Build

```bash
docker pull lmsysorg/sglang:qwen38flashnext
docker build -f Dockerfile.sm121 -t qwen38fn:sm121 .
# copy image to the second node over your inter-node link (never download twice):
docker save qwen38fn:sm121 | ssh node2 docker load
```

## Traps we paid for

1. **Don't apply PR #36845 yourself.** The `lmsysorg/sglang:qwen38flashnext` tag has
   been rebuilt upstream and **already contains** the merged PR (bundled diffs from
   older recipes fail with "already exists in working directory"). Check with
   `git apply --check` before patching anything.

2. **The mrope partial-rotary OOB fix is still needed** (vision calls read out of
   bounds in the fused M-RoPE Triton kernel under CUDA graphs). The kernel moved to
   `python/sglang/kernels/ops/attention/rotary_triton.py`; `Dockerfile.sm121` bounds
   the temporal mask there (a no-op for full-rotary models).

3. **Do NOT `LD_PRELOAD` a host NCCL** (older recipes preload 2.30.4). The rebuilt
   image bundles NCCL 2.30.7 and DeepEP hard-asserts on a duplicate NCCL runtime:
   `AssertionError: Duplicate NCCL runtime found`. Drop the preload and the mount.

4. **The vLLM route is not usable today** (checked on a 0.28.1 nightly with NVIDIA's
   fresh `qwen4_exp` implementation): (a) `--kv-cache-dtype fp8` is rejected — QSA
   requires a BF16 main KV cache; (b) the ModelOpt NVFP4 checkpoint's FP8-quantized
   PLE n-gram table fails to load ([vllm #54765](https://github.com/vllm-project/vllm/issues/54765));
   (c) with (b) worked around, a device-side assert (`vectorized_gather_kernel: index
   out of bounds`) fires during sampler warmup. We stopped there and moved to SGLang.

5. **GB10 unified memory**: purge the model blobs from the Linux page cache before
   loading (posix_fadvise DONTNEED or `drop_caches`), or the GPU allocator starves.

6. **Launch order matters**: start the worker (rank 1) first, then the head (rank 0).
   The head serves the OpenAI-compatible API on `:8000`.

7. **Idle GB10s can report 96% GPU-util with no process** — `nvidia-smi` holds the
   last sample. Power draw (~19 W idle) tells the truth.

## Context length

Served at the native 262,144 here. The model card allows YaRN scaling to ~1M
(RoPE factor 4.0); the KV pool reaches ~1.05M tokens at mem-fraction 0.82 with
radix cache off (see tonyd2wild's report). At 10–14 tok/s we did not pursue it —
prefill time dominates long before memory does.

## Credits

- [tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark) — the base recipe, the SM121 story, the mrope analysis
- [sglang #36806](https://github.com/sgl-project/sglang/pull/36806) / [#36845](https://github.com/sgl-project/sglang/pull/36845) — the corruption find and the safe kernel
- [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) — the NVFP4 checkpoint
