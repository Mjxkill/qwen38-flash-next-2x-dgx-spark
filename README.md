# Qwen3.8-Flash-Next (NVFP4) on 2× DGX Spark — the long-context-SAFE path

Running [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(180B hybrid MoE, 6B active, vision, native 262K context) on **two NVIDIA DGX Spark
(GB10, SM121)** linked by a 200G RoCE cable, TP=2, via **SGLang**.

Two working paths are documented here, both at full context and neither using the
kernel that corrupts long context on SM121: **vLLM at 26.5 tok/s with a 1M-token
context** (added 2026-09-10, needs one small patch) and **SGLang at 10–14 tok/s**
(262K). As far as we know these are the first published numbers for either on this
hardware.

## Update 2026-09-10 — the vLLM path now works, and it is ~2× faster

The vLLM route that was a dead end on 2026-09-03 (see trap #4 below) now runs,
on the `eugr/spark-vllm:nightly-20260909` nightly plus **one local patch**:

| Path | Decode | Context | Notes |
|---|---|---|---|
| **vLLM 0.28.1 nightly + PLE patch** | **26.5 tok/s** | **1,048,576** (YaRN ×4) | this section |
| SGLang, safe QSA kernel | 10–14 tok/s | 262,144 | rest of this repo |
| SGLang, TRT-LLM kernel | 50–70 tok/s | corrupts >120K | do not use |

Same hardware (2× DGX Spark, GB10/SM121, TP=2 over 200G RoCE), same NVFP4
checkpoint. Files: `Dockerfile.vllm`, `patch_ple.py`, `qwen38-flash.yaml`.

### The one patch you still need

Upstream's fix for [vllm #54765](https://github.com/vllm-project/vllm/issues/54765)
landed but is **incomplete**: it only handles `ModelOptMixedPrecisionConfig`.
This checkpoint carries a plain `ModelOptNvFp4Config` whose `quantization_config`
lists the PLE n-gram table under `ignore` **while the checkpoint quantizes it to
FP8 anyway** (the `ngram_embedding.weight_scale` tensor is right there in the
index). So the loader still builds a plain embedding, registers no
`weight_scale`, and dies at shard 203/206. `patch_ple.py` selects the FP8
embedding method for ModelOpt NVFP4 configs; `Dockerfile.vllm` applies it.

**Build it with `docker build`, never `docker commit`.** A commit freezes the
patch container's config and overwrites the image's `ENTRYPOINT`
(`/opt/nvidia/nvidia_entrypoint.sh`), after which the Ray head never becomes
ready and sparkrun reports only "Ray head failed to become ready". Rebuild the
image on each node from the same 3-line Dockerfile rather than shipping 27 GB
over the wire.

### Config gotchas (vLLM 0.28)

- **1M context**: `--rope-scaling` no longer exists. Use `--hf-overrides`, and
  **nest the override under `text_config`** — this is a multimodal model, so a
  top-level `rope_scaling` is ignored and vLLM still derives 262144, then
  refuses your `--max-model-len 1048576`.
- **KV cache must be bf16**: `--kv-cache-dtype fp8` is rejected outright
  (`Qwen4Exp QSA requires a BF16 main KV cache`). At 1M that is a 96 GB KV pool.
- **Thinking is on by default** and leaks into answers ("We need answer user's
  request in French: ..."). Pass `--chat-template-kwargs '{"enable_thinking":
  false}'` at serve time, or `chat_template_kwargs` per request.
- **Free-memory check**: the engine wants `gpu_memory_utilization × 121 GB` free
  *at startup*. On GB10's unified memory anything else resident counts — a
  leftover ComfyUI on the second node (11 GB) is enough to abort the launch with
  "Free memory 88.15/121.63 GiB is less than desired". Kill other engines on
  **both** nodes first.
- **Never launch while a large rsync is running**: sparkrun's `sync` step blocks
  on the dirty pages of the copy, silently, for tens of minutes.

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

vLLM path (faster, 1M context):
- `Dockerfile.vllm` — nightly base + the PLE patch
- `patch_ple.py` — the patch itself (self-checking; refuses to double-patch)
- `qwen38-flash.yaml` — serve recipe (sparkrun-style; the `command:` block is a
  plain `vllm serve` you can run by hand)

SGLang path (262K):
- `launch-cluster.sh` — orchestrator: cleanup, then worker (rank 1) and head (rank 0)
- `sglang-node.sh` — per-node `docker run` with the full serve command
- `Dockerfile.sm121` — day-0 base + the mrope patch (see trap #2)

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

4. **The vLLM route was a dead end for a week, and now is not** — see the
   2026-09-10 section above. For the record, the three walls were: (a)
   `--kv-cache-dtype fp8` rejected (QSA needs bf16); (b) the FP8 PLE n-gram table
   failing to load ([vllm #54765](https://github.com/vllm-project/vllm/issues/54765),
   still needs the local patch); (c) a device-side assert
   (`vectorized_gather_kernel: index out of bounds`) during sampler warmup, which
   the `skip_topk` MTP fix in the 2026-09-09 nightly cleared.

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
