# Qwen3.8-Flash-Next (NVFP4) on 2× DGX Spark — the long-context-SAFE path

Running [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(180B hybrid MoE, 6B active, vision, native 262K context) on **two NVIDIA DGX Spark
(GB10, SM121)** linked by a 200G RoCE cable, TP=2, via **SGLang**.

Three sets of numbers are documented here — start with the 2026-09-17 update, it supersedes the rest. Two working paths, both at full context and neither using the
kernel that corrupts long context on SM121: **vLLM at 26.5 tok/s with a 1M-token
context** (added 2026-09-10, needs one small patch) and **SGLang at 10–14 tok/s**
(262K). As far as we know these are the first published numbers for either on this
hardware.

## Heads-up 2026-09-21 — the QSA indexer OOM has a Qwen3.8 instance, and a one-line workaround

Not ours and not yet verified here, but it targets exactly this configuration
and costs nothing to try. **[vllm #56457](https://github.com/vllm-project/vllm/issues/56457)**
(opened 2026-09-11) reports the QSA indexer's per-chunk logits buffer growing
with `max_seq_len` on GB10 until it OOMs or hangs — reproducibly at 166,400
computed tokens, on 2 nodes at TP=2, fp8 KV, 262k context, with *this* NVFP4
checkpoint. The reporter's workaround is one environment variable:

```bash
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
```

and they report it **also gains roughly 10 % decode**. That is their
measurement, not ours — we have not re-run the ladder below with it.

Same root cause as [#55569](https://github.com/vllm-project/vllm/issues/55569)
(GLM-5.3-Flash, 2026-09-06). Neither is fixed upstream: the first patch
([#55572](https://github.com/vllm-project/vllm/pull/55572)) was **closed
without merging** on 2026-09-07, and the current candidate
([#57105](https://github.com/vllm-project/vllm/pull/57105), 2026-09-16 —
reserving the worst-case workspace up front, 55 segments/13.5 GB down to
3 segments/534 MiB on GB10) is still open. If you run long contexts here, set
the variable rather than wait for a release.

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

## Update 2026-09-17 — 26.5 → 35 tok/s prose, 58 tok/s code, by six flags

The vLLM path documented above works, but its settings were *our* settings, not
the fast ones. Flag for flag, that configuration is the **28.4 tok/s baseline**
in [tonyd2wild's per-setting ladder](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)
— the same hardware, the same weights. Nothing needed patching; six settings
needed changing.

| Measured here (2× DGX Spark, TP=2, NVFP4, 262k) | Before | **After** |
|---|---|---|
| Prose | 26.5 tok/s | **35.0** |
| Code | 26.5 tok/s | **58.0** |

Upstream's ladder, each step measured alone, explains where it comes from:

| Step | tok/s |
|---|---|
| Baseline — compile off, PLE table resident | 28.4 |
| \+ `--max-num-batched-tokens 4096` | **45.1** (+59%) |
| \+ MTP 3 and `--max-num-seqs 6` | **53.7** (+19%) |

### The six settings

```bash
--max-num-batched-tokens 4096          # was 8192 — the single biggest lever on GB10
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--max-num-seqs 6
--no-enable-prefix-caching             # crashes the GDN kernel on growing multi-turn prompts (vllm #54173)
--no-enable-flashinfer-autotune        # "Invalid gemm2 profile id" on FlashInfer <= 0.6.17
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

plus two environment variables that matter on this hardware:

```bash
VLLM_USE_DEEP_GEMM=0        # blockwise FP8 DeepGEMM: "unspecified launch failure" on sm_121 (vllm #54125)
VLLM_PLE_CPU_OFFLOAD=0      # see below — this one is a trap that arrived last week
```

### `VLLM_PLE_CPU_OFFLOAD` — new default, wrong default for GB10

[PR #54371](https://github.com/vllm-project/vllm/pull/54371) (merged 2026-09-09,
after the v0.29.0 cut) added `VLLM_PLE_CPU_OFFLOAD` **defaulting to `True`**: the
n-gram tables go to pinned CPU memory for UVA lookup. On a discrete GPU that
frees VRAM. On GB10 **host RAM is the GPU pool** — pinning them thrashes the box
instead. On any nightly from 2026-09-10 onward, set it to `0` explicitly.

### About the 1M context we used to serve

We served 1,048,576 via YaRN. Upstream is blunt about it: *"1M is a lab ceiling,
not a trained window"*, and community YaRN at 1M on GB10 hangs on long prefills
([vllm #54629](https://github.com/vllm-project/vllm/issues/54629)). The native
window is **262,144**. We went back to it — the only thing this update gives up,
and it buys the stability the rest depends on.

### Thinking off, not "medium"

We had been forcing `reasoning_effort: medium` via a patched chat template. That
is the *risky* half of a documented failure: thinking **plus** declared tools
makes the model loop on token ID 0 (`!`) until `max_tokens`
([sglang #36537](https://github.com/sgl-project/sglang/issues/36537)) —
signature `accept len: 1.00, accept rate: 0.00`. Every fast recipe ships
`--default-chat-template-kwargs '{"enable_thinking": false}'` server-side
instead. So do we now.

### Still open, deliberately

- We keep the **RadixArk** checkpoint and our local PLE patch. Upstream's
  recipes use `nvidia/Qwen3.8-Flash-Next-NVFP4`, supported natively since
  [PR #54882](https://github.com/vllm-project/vllm/pull/54882) — 135 GB to
  re-download, which would retire our patch and issue #54765 with it.
- We keep the Ray executor; every validated recipe uses `mp`. Worth an A/B.

`qwen38-flash.yaml` in this repo is the exact serve recipe, every setting
commented with why it is there.

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
