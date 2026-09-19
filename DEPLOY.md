# Deployment: 16 GB VRAM llama.cpp chat + embeddings (fork)

This fork is **llama.cpp only**: no ComfyUI, no TTS, no Vosk. It serves one
OpenAI-compatible chat endpoint (port **11434**) from a pinned
GenerelSchwerz `moe-cache` llama.cpp build, following the owner's
[16GB-VRAM-Setup](https://github.com/GenerelSchwerz/llama.cpp/wiki/16GB-VRAM-Setup).

## Hardware sizing

- **GPU:** one NVIDIA GPU with ~16 GB VRAM (guide owner measured on
  RTX 5070 Ti, 16,303 MiB). The CUDA build targets
  `CUDA_DOCKER_ARCH=75;80;86;89;90;120;126;127` (Ampere–Blackwell consumer).
- **VRAM at runtime:** cache 132 reached 14,369 MiB peak (NP1) / 14,675 MiB
  (NP4) on the owner card. `LLAMA_MOE_CACHE_SIZE=96` is the recommended start
  for mixed use or shared displays; raise/lower it per request geometry:
  NP4 start at 48, NP8 at 64, NP16 at 48 in the guide's parallel table.
- **System RAM:** 32 GB practical minimum for the 20.4 GB Q4_K_M with
  `--load-mode none`; 62 GiB+ preferred (guide owner system).

## Provisioning

Weights are never baked into the image:

1. Set `LLAMA_DOWNLOAD_MODELS=1` in `runtime.env` (license: Qwen3.6, Apache-2.0;
   pinned revision in `scripts/llama-models.json`). First chat start downloads
   and SHA256-verifies `Qwen3.6-35B-A3B-Q4_K_M.gguf` (20,419,565,568 bytes)
   into `$LLAMA_MODEL_DIR`, then requires a CUDA device and waits for the
   `moe-cache: CUDA_MoE_Cache_Pool[...]` activation line before writing its
   readiness receipt. A missing activation is fatal, not silently accepted.
2. Nomic embeddings download into `$LLAMA_EMBEDDING_MODEL_DIR` on their own
   start; the second llama-server is CPU-only (`--device none`).
3. `netbird` enrolls the box on the private overlay; the nginx proxy on 11434
   rejects browser origins and proxies 11435/11436 loopback servers.

## Runtime tuning

`LLAMA_MOE_CACHE_SIZE` (0..1024), context/batch/ubatch and reasoning controls
can be overridden persistently in
`/workspace/llama.cpp/runtime-settings.json` (bounded allowlist, restart the
chat service only after checking `nvidia-smi`). Old Codacus/Flash recipe
variables (`LLAMA_MOE_CACHE_SLOTS`, `LLAMA_MTP_MODEL_DIR`,
`LLAMA_GENERATE_PROFILE`, `GGML_SCHED_PREFETCH_EXPERTS`, `OLLAMA_*`,
`LLAMA_ARG_*`) are rejected at startup and must be removed.

## Operations

- `supervisorctl -c /etc/supervisor/praxis.conf status` — services:
  `netbird`, `llama-chat`, `llama-proxy`, `llama-embedding`. Chat/embedding
  never autorestart after a failed preparation (missing weights, OOM, cache
  fallback): restart them deliberately after fixing the cause.
- Logs: `/workspace/logs/{llama-chat,llama-embedding,llama-proxy-supervisor}.log`.
- Container health (`/opt/praxis/bin/healthcheck.py`) requires the chat
  readiness receipt (same argv + same PID + active cache), `status=ok` on
  OpenAI `/health`, embeddings when enabled, and a ready NetBird peer.
- Acceptance: `python3 /opt/praxis/bin/acceptance_llama.py --run` runs six
  bounded GPU generations (plain, tool call, ~2k-token prompt), a CPU embedding
  request, and writes a private JSON report.

## Image build and verification

```sh
bash scripts/build.sh          # builds the pinned moe-cache fork with CUDA
bash scripts/verify-image.sh   # CPU-only checks: flags, supervisor, proxy, router mode
```

The image build fails if the pinned commit's `llama-server --help` lacks
`--moe-expert-cache-size`, `--cache-ram`, `--load-mode`, `--kv-offload`. Do not
use the expert-cache flags on Vulkan/ROCm/Metal/CPU-only backends (owner
warning; CUDA required).

## Praxis client

Point Praxis at the NetBird IP of this box on port 11434 (`LLAMA_*`-free
context: `settings.llama_base_url=http://<netbird-ip>:11434`). Use
`qwen3.6-35b-a3b` as the model id. Embedding dimensions stay 768 (Nomic v1.5),
so existing Praxis vector indices remain compatible.