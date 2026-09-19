# praxis-llamap-gs-gpu-fork

16 GB VRAM llama.cpp chat/embedding companion for Praxis. Fork of
`praxis-llamap-gpu` **stripped to the llama.cpp stack only** (no ComfyUI, no
TTS, no Vosk) and rebuilt around **GenerelSchwerz's `moe-cache` branch** of
llama.cpp, following the owner's [16GB VRAM Setup](https://github.com/GenerelSchwerz/llama.cpp/wiki/16GB-VRAM-Setup).

- **Model choice via `LLAMA_MODEL`** (pinned variants in
  `scripts/llama-models.json`); unset means the manifest default:
  - **`qwen3.8-flash-next`** (default) — `unsloth/Qwen3.8-Flash-Next-GGUF`
    UD-IQ3_XXS, 3 shards, **82 GB** + shared **Q8 MTP draft head** (2.8 GB).
    177B MoE; needs an ~82 GB VRAM-class GPU. Geometry: 4096 ctx / 512 batch /
    ubatch / 56 expert-cache slots. This is the model that works (the original
    stack's recipe, now on the GenerelSchwerz moe-cache fork with draft-mtp).
  - **`qwen3.6-35b-a3b`** — `ggml-org/Qwen3.6-35B-A3B-GGUF` Q4_K_M, one 20.4 GB
    GGUF, **16 GB VRAM guide** model with CUDA MoE expert cache:
    `LLAMA_MOE_CACHE_SIZE=96` (cache 132 measured 14,369 MiB peak on the
    owner's RTX 5070 Ti 16 GB), `-c 65536 -b 8192 -ub 8192`, F16 KV offloaded,
    `--load-mode none --cache-ram 0 -fit off -fa on --jinja`.
    **32 GB system RAM is the practical minimum** for this one; 62 GiB+ preferred.
- Every variant carries its own default geometry; override any value between
  launcher bounds with `LLAMA_CONTEXT_SIZE`, `LLAMA_BATCH_SIZE`,
  `LLAMA_UBATCH_SIZE`, `LLAMA_MOE_CACHE_SIZE`.
- One CPU worker per available physical core, auto-detected with
  affinity/cpuset and CPU-quota limits (`LLAMA_THREADS=auto`).
- Runtime env opts into automatic download with SHA256 verification; verified
  files are reused (`LLAMA_DOWNLOAD_MODELS=1`).
- Nomic embeddings remain available through a second, CPU-only llama-server.
- Private VPN port: **11434** (OpenAI-compatible, nginx origin guard).

```sh
cp runtime.env.example runtime.env
chmod 600 runtime.env
# Review model licenses and set NB_SETUP_KEY privately in runtime.env.
bash scripts/build.sh
bash scripts/verify-image.sh
docker run -d --name praxis-llamap-gs-gpu --gpus all \
  --env-file runtime.env -v praxis-llamap-gs-gpu-data:/workspace praxis-llamap-gpu:dev
```

### Switching models at runtime

```sh
# 82 GB default (Flash-Next 177B + MTP draft):
LLAMA_MODEL=qwen3.8-flash-next

# 16 GB VRAM class (Qwen3.6-35B-A3B, needs >=32 GB system RAM):
LLAMA_MODEL=qwen3.6-35b-a3b
```

That is a single environment variable; the first start downloads and
SHA256-verifies that variant's pinned files. **Adding your own Qwen model**
means adding a pinned variant entry to `scripts/llama-models.json` (repository,
revision, per-file size + sha256 — e.g. `sha256sum` of the downloaded GGUF),
its default geometry, and optionally a `draft` block; then rebuild the script
layer (`docker build` reuses the heavy layers). Example manifest entry below.

**No public port mappings.** First startup downloads ~20.4 GB of chat weights
plus ~274 MB of embeddings and verifies pinned SHA256s; subsequent starts reuse
verified files. Do not enable `--moe-expert-cache-size` on non-CUDA backends;
the launcher refuses a CPU-only chat fallback.

Source pins: [`LLAMA_CPP.json`](LLAMA_CPP.json) (GenerelSchwerz `moe-cache`
commit) and [`scripts/llama-models.json`](scripts/llama-models.json) (pinned
variants: repository revision + SHA256 + geometry defaults). Adding a variant:

```json
"my-qwen": {
  "alias": "my-qwen",
  "license": "<license url>",
  "note": "<what GPU/RAM it needs>",
  "context": 8192, "batch": 1024, "ubatch": 1024, "cache_size": 48,
  "repository": "<org>/<repo>-GGUF",
  "revision": "<40 hex chars>",
  "files": [{"path": "<path-in-repo>.gguf", "size": <bytes>, "sha256": "<64 hex>"}]
}
```

Then `LLAMA_MODEL=my-qwen` activates it. Multi-file GGUFs list every shard;
add a `"draft"` block with a separately pinned MTP head to enable draft-mtp.
[`BASE_STACK.json`](BASE_STACK.json) records the remaining byte-identical base
stack files. Run `python3 /opt/praxis/bin/acceptance_llama.py --run` against an
idle, ready container for bounded multi-request API regression checks.

See **[DEPLOY.md](DEPLOY.md)** for hardware sizing, VRAM tuning and operational
notes.