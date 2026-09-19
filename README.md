# praxis-llamap-gs-gpu-fork

16 GB VRAM llama.cpp chat/embedding companion for Praxis. Fork of
`praxis-llamap-gpu` **stripped to the llama.cpp stack only** (no ComfyUI, no
TTS, no Vosk) and rebuilt around **GenerelSchwerz's `moe-cache` branch** of
llama.cpp, following the owner's [16GB VRAM Setup](https://github.com/GenerelSchwerz/llama.cpp/wiki/16GB-VRAM-Setup).

- **Qwen3.6-35B-A3B Q4_K_M** (one pinned GGUF, ~20.4 GB) with the **CUDA MoE
  expert cache**: routed experts live in a bounded GPU working set; cold experts
  stream from system RAM on demand. `LLAMA_MOE_CACHE_SIZE=96` is the guide's
  recommended start (cache 132 measured 14,369 MiB peak on the owner's
  RTX 5070 Ti 16 GB).
- Guide geometry: `-c 65536 -b 8192 -ub 8192`, F16 KV with KV offload,
  `--load-mode none --cache-ram 0 -fit off -fa on --jinja`.
- **32 GB system RAM is the practical minimum** for the 20.4 GB GGUF with
  `--load-mode none`; 62 GiB+ preferred.
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

**No public port mappings.** First startup downloads ~20.4 GB of chat weights
plus ~274 MB of embeddings and verifies pinned SHA256s; subsequent starts reuse
verified files. Do not enable `--moe-expert-cache-size` on non-CUDA backends;
the launcher refuses a CPU-only chat fallback.

Source pins: [`LLAMA_CPP.json`](LLAMA_CPP.json) (GenerelSchwerz `moe-cache`
commit) and [`scripts/llama-models.json`](scripts/llama-models.json) (GGUF pins).
[`BASE_STACK.json`](BASE_STACK.json) records the remaining byte-identical base
stack files. Run `python3 /opt/praxis/bin/acceptance_llama.py --run` against an
idle, ready container for bounded multi-request API regression checks.

See **[DEPLOY.md](DEPLOY.md)** for hardware sizing, VRAM tuning and operational
notes.