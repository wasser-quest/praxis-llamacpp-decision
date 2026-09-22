# Praxis llama.cpp Decision GPU fork

Private NetBird LLM stack: chat, CPU Nomic embeddings, and an independent
`/v1/decision` server. No ComfyUI, TTS or Vosk. This repository supplies both
`vayayo/praxis-llamacpp-decision` and its `vayayo/praxis-llamap-gs-gpu` alias;
**never publish the historical `fork-llama` over that alias**.

Local changes are not automatically in published images. Offline tests and
upstream performance references are **not GPU, throughput or production acceptance**.
See [DEPLOY.md](DEPLOY.md) for separate release and deployment gates.

## Three pinned backends

[LLAMA_CPP.json](LLAMA_CPP.json) records all three source pins:

| Use | Selection | Binary directory |
| --- | --- | --- |
| GS chat, default | `LLAMA_CHAT_RECIPE=gs` | `/opt/llama.cpp/bin` |
| Codacus chat, alternative | `LLAMA_CHAT_RECIPE=codacus` | `/opt/llama.cpp-codacus/bin` |
| Parallel Decision, separate process | `LLAMA_DECISION_ENABLED=1` (default) | `/opt/llama.cpp-decision/bin` |

The launchers set each process's own library directory first; incompatible
fork libraries are not placed together in the global loader configuration.
Embeddings use the GS binary with `--device none --n-gpu-layers 0`.

GS uses `--moe-expert-cache-size`, CPU experts and `--load-mode none`.
Codacus uses mmap/lazy loading and `--moe-cache-profile/--moe-cache-slots`.
Its supplied profile workflow is **Flash Next only**; other models require
explicit `LLAMA_MOE_CACHE_ENABLED=0` rather than reusing an incompatible profile.
A required cache failing to activate is an error. Explicit cache disable and
the dense model are distinct, supported configurations, not failed activation.

## Model selection and unchanged current defaults

Models, revisions, file sizes and SHA256s are pinned in
[scripts/llama-models.json](scripts/llama-models.json).

| `LLAMA_MODEL` | Weight files, approximately | Context | Batch/ubatch | GS cache |
| --- | --- | ---: | ---: | ---: |
| `qwen3.8-flash-next` (default) | 82 GB + 2.8 GB Q8 MTP | 262144 | 512 | 56 |
| `qwen3.6-35b-a3b` | 20.4 GB Q4_K_M, no MTP | 131072 | 8192 | 96 |
| `qwen3.8-27b` | 16.5 GB + 1.4 GB Q4_0 MTP, dense | 32768 | 1024 | 0 |

**Weight-file size is not VRAM demand.** The 82 GB Flash Next model offloads
experts to CPU memory; it does not intrinsically require an 82 GB GPU. Conversely,
16 GB VRAM alone does not demonstrate that the current context/cache settings fit.
Budget host RAM, GPU KV/runtime/cache memory, the concurrent Decision server,
local SSD and download charges. For GS Flash Next, prefer 96 GiB+ host RAM;
62 GiB is below the complete weight set. Qwen3.6 needs at least roughly 32 GB
host RAM; a dense 27B on a 24 GB GPU still needs real headroom/quality testing.
The upstream [16 GB guide](https://github.com/GenerelSchwerz/llama.cpp/wiki/16GB-VRAM-Setup)
used different geometry; it is not evidence for these current defaults.

Override context/batch/ubatch/cache within launcher bounds using
`LLAMA_CONTEXT_SIZE`, `LLAMA_BATCH_SIZE`, `LLAMA_UBATCH_SIZE`,
`LLAMA_MOE_CACHE_SIZE`. Defaults were **not silently reduced to make tests pass**.
`LLAMA_THREADS=auto` uses physical cores, affinity/cpuset and visible CPU quotas.
Tuning also supports the allowlisted `/workspace/llama.cpp/runtime-settings.json`;
apply only while idle and restart the affected service deliberately.

## Provisioning and profiles

- `LLAMA_DOWNLOAD_MODELS=1` explicitly authorizes startup downloads. The example
  enables it: inspect licenses, disk/RAM and rental/download budgets first.
- Chat variants go under `/workspace/llama.cpp/models/<variant>`; an optional
  `LLAMA_MODEL_DIR` overrides this. MTP is under `MTP/`. CPU embeddings and the
  enabled Decision model are additional downloads, not included in chat sizes.
- Verified local files are reused. Missing/changed files fail closed without
  authorization. Existing destination files are not silently overwritten.
- Current model transport does **not yet implement bounded parallel/resumable
  Range downloads or router ETA×price cancellation**. Failed `.gguf.part` files
  are retained for inspection; do not mistake retention for resumable transfer.
- Codacus profile capture performs real GPU inference: opt in with
  `LLAMA_GENERATE_PROFILE=1` only on an idle, budget-approved backer. Verified
  chat/draft weights precede capture. Two synthetic traces each have
  `LLAMA_TRACE_TIMEOUT_SECONDS=3600` by default (60–7200 seconds each); timeout
  leaves staging for inspection and does not publish/retry a partial profile.
- Existing CSVs require a matching model/fork/profile receipt. Inspect mismatches
  before explicit `python3 /opt/praxis/bin/profile_llama.py --adopt-existing`
  with `LLAMA_CHAT_RECIPE=codacus`; existing data is never regenerated automatically.
  A corrected recipe receipt can require explicit re-adoption after an upgrade.

Adding a variant requires a reviewed model/revision, every file's exact size and
SHA256, explicit geometry and optional pinned draft. Rebuild, use a distinct
performance profile, and perform hardware acceptance before scoring/routing it.

## Interfaces and health

- **11434**, loopback nginx exposed only through approved NetBird ACLs:
  OpenAI chat/completions/models/embeddings, `/health`, `/health/embedding`.
- `/props` and `/metrics` are read-only GET/HEAD routes on 11434. No `--props`
  mutation flag is enabled. Metrics are enabled on all three llama servers.
- **11435** chat backend, **11436** CPU embeddings; both loopback.
- **11440** Decision backend by default, loopback, independent of chat.
  `LLAMA_DECISION_MODEL=qwen3-1.7b`, `LLAMA_DECISION_NGL=999`,
  `LLAMA_DECISION_CONTEXT=8192` are current defaults. `NGL=0` uses CPU/RAM,
  not free resources. Review ACLs and router service configuration separately.
- Container health checks enabled services, process/argv-bound readiness receipts,
  required cache activation and NetBird. The agent's configured HTTP probes are
  a **different gate**; they do not automatically inherit this full receipt check.
- No published host ports, host networking or added capabilities are required.

## Offline checks and build

```sh
python3 -m unittest discover -s tests -v
```

```sh
cp runtime.env.example runtime.env && chmod 600 runtime.env
```

Set secrets privately. The Dockerfile pins router `0.26.3` by digest and embeds
agent `0.2.1` (fatal listener errors exit nonzero for Supervisor). For a later
release, first publish/verify the matching router and pass its immutable
reference through `PGPU_IMAGE` when running `bash scripts/build.sh`.
Existing Vast contracts retain their old agent across stop/start.
`IMAGE_TAG` defaults to `praxis-llamacpp-decision:dev`.
`bash scripts/verify-image.sh` checks the built image without network, GPU,
models, credentials, host mounts or published ports.

[BASE_STACK.json](BASE_STACK.json) preserves **historical** base-file hashes and
explicit local divergences. It is not an assertion that this evolved fork is
byte-identical to its ancestor. Current behavior tests cover the overrides.
