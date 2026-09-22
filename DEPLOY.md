# Decision / GS alias: deployment and acceptance

This image contains three separately pinned llama.cpp backends, not only the
historical 16 GB GS recipe. See [README.md](README.md) for current model/context
sizes and [LLAMA_CPP.json](LLAMA_CPP.json) for all source pins.

## Release is not deployment

1. Run offline tests: `python3 -m unittest discover -s tests -v`.
2. Build/publish and verify the matching router/agent first. The Dockerfile pins
   router `0.26.3` by immutable digest, embedding agent `0.2.1` with corrected
   fatal listener exit status. For a later release, set `PGPU_IMAGE` to its
   verified immutable router reference. Existing Vast contracts keep their old
   agent across stop/start; publishing this image does not upgrade them.
3. `bash scripts/build.sh` builds amd64; `IMAGE_TAG` defaults to
   `praxis-llamacpp-decision:dev`. `PGPU_IMAGE` is forwarded as a build argument.
4. `bash scripts/verify-image.sh` checks the chosen image with no network, GPU,
   model download, credentials, host mounts or published ports. It checks all
   source pins, per-binary library resolution, metrics/cache flags, Supervisor,
   nginx origin guards and forbidden mutations.
5. Publish Decision and the `praxis-llamap-gs-gpu` alias from **this repository**,
   independently verify registry digests and embedded agent identity, and retain
   rollback tags. Never overwrite the alias with historical `fork-llama`.
6. Deployment restart, live rentals, profiling, benchmarks and acceptance are
   separate operator-authorized steps. Tests do not restart a deployment.

## Resource and startup planning

Select model, image digest, host RAM, GPU VRAM, local SSD, download charges and
maximum startup cost together. Flash Next's ~82 GB weight size is **not** its
VRAM requirement: GS CPU experts consume host RAM. The current 256K/128K/32K
contexts are not verified fits on an arbitrary 16/24 GB card. Also account for
MTP, CPU embeddings and the enabled Decision model. GPU name alone is not a
suitability test; upstream guide measurements used different geometry.

Weights are not baked in. `LLAMA_DOWNLOAD_MODELS=1` explicitly permits downloads;
chat, optional MTP, embeddings and Decision have separate pinned files/receipts.
Verified files are reused, invalid public files preserved rather than overwritten.
Current transport is still a single-attempt download with retained partials,
**not** a resumable/parallel mirror implementation. Automatic preparation retry,
full prepare/serve separation and router ETA×price cancellation remain open.

The Codacus Flash Next recipe needs a verified routing profile unless
`LLAMA_MOE_CACHE_ENABLED=0`. `LLAMA_GENERATE_PROFILE=1` permits two real GPU
traces after weight verification; each is bounded by
`LLAMA_TRACE_TIMEOUT_SECONDS` (default 3600, range 60–7200). Do not opt in on a
busy or unbudgeted machine. Failures leave diagnostic staging, not an automatically
published/retried profile. Existing mismatched receipts need inspection and
explicit adoption, including after a changed recipe identity.

## Private services and readiness

Supervisor programs: `netbird`, `gpu-agent`, `llama-proxy`, `llama-chat`,
`llama-embedding`, `decision-server`. All servers bind loopback. NetBird ACLs and
router services must allow only intended paths/ports; do not publish host ports.

- 11434: private nginx OpenAI API, health, read-only GET/HEAD `/props` and `/metrics`.
  Browser-origin requests and mutation requests to those introspection paths fail.
- 11435: chat backend; 11436: CPU embeddings; 11440: independent Decision backend
  by default (`LLAMA_DECISION_PORT` may change it).
- Container health checks enabled services, process identity/argv receipts,
  required cache activation and NetBird. Agent HTTP probes are independently
  configured and **do not incorporate all container receipt checks** automatically.
- Explicitly disabled cache/dense models do not require a nonexistent activation
  marker. A requested GS/Codacus cache still must actually engage.
- Model preparation and serving still share one launcher; automatic retries remain
  disabled. After fixing a failure, deliberately restart only the affected idle
  service. Do not restart a live generation for a tuning/assets change.

Inspect services with `supervisorctl -c /etc/supervisor/praxis.conf status`.
Logs live in `/workspace/logs`. Keep runtime credentials, profiles, weights and
private reports out of Git/build context. Back up persistent state privately.

## Hardware acceptance (separate authorization)

Measure cold download/load/first-token time, physical GPU/RAM/VRAM headroom,
real prefill/decode, concurrency, cancellation, long context, stop/resume and
VPN reachability on the actual allocation. A missing measurement is unknown.
`python3 /opt/praxis/bin/acceptance_llama.py --run` performs bounded real inference
and writes a private report; it is **not** part of offline tests and does not
establish all Decision endpoint, quality, lifecycle or commercial acceptance.

Praxis chat uses the router or private 11434 base URL and the actual selected
`/v1/models` alias. Configure Decision separately on its approved endpoint;
verify CPU embeddings and existing vector-index compatibility independently.
