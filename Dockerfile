# syntax=docker/dockerfile:1
# pgpu: Router-Image (gpu-agent-Artefakt) — globaler ARG vor dem ersten FROM.
ARG PGPU_IMAGE=vayayo/praxis-gpu-router:0.26.3@sha256:87d379069a3e75da6125144a52f56689ed0d87048d50ef27c6544a7171f1140b
FROM netbirdio/netbird:0.78.2-rootless@sha256:4a30e98bb0a98848c3b2a3ca4778cebfbe87d210f05b1f40288bb072d627c3fd AS netbird
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97 AS llama-build
ARG LLAMA_CPP_REV=907a73da9a149faa8c42ccde890f1d575586810f
ARG CUDA_DOCKER_ARCH=75;80;86;89;90;120
ARG LLAMA_BUILD_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*
# A parallel build that dies mid-way (flaky cicc segfaults — observed 3× on
# 2026-09-20, kernel: null-deref at various IPs; the identical Dockerfile
# built clean 16:02 the day before, so it is machine-state/timing dependent)
# discards the ENTIRE RUN layer. Retry incrementally instead: cmake --build
# resumes at the dead TU, and each attempt de-escalates parallelism (4→2→1)
# because the crashes cluster on the fat mma/mmq template units under 4-way
# cicc load.
RUN git init /src/llama.cpp && cd /src/llama.cpp \
    && git remote add origin https://github.com/GenerelSchwerz/llama.cpp.git \
    && git fetch --depth 1 origin "$LLAMA_CPP_REV" && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$LLAMA_CPP_REV" \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
       -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
       -DCMAKE_CUDA_ARCHITECTURES="$CUDA_DOCKER_ARCH" \
       -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
       -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && ( ok=0; \
         for attempt in 1 2 3 4 5 6; do \
           j=$(( LLAMA_BUILD_JOBS >> (attempt - 1) )); \
           [ "$j" -lt 1 ] && j=1; \
           echo "llama-server build attempt $attempt — parallel $j" >&2; \
           if cmake --build build --target llama-server --parallel "$j"; then ok=1; break; fi; \
           echo "attempt $attempt failed — incremental retry, weniger Parallelitaet" >&2; \
         done; \
         test "$ok" = 1 ) \
    && mkdir -p /opt/llama.cpp/bin \
    && cp build/bin/llama-server /opt/llama.cpp/bin/ \
    && find build -name '*.so*' -exec cp -P {} /opt/llama.cpp/bin/ \; \
    && cp LICENSE /opt/llama.cpp/LICENSE \
    && git rev-parse HEAD > /opt/llama.cpp/REVISION

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97 AS codacus-build
ARG CODACUS_LLAMA_CPP_REV=27c54b4bbcefadedcec6397477cc2e866c1db716
ARG CUDA_DOCKER_ARCH=75;80;86;89;90;120
ARG CODACUS_BUILD_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libssl-dev \
        && rm -rf /var/lib/apt/lists/*
RUN git init /src/llama.cpp && cd /src/llama.cpp \
    && git remote add origin https://github.com/thecodacus/llama.cpp.git \
    && git fetch --depth 1 origin "$CODACUS_LLAMA_CPP_REV" && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$CODACUS_LLAMA_CPP_REV" \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
       -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
       -DCMAKE_CUDA_ARCHITECTURES="$CUDA_DOCKER_ARCH" \
       -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
       -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && ( ok=0; \
         for attempt in 1 2 3 4 5 6; do \
           j=$(( CODACUS_BUILD_JOBS >> (attempt - 1) )); \
           [ "$j" -lt 1 ] && j=1; \
           echo "codacus build attempt $attempt — parallel $j" >&2; \
           if cmake --build build --target llama-server llama-moe-trace --parallel "$j"; then ok=1; break; fi; \
         done; \
         test "$ok" = 1 ) \
    && mkdir -p /opt/llama.cpp-codacus/bin \
    && cp build/bin/llama-server build/bin/llama-moe-trace /opt/llama.cpp-codacus/bin/ \
    && find build -name '*.so*' -exec cp -P {} /opt/llama.cpp-codacus/bin/ \; \
    && cp LICENSE /opt/llama.cpp-codacus/LICENSE \
    && git rev-parse HEAD > /opt/llama.cpp-codacus/REVISION

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97 AS decision-build
ARG DECISION_LLAMA_CPP_REV=14d04e755fa28653e87b9a07072892265bdc0fad
ARG CUDA_DOCKER_ARCH=75;80;86;89;90;120
ARG DECISION_BUILD_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libssl-dev \
        && rm -rf /var/lib/apt/lists/*
RUN git init /src/llama.cpp && cd /src/llama.cpp \
    && git remote add origin https://github.com/thecodacus/llama.cpp.git \
    && git fetch --depth 1 origin "$DECISION_LLAMA_CPP_REV" && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$DECISION_LLAMA_CPP_REV" \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
       -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
       -DCMAKE_CUDA_ARCHITECTURES="$CUDA_DOCKER_ARCH" \
       -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
       -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && ( ok=0; \
         for attempt in 1 2 3 4 5 6; do \
           j=$(( DECISION_BUILD_JOBS >> (attempt - 1) )); \
           [ "$j" -lt 1 ] && j=1; \
           echo "decision llama-server build attempt $attempt — parallel $j" >&2; \
           if cmake --build build --target llama-server --parallel "$j"; then ok=1; break; fi; \
         done; \
         test "$ok" = 1 ) \
    && mkdir -p /opt/llama.cpp-decision/bin \
    && cp build/bin/llama-server /opt/llama.cpp-decision/bin/ \
    && find build -name '*.so*' -exec cp -P {} /opt/llama.cpp-decision/bin/ \; \
    && cp LICENSE /opt/llama.cpp-decision/LICENSE \
    && git rev-parse HEAD > /opt/llama.cpp-decision/REVISION

# pgpu: Router-Image liefert das statische gpu-agent-Artefakt (/gpu-agent).
FROM ${PGPU_IMAGE} AS pgpu-agent-image

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    NB_USE_NETSTACK_MODE=true NB_ENABLE_NETSTACK_LOCAL_FORWARDING=true \
    NB_NETSTACK_SKIP_PROXY=true NB_DISABLE_DNS=true NB_ENABLE_CAPTURE=false \
    NB_CONFIG=/workspace/netbird/config.json NB_STATE_DIR=/workspace/netbird \
    NB_DAEMON_ADDR=unix:///workspace/netbird/netbird.sock NB_LOG_FILE=console \
    NB_ENTRYPOINT_SERVICE_TIMEOUT=60
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 ca-certificates nginx supervisor tini curl util-linux libgomp1 libssl3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=netbird /usr/local/bin/netbird /usr/local/bin/netbird
COPY --from=netbird /usr/local/bin/netbird-entrypoint.sh /usr/local/bin/netbird-entrypoint.sh
# Executables and ALL shared libraries come from the same pinned fork build.
COPY --from=llama-build /opt/llama.cpp /opt/llama.cpp
# Decision-Server (parallel-decision) + Codacus-Chat-Alternative: eigene
# Binaries + eigene Backends (Backend-DL .so mischen sich nicht).
COPY --from=decision-build /opt/llama.cpp-decision /opt/llama.cpp-decision
COPY --from=codacus-build /opt/llama.cpp-codacus /opt/llama.cpp-codacus
# No global ld.so path containing incompatible forks: launchers set the
# selected binary's own directory first in LD_LIBRARY_PATH per process.
COPY LLAMA_CPP.json /opt/praxis/LLAMA_CPP.json
COPY scripts /opt/praxis/bin
# pgpu: Router-Call-home-Agent (statisches musl-Binary aus dem Router-Image).
# --from kann keine Variablen → Alias-Stage oben definiert (ARG global).
COPY --from=pgpu-agent-image /gpu-agent /opt/praxis/bin/gpu-agent
# pgpu-Patch: netbird up --hostname $NB_HOSTNAME (sonst enrollt Vast unter
# Container-ID; Router-Connector findet gpu-<role>-<tok8> nicht).
COPY netbird-entrypoint-praxis.sh /usr/local/bin/netbird-entrypoint.sh
RUN chmod +x /usr/local/bin/netbird-entrypoint.sh /opt/praxis/bin/gpu-agent
ENV NB_SOCKS5_LISTENER_PORT=1080 PRAXIS_ROLE=llm \
    ROUTER_URL= PRAXIS_NODE_TOKEN= \
    PRAXIS_AGENT_HEALTH_URLS=http://127.0.0.1:11434/health,http://127.0.0.1:11436/health \
    PRAXIS_AGENT_SUPERVISOR_CONF=/etc/supervisor/praxis.conf \
    PRAXIS_AGENT_MODEL_DIR=/workspace/llama.cpp/models \
    PRAXIS_AGENT_ACCEPTANCE_CMD="/usr/bin/python3 /opt/praxis/bin/acceptance_llama.py --run"
COPY supervisord.conf /etc/supervisor/praxis.conf
RUN chmod +x /opt/praxis/bin/*.sh \
    && LD_LIBRARY_PATH="/opt/llama.cpp/bin:${LD_LIBRARY_PATH}" /opt/llama.cpp/bin/llama-server --help > /tmp/llama-help.txt \
    && grep -q -- '--moe-expert-cache-size' /tmp/llama-help.txt \
    && grep -q -- '--cache-ram' /tmp/llama-help.txt \
    && grep -q -- '--load-mode' /tmp/llama-help.txt \
    && grep -q -- '--kv-offload' /tmp/llama-help.txt \
    && grep -q -- '--reasoning-effort' /tmp/llama-help.txt \
    && grep -q -- '--metrics' /tmp/llama-help.txt \
    && LD_LIBRARY_PATH="/opt/llama.cpp-codacus/bin:${LD_LIBRARY_PATH}" /opt/llama.cpp-codacus/bin/llama-server --help > /tmp/codacus-help.txt \
    && grep -q -- '--moe-cache-profile' /tmp/codacus-help.txt \
    && grep -q -- '--metrics' /tmp/codacus-help.txt \
    && LD_LIBRARY_PATH="/opt/llama.cpp-decision/bin:${LD_LIBRARY_PATH}" /opt/llama.cpp-decision/bin/llama-server --help > /tmp/decision-help.txt \
    && grep -q -- '--metrics' /tmp/decision-help.txt
# Syntax check only: nginx does not start during this build step.
RUN mkdir -p /workspace/logs && nginx -t -c /opt/praxis/bin/llama-proxy.conf
# Do not EXPOSE API ports: Vast automatically publishes EXPOSE declarations.
WORKDIR /workspace
HEALTHCHECK --interval=30s --start-period=900s --timeout=25s --retries=3 \
    CMD python3 /opt/praxis/bin/healthcheck.py
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/praxis/bin/start.sh"]