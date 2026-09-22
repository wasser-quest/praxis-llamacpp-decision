#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
image="${IMAGE_TAG:-praxis-llamacpp-decision:dev}"
agent_args=()
if [[ -n "${PGPU_IMAGE:-}" ]]; then agent_args+=(--build-arg "PGPU_IMAGE=$PGPU_IMAGE"); fi
if docker info >/dev/null 2>&1; then
  engine=(docker)
else
  echo 'Docker requires sudo. Enter your password in this terminal, never in chat.'
  engine=(sudo docker)
  sudo -v
fi
"${engine[@]}" build --platform linux/amd64 --progress=plain \
  --build-arg "CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH:-75;80;86;89;90;120}" \
  --build-arg "LLAMA_BUILD_JOBS=${LLAMA_BUILD_JOBS:-4}" \
  "${agent_args[@]}" -t "$image" . 2>&1 | tee logs/build.log
"${engine[@]}" image inspect "$image" --format '{{.Id}}'
echo 'Build complete. GPU, NetBird peer access, and model smoke tests still required.'
