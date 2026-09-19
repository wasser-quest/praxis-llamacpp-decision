#!/usr/bin/env bash
set -euo pipefail
umask 077
mkdir -p /workspace/{netbird,llama.cpp,huggingface,logs}
chmod 700 /workspace/netbird
# A repeated Vast on-start hook must not launch a second stack.
[[ ! -L /workspace/.praxis-stack.lock ]] || { echo 'Refusing symlink startup lock' >&2; exit 1; }
exec 9>/workspace/.praxis-stack.lock
flock -n 9 || { echo 'Praxis llama.cpp GPU stack is already running'; exit 0; }
if [[ -z "${NB_SETUP_KEY:-}" && ! -s /workspace/netbird/config.json ]]; then
  echo 'Set NB_SETUP_KEY privately at runtime to enroll NetBird.' >&2
  exit 1
fi
# No TUN, extra capabilities, public listeners, or host networking required.
# NetBird forwards allowed inbound overlay connections to local listeners.
exec /usr/bin/supervisord -n -c /etc/supervisor/praxis.conf