#!/usr/bin/env python3
"""One startup attempt: verify/download, then supervise the pinned llama-server.

Preparation failures are not restart-looped. The 16 GB recipe treats a missing
MoE expert-cache activation as fatal, not as a quietly accepted baseline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from llama_config import BIN, child_environment, model_spec, runtime_environment, server_command, settings
from provision_llama import atomic_json, provision, read_receipt

STATE_DIR = Path('/workspace/llama.cpp/run')
# GenerelSchwerz moe-cache prints one pool line per installed cached expert tensor:
#   moe-cache: CUDA_MoE_Cache_Pool[<tensor>] = <MiB>  (<slots> slots x <MiB>)
CACHE_LINE = re.compile(r'moe-cache: CUDA_MoE_Cache_Pool\[')


def command_digest(command):
    return hashlib.sha256(json.dumps(command).encode()).hexdigest()


def process_identity(pid):
    # starttime prevents a recycled PID from satisfying an old readiness receipt.
    text = Path(f'/proc/{pid}/stat').read_text()
    return [Path('/proc/sys/kernel/random/boot_id').read_text().strip(), text.rsplit(')', 1)[1].split()[19]]


def cache_active(line):
    return bool(CACHE_LINE.search(line))


def runtime_ready(kind, env):
    receipt = read_receipt(STATE_DIR / (kind + '.json'))
    try:
        return (receipt.get('command') == command_digest(server_command(kind, env))
                and receipt.get('process') == process_identity(int(receipt['pid']))
                and (kind != 'chat' or receipt.get('cache_active') is True))
    except (OSError, KeyError, TypeError, ValueError):
        return False


def require_cuda(env):
    result = subprocess.run([str(BIN / 'llama-server'), '--list-devices'], env=child_environment(env),
                            capture_output=True, text=True, check=True, timeout=60)
    if not re.search(r'^\s*CUDA\d+:', result.stdout, re.MULTILINE):
        raise RuntimeError('No CUDA device reported by the fork binary; refusing CPU-only chat fallback')


def serve(kind, env):
    config = settings(env)
    command = server_command(kind, env)  # validate before downloads/inference
    has_draft = 'draft' in model_spec('chat', env)
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    state = STATE_DIR / (kind + '.json')
    state.unlink(missing_ok=True)
    if kind == 'embedding' and not config['embedding']:
        print('llama.cpp embedding service explicitly disabled', flush=True)
        return 0
    mode = 'auto: physical cores, affinity and CPU quota' if runtime_environment(env).get('LLAMA_THREADS', 'auto') == 'auto' else 'manual override'
    print(f'llama.cpp {kind}: {config["threads"]} CPU worker threads ({mode})', flush=True)
    if kind == 'chat':
        require_cuda(env)
    provision(kind, env, download=config['download'])
    if kind == 'chat' and has_draft:
        provision('draft', env, download=config['download'])
    child = child_environment(env)
    if kind == 'embedding':
        child.pop('GGML_CUDA_REGISTER_HOST', None)
    process = subprocess.Popen(command, env=child, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors='replace', bufsize=1)
    previous_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, lambda signum, frame: process.send_signal(signum))
    receipt = {'pid': process.pid, 'process': process_identity(process.pid),
               'command': command_digest(command), 'cache_active': False}
    try:
        if kind == 'embedding':
            atomic_json(state, receipt)
        for line in process.stdout:
            print(line, end='', flush=True)
            if kind == 'chat' and cache_active(line) and not receipt['cache_active']:
                receipt['cache_active'] = True
                atomic_json(state, receipt)
            if (kind == 'chat' and has_draft
                    and 'common_speculative_init_result: loading draft model ' in line
                    and not receipt['cache_active']):
                raise RuntimeError('Main model CUDA expert cache was not activated before MTP drafting')
        code = process.wait()
        if kind == 'chat' and not receipt['cache_active']:
            raise RuntimeError('No CUDA MoE expert-cache activation was reported')
        return code
    finally:
        state.unlink(missing_ok=True)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        process.stdout.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['chat', 'embedding'])
    parser.add_argument('--print-command', action='store_true', help='Print JSON argv; no download or inference')
    args = parser.parse_args()
    os.umask(0o077)
    if args.print_command:
        print(json.dumps(server_command(args.kind, os.environ), indent=2))
    else:
        try:
            sys.exit(serve(args.kind, os.environ))
        except Exception as error:
            print(f'llama.cpp startup failed: {error}', file=sys.stderr, flush=True)
            sys.exit(1)