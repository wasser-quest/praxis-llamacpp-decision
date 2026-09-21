#!/usr/bin/env python3
"""Capture Codacus code/chat routing profiles once, merge, and bind them to the model.

This runs GPU inference. Run only with an idle GPU. Existing profiles are never
replaced; --adopt-existing explicitly accepts a profile you know belongs to this model.
"""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from llama_config import CODACUS_BIN, CODACUS_REVISION, child_environment, model_path, models, profile_path, settings
from provision_llama import atomic_json, fingerprint, lock, provision, read_receipt

PROMPTS = {
    'code': 'Write a Python implementation of a bounded LRU cache, explain its design, '
            'and write tests for eviction, replacement, empty input and concurrent access. Be detailed.',
    'chat': 'Explain how to plan a week of language learning for a beginner learning German, French '
            'and Japanese. Discuss practice, realistic goals, common mistakes and examples in detail.',
}
MAX_PROFILE = 64 * 1024 * 1024


def validate_profile(path):
    info = fingerprint(path)
    if not 0 < info[2] <= MAX_PROFILE:
        raise RuntimeError('Routing profile is empty or exceeds 64 MiB')
    decode_layers = set()
    with path.open(newline='') as stream:
        for row in csv.reader(stream):
            if len(row) != 12:  # position + layer + 10 active experts for Flash Next
                raise RuntimeError('Expected Flash Next routing CSV with 10 active experts per row')
            try:
                position, layer, *experts = [int(v) for v in row]
            except ValueError as error:
                raise RuntimeError('Routing CSV must contain integer IDs') from error
            if not 0 <= layer < 1024 or len(set(experts)) != 10 or any(not 0 <= e < 512 for e in experts):
                raise RuntimeError('Invalid Flash Next layer/expert IDs')
            if position >= 0:
                decode_layers.add(layer)
    if not decode_layers:
        raise RuntimeError('Profile has no decode rows; Codacus would disable the cache')
    if fingerprint(path) != info:
        raise RuntimeError('Profile changed during validation')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe():
    config = settings(os.environ)
    return {'fork_revision': CODACUS_REVISION,
            'model': models()['variants'][config['variant']],
            'prompts': PROMPTS, 'tokens_per_prompt': 512}


def ensure_profile(env, generate=False, adopt=False):
    config, target = settings(env), profile_path(env)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = target.with_suffix('.json')
    with lock(target.with_suffix('.lock')):
        if target.exists() or target.is_symlink():
            digest = validate_profile(target)
            previous = read_receipt(metadata)
            if previous.get('recipe') == recipe() and previous.get('sha256') == digest:
                print('Reusing verified code/chat routing profile', flush=True)
                return target
            if not adopt:
                raise RuntimeError('Existing profile is unverified or belongs to another recipe. '
                                   'Preserved; inspect it before explicit --adopt-existing.')
            atomic_json(metadata, {'recipe': recipe(), 'sha256': digest, 'source': 'operator-adopted'})
            return target
        if adopt or not generate:
            raise RuntimeError('Missing routing profile; set LLAMA_GENERATE_PROFILE=1 for two one-time GPU traces')
        # Leave a failed staging directory for inspection, never silently retry a GPU job.
        stage = Path(tempfile.mkdtemp(prefix='flash-trace-', dir=target.parent))
        traces = []
        for name, prompt in PROMPTS.items():
            trace = stage / (name + '.csv')
            command = [str(CODACUS_BIN / 'llama-moe-trace'), '--model', str(model_path('chat', env))]
            # Original-Placement aus praxis-llamap-gpu (perf-Fork): Experten via
            # mmap auf CPU — der Trace muss die CPU-Routing-Realitaet treffen.
            command += ['--n-gpu-layers', '99', '--n-cpu-moe', '99', '--flash-attn', 'on',
                        '--fit', 'off', '--load-mode', 'mmap', '--lazy-mode', 'on',
                        '--threads', str(config['threads']), '--threads-batch', str(config['threads']),
                        '--ctx-size', '4096', '--predict', '512', '--prompt', prompt]
            child = child_environment(env); child['MOE_TRACE_OUT'] = str(trace)
            print(f'Capturing {name} routing profile (up to 512 generated tokens; '
                  f'{config["threads"]} CPU worker threads)', flush=True)
            subprocess.run(command, env=child, check=True, timeout=config['trace_timeout'])
            validate_profile(trace)
            traces.append(trace)
        merged = stage / 'merged.csv'
        with merged.open('xb') as out:
            for trace in traces:
                data = trace.read_bytes()
                out.write(data)
                if not data.endswith(b'\n'):
                    out.write(b'\n')
            out.flush(); os.fsync(out.fileno())
        digest = validate_profile(merged)
        os.link(merged, target)
        atomic_json(metadata, {'recipe': recipe(), 'sha256': digest, 'source': 'code+chat'})
        print(f'Routing profile ready: {target}; source traces retained in {stage}', flush=True)
        return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt-existing', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    provision('chat', os.environ)  # no implicit downloads in this manual tool
    ensure_profile(os.environ, generate=not args.adopt_existing, adopt=args.adopt_existing)
