"""Readiness includes actual CUDA MoE expert-cache activation, not model quality."""
import json
import os
import subprocess
import urllib.request

from llama_config import server_command, settings
from run_llama import runtime_ready

MAX_JSON = 2 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def main():
    config = settings(os.environ)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    urls = ['http://127.0.0.1:11434/health']
    if config['embedding']:
        urls.append('http://127.0.0.1:11434/health/embedding')
    if config['decision']:
        command = server_command('decision', os.environ)
        urls.append(f'http://127.0.0.1:{command[command.index("--port") + 1]}/health')
    for url in urls:
        with opener.open(url, timeout=3) as response:
            raw = response.read(MAX_JSON + 1)
            if len(raw) > MAX_JSON:
                raise RuntimeError('Health response exceeds size limit')
            body = json.loads(raw)
        if body.get('status') != 'ok':
            raise RuntimeError('llama.cpp is not ready')
    kinds = ['chat'] + (['embedding'] if config['embedding'] else []) + (['decision'] if config['decision'] else [])
    for kind in kinds:
        if not runtime_ready(kind, os.environ):
            raise RuntimeError(f'llama.cpp {kind} readiness receipt is missing/stale; inspect service log')
    subprocess.run(['netbird', 'status', '--check', 'ready'], check=True, timeout=4,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()