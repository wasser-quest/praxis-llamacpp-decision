#!/usr/bin/env bash
# CPU-only image checks. No credentials, model downloads, host mounts or published ports.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
if docker info >/dev/null 2>&1; then engine=(docker); else engine=(sudo docker); sudo -v; fi
image="${IMAGE_TAG:-praxis-llamap-gpu:dev}"
"${engine[@]}" image inspect "$image" --format 'Image={{.Id}} Architecture={{.Architecture}} Size={{.Size}} ExposedPorts={{json (index .Config "ExposedPorts")}}' | tee logs/verify-image.log
"${engine[@]}" run --rm -i --network none --entrypoint /bin/bash "$image" <<'CHECKS' 2>&1 | tee -a logs/verify-image.log
set -euo pipefail
python3 -m py_compile /opt/praxis/bin/run_llama.py && echo 'launcher compile OK'
! command -v ollama
! command -v freetoken
/opt/llama.cpp/bin/llama-server --help >/tmp/llama-help
for flag in --moe-expert-cache-size --cache-ram --load-mode --kv-offload --jinja --reasoning-effort; do grep -q -- "$flag" /tmp/llama-help; done
test "$(cat /opt/llama.cpp/REVISION)" = "$(python3 -c 'import json;print(json.load(open("/opt/praxis/LLAMA_CPP.json"))["commit"])')"
nginx -t -c /opt/praxis/bin/llama-proxy.conf
netbird version
python3 /opt/praxis/bin/run_llama.py chat --print-command >/tmp/chat-command.json && python3 - <<'PY'
import json, subprocess
args = json.load(open('/tmp/chat-command.json'))
assert args[0] == '/opt/llama.cpp/bin/llama-server', args
assert '--moe-expert-cache-size' in args and args[args.index('--moe-expert-cache-size')+1] == '56'
assert '--load-mode' in args and args[args.index('--load-mode')+1] == 'none'
assert '--cache-ram' in args and args[args.index('--cache-ram')+1] == '0'
assert '--kv-offload' in args and '--cache-type-k' in args
assert '--spec-type' in args and args[args.index('--spec-type')+1] == 'draft-mtp'
assert '--moe-cache-profile' not in args and '--moe-cache-slots' not in args
print('default Flash-Next variant argument geometry OK')
PY
python3 - <<'PY'
import configparser, json
from pathlib import Path
assert Path('/opt/llama.cpp/REVISION').read_text().strip() == json.loads(Path('/opt/praxis/LLAMA_CPP.json').read_text())['commit']
c=configparser.ConfigParser(interpolation=None)
c.read('/etc/supervisor/praxis.conf')
assert {s for s in c if s.startswith('program:')} == {'program:'+p for p in ['netbird','llama-chat','llama-proxy','llama-embedding']}
for kind in ['chat','embedding']:
    assert c['program:llama-'+kind]['autorestart']=='false'
    assert c['program:llama-'+kind]['startretries']=='0'
print('Pinned fork and Supervisor definitions OK')
PY
mkdir -p /workspace/{llama.cpp,logs} /tmp/empty-models
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
# Router mode exercises the real binaries/HTTP stack without model weights or GPU.
for port in 11435 11436; do
  /opt/llama.cpp/bin/llama-server --host 127.0.0.1 --port "$port" --models-dir /tmp/empty-models --no-models-autoload --no-webui >"/tmp/llama-$port.log" 2>&1 &
  pids+=("$!")
done
nginx -c /opt/praxis/bin/llama-proxy.conf -g 'daemon off;' >/tmp/proxy.log 2>&1 &
pids+=("$!")
python3 - <<'PY'
import json,time,urllib.request,urllib.error
from pathlib import Path
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for name,url in [('llama.cpp','http://127.0.0.1:11434/v1/models'),('embedding','http://127.0.0.1:11434/health/embedding')]:
    deadline=time.monotonic()+300
    while True:
        try:
            with opener.open(urllib.request.Request(url,headers={'Host':'100.64.0.9:11434'}),timeout=3) as response:
                body=json.load(response)
            if name=='llama.cpp': assert body['data']==[], body
            if name=='embedding': assert body['status']=='ok', body
            print(name, 'CPU API check OK',flush=True)
            break
        except Exception:
            if time.monotonic()>=deadline:
                for f in ('llama-11435','llama-11436','proxy'):
                    print(Path('/tmp/'+f+'.log').read_text()[-12000:])
                raise
            time.sleep(2)
for path,headers,expected in [('/v1/models',{'Origin':'https://untrusted.example'},403),('/api/tags',{},404),('/props',{},404)]:
    try: opener.open(urllib.request.Request('http://127.0.0.1:11434'+path,headers=headers),timeout=3)
    except urllib.error.HTTPError as error: assert error.code==expected
    else: raise AssertionError('Expected private proxy rejection')
print('PASS: CPU image checks only. GPU cache/VRAM geometry, real weights and VPN still require acceptance tests.')
PY
CHECKS