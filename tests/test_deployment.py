import configparser
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

ROOT = Path(__file__).resolve().parents[1]

_MODULES = {}


def module(name):
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        _MODULES[name] = loaded
    return _MODULES[name]


class DeploymentTests(unittest.TestCase):
    def test_llama_proxy_keeps_private_listeners_and_routes_embeddings(self):
        proxy = (ROOT / 'scripts/llama-proxy.conf').read_text()
        for text in ['listen 127.0.0.1:11434;', 'proxy_pass http://127.0.0.1:11435;',
                     'proxy_pass http://127.0.0.1:11436;', 'location = /v1/embeddings',
                     'proxy_buffering off;', 'proxy_next_upstream off;',
                     'if ($http_origin != "") { return 403; }', 'location / { return 404; }']:
            self.assertIn(text, proxy)
        self.assertNotIn('0.0.0.0', proxy)
        docker = (ROOT / 'Dockerfile').read_text()
        self.assertNotIn('OLLAMA_', docker)
        self.assertFalse(any(l.strip().startswith('EXPOSE ') for l in docker.splitlines()))

    def test_supervisor_keeps_only_llama_services_without_download_retry_loops(self):
        c = configparser.ConfigParser(interpolation=None); c.read(ROOT / 'supervisord.conf')
        self.assertEqual(c['unix_http_server']['chmod'], '0700')
        for kind in ['chat', 'embedding']:
            setup = c['program:llama-' + kind]
            self.assertEqual(setup['autorestart'], 'false')
            self.assertEqual(setup['startsecs'], '0')
            self.assertEqual(setup['startretries'], '0')
            self.assertIn('run_llama.py ' + kind, setup['command'])
        names = ['netbird', 'llama-chat', 'llama-embedding', 'llama-proxy']
        self.assertEqual({s for s in c if s.startswith('program:')}, {'program:' + n for n in names})
        for name in names:
            self.assertEqual(c['program:' + name]['stopasgroup'], 'true')
        for name in ['comfyui', 'vosk']:
            self.assertNotIn('program:' + name, c)

    def test_build_pins_generelschwerz_moe_cache_and_16gb_recipe(self):
        source = json.loads((ROOT / 'LLAMA_CPP.json').read_text())
        docker = (ROOT / 'Dockerfile').read_text()
        self.assertEqual(source['repository'], 'https://github.com/GenerelSchwerz/llama.cpp')
        self.assertEqual(source['branch'], 'moe-cache')
        self.assertIn(source['commit'], docker)
        self.assertEqual(source['commit'], module('llama_config').REVISION)
        for text in ['-DGGML_CUDA=ON', '-DGGML_NATIVE=OFF', 'llama-server',
                     'COPY --from=llama-build /opt/llama.cpp /opt/llama.cpp']:
            self.assertIn(text, docker)
        for old in ['llama-moe-trace', 'flash-attn', 'qwen-tts-venv', 'download_qwen_model.py']:
            self.assertNotIn(old, docker)

    def test_speech_and_comfyui_are_removed_from_the_llama_fork(self):
        for path in ['custom_nodes', 'workflows', 'constraints.txt', 'docs/COMFYUI_QWEN3.md',
                     'scripts/run_vosk.py', 'scripts/vosk_server.py', 'scripts/provision_vosk.py',
                     'scripts/xtts_worker.py', 'scripts/install_workflows.py', 'scripts/smoke.py',
                     'scripts/profile_llama.py']:
            self.assertFalse((ROOT / path).exists(), path)
        env = (ROOT / 'runtime.env.example').read_text()
        for key in ['LLAMA_MODEL=qwen3.8-flash-next', 'LLAMA_DOWNLOAD_MODELS=1',
                    'LLAMA_MODEL=qwen3.6-35b-a3b', 'qwen3.6-35b-a3b']:
            self.assertIn(key, env)
        for key in ['PRAXIS_QWEN_TTS', 'VOSK_DOWNLOAD_MODELS', 'COQUI_TOS_AGREED', 'LLAMA_MTP_MODEL_DIR']:
            self.assertNotIn(key, env)

    def test_runtime_example_selects_the_default_flash_recipe_and_can_switch(self):
        config = module('llama_config')
        env = dict(line.split('=', 1) for line in (ROOT / 'runtime.env.example').read_text().splitlines() if '=' in line and not line.startswith('#'))
        self.assertEqual(env['LLAMA_THREADS'], 'auto')
        with patch.object(config, 'physical_thread_count', return_value=6) as detect:
            plan = config.settings(env)
            detect.assert_called_once_with()
        self.assertTrue(plan['download'])
        self.assertEqual((plan['variant'], plan['cache_size'], plan['threads'], plan['context'],
                          plan['batch'], plan['ubatch']), ('qwen3.8-flash-next', 56, 6, 4096, 512, 512))
        chat = config.server_command('chat', env)
        self.assertEqual(chat[chat.index('--moe-expert-cache-size') + 1], '56')
        self.assertIn('draft-mtp', chat)
        self.assertIn('Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf', ' '.join(chat))
        # Switching the model is one env var.
        switched = config.server_command('chat', dict(env, LLAMA_MODEL='qwen3.6-35b-a3b'))
        self.assertEqual(switched[switched.index('--ctx-size') + 1], '65536')
        self.assertEqual(switched[switched.index('--moe-expert-cache-size') + 1], '96')
        self.assertNotIn('--spec-type', switched)
        embedding = config.server_command('embedding', env)
        self.assertEqual(embedding[embedding.index('--ctx-size') + 1], '2048')
        self.assertNotIn('LLAMA_MTP_MODEL_DIR', env)

    def test_no_speech_deps_or_venvs_in_dockerfile(self):
        docker = (ROOT / 'Dockerfile').read_text()
        for text in ['qwen-tts-venv', 'comfy-venv', 'vosk-venv', '/opt/ComfyUI', 'aiohttp', 'coqui-tts',
                     'flash_attn', 'ffmpeg', 'espeak-ng', 'sox', 'install_workflows']:
            self.assertNotIn(text, docker)
        for text in ['netbird', 'libgomp1', 'libssl3', 'supervisor', 'llama-proxy.conf']:
            self.assertIn(text, docker)

    def test_base_stack_records_only_remaining_files_and_matches_them(self):
        base = json.loads((ROOT / 'BASE_STACK.json').read_text())
        for path, expected in base['sha256'].items():
            self.assertTrue((ROOT / path).is_file(), path)
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected, path)
        for gone in ['custom_nodes', 'workflows', 'docs/', 'constraints.txt', 'scripts/vosk_server.py']:
            self.assertFalse(any(p.startswith(gone) for p in base['sha256']), gone)

    def test_dockerfile_and_verify_image_check_the_moe_cache_flags(self):
        docker = (ROOT / 'Dockerfile').read_text()
        for flag in ['--moe-expert-cache-size', '--cache-ram', '--load-mode', '--kv-offload']:
            self.assertIn(flag, docker)
        verify = (ROOT / 'scripts/verify-image.sh').read_text()
        for text in ['py_compile /opt/praxis/bin/run_llama.py', '--moe-expert-cache-size', '--print-command']:
            self.assertIn(text, verify)


if __name__ == '__main__':
    unittest.main()