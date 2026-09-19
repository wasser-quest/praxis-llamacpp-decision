"""16 GB VRAM guide settings for the GenerelSchwerz moe-cache fork; no shell-expanded extras."""
import json
import os
from pathlib import Path
import stat

from cpu_threads import physical_thread_count

BIN = Path('/opt/llama.cpp/bin')
REVISION = '907a73da9a149faa8c42ccde890f1d575586810f'
MANIFEST = Path(__file__).with_name('llama-models.json')
RUNTIME_SETTINGS = Path('/workspace/llama.cpp/runtime-settings.json')
TUNING_KEYS = {'LLAMA_CONTEXT_SIZE', 'LLAMA_BATCH_SIZE', 'LLAMA_UBATCH_SIZE',
               'LLAMA_REASONING_EFFORT', 'LLAMA_THINKING_BUDGET', 'LLAMA_REASONING_FORMAT',
               'LLAMA_THINKING_BUDGET_MESSAGE', 'LLAMA_POLL',
               'LLAMA_MOE_CACHE_SIZE', 'LLAMA_THREADS', 'LLAMA_DOWNLOAD_TIMEOUT_SECONDS'}


def models():
    return json.loads(MANIFEST.read_text())


def number(env, key, default, minimum, maximum):
    value = str(env.get(key, default))
    if not value.isascii() or not value.isdecimal() or not minimum <= int(value) <= maximum:
        raise ValueError(f'{key} must be an integer from {minimum} to {maximum}')
    return int(value)


def enabled(env, key, default='0'):
    value = env.get(key, default)
    if value not in ('0', '1'):
        raise ValueError(f'{key} must be 0 or 1')
    return value == '1'


def absolute_path(value):
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or any(ord(c) < 32 for c in str(path)):
        raise ValueError('Model directories must be absolute paths without traversal')
    return path


def model_dir(kind, env):
    if kind == 'chat':
        return absolute_path(env.get('LLAMA_MODEL_DIR', '/workspace/llama.cpp/models/qwen3.6-35b-a3b'))
    if kind == 'embedding':
        return absolute_path(env.get('LLAMA_EMBEDDING_MODEL_DIR', '/workspace/llama.cpp/models/nomic-embed-text'))
    raise ValueError('Unknown model kind')


def model_path(kind, env):
    return model_dir(kind, env) / Path(models()[kind]['files'][0]['path']).name


def thread_count(env):
    if env.get('LLAMA_THREADS', 'auto') == 'auto':
        return physical_thread_count()
    # A numeric override is deliberate and bypasses automatic topology/quota checks.
    return number(env, 'LLAMA_THREADS', None, 1, 512)


def runtime_environment(env):
    """Instance tuning shared by launchers and health checks; never source shell code.

    The optional persistent JSON overrides only bounded tuning values, not model
    identities, downloads, credentials or the private listeners. A restart is required.
    """
    result = dict(env)
    try:
        fd = os.open(RUNTIME_SETTINGS, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return result
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Runtime settings must be a regular JSON file')
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError('Runtime settings exceed 4096 bytes')
    values = json.loads(raw)
    if not isinstance(values, dict) or not set(values) <= TUNING_KEYS:
        raise ValueError('Runtime settings may contain only the documented tuning keys')
    if any(type(value) not in (str, int) for value in values.values()):
        raise ValueError('Runtime tuning values must be strings or integers')
    result.update(values)
    return result


def settings(env):
    env = runtime_environment(env)
    # Other forks' variables and the old Codacus/Flash recipe must not silently
    # override the chosen 16 GB guide recipe.
    legacy = {'LLAMA_MOE_CACHE_SLOTS', 'LLAMA_MOE_CACHE_PROFILE', 'LLAMA_MTP_MODEL_DIR',
              'LLAMA_GENERATE_PROFILE', 'GGML_SCHED_PREFETCH_EXPERTS'}
    forbidden = [k for k, v in env.items() if v and (
        k in legacy or k.startswith('LLAMA_ARG_') or k.startswith('GGML_MOE_CACHE_') or k.startswith('OLLAMA_'))]
    if forbidden:
        raise ValueError('Remove conflicting legacy variables: ' + ', '.join(sorted(forbidden)))
    return {
        'threads': thread_count(env),
        'context': number(env, 'LLAMA_CONTEXT_SIZE', 65536, 512, 262144),
        'batch': number(env, 'LLAMA_BATCH_SIZE', 8192, 1, 8192),
        'ubatch': number(env, 'LLAMA_UBATCH_SIZE', 8192, 1, 8192),
        'cache_size': number(env, 'LLAMA_MOE_CACHE_SIZE', 96, 0, 1024),
        'download': enabled(env, 'LLAMA_DOWNLOAD_MODELS'),
        'download_timeout': number(env, 'LLAMA_DOWNLOAD_TIMEOUT_SECONDS', 21600, 60, 86400),
        'embedding': enabled(env, 'LLAMA_EMBEDDING_ENABLED', '1'),
        'reasoning_effort': (str(env['LLAMA_REASONING_EFFORT'])
                             if env.get('LLAMA_REASONING_EFFORT') in ('minimal', 'low', 'medium', 'high', 'max', 'xhigh')
                             else None),
        'thinking_budget': int(env['LLAMA_THINKING_BUDGET']) if str(env.get('LLAMA_THINKING_BUDGET', '')).lstrip('-').isdigit() else None,
        'reasoning_format': (str(env['LLAMA_REASONING_FORMAT'])
                             if env.get('LLAMA_REASONING_FORMAT') in ('none', 'deepseek', 'deepseek-legacy')
                             else None),
        'thinking_budget_message': (str(env['LLAMA_THINKING_BUDGET_MESSAGE'])
                                    if env.get('LLAMA_THINKING_BUDGET_MESSAGE') else None),
        'poll': int(env['LLAMA_POLL']) if str(env.get('LLAMA_POLL', '')).lstrip('-').isdigit() else None,
    }


def child_environment(env):
    settings(env)
    child = {key: str(value) for key, value in runtime_environment(env).items()}
    # Host registration is presence-based in this revision: setting it to "0"
    # would STILL enable it. Translate our boolean to presence/absence correctly.
    if enabled(env, 'GGML_CUDA_REGISTER_HOST', '1'):
        child['GGML_CUDA_REGISTER_HOST'] = '1'
    else:
        child.pop('GGML_CUDA_REGISTER_HOST', None)
    return child


def thread_args(config):
    return ['--threads', str(config['threads']), '--threads-batch', str(config['threads'])]


def placement_args(config):
    # 16 GB guide geometry: everything in RAM, CUDA MoE expert cache on the GPU,
    # F16 KV offloaded. --fit does not account for the expert-cache pools.
    return ['--n-gpu-layers', '99', '--n-cpu-moe', '99', '--flash-attn', 'on',
            '--fit', 'off', '--load-mode', 'none', '--cache-type-k', 'f16',
            '--cache-type-v', 'f16', '--kv-offload'] + thread_args(config)


def server_command(kind, env):
    if kind not in ('chat', 'embedding'):
        raise ValueError('Only chat and embedding models can be served')
    config = settings(env)
    args = [str(BIN / 'llama-server'), '--model', str(model_path(kind, env)),
            '--alias', models()[kind]['alias'], '--host', '127.0.0.1',
            '--port', '11435' if kind == 'chat' else '11436',
            '--parallel', '1', '--no-slots', '--no-warmup']
    if kind == 'chat':
        if config['ubatch'] > config['batch']:
            raise ValueError('LLAMA_UBATCH_SIZE must not exceed LLAMA_BATCH_SIZE')
        args += placement_args(config) + [
            '--ctx-size', str(config['context']), '--batch-size', str(config['batch']),
            '--ubatch-size', str(config['ubatch']), '--jinja', '--cache-ram', '0',
            '--moe-expert-cache-size', str(config['cache_size'])]
        if config.get('reasoning_effort'):
            args += ['--reasoning-effort', str(config['reasoning_effort'])]
        if config.get('thinking_budget') is not None:
            args += ['--reasoning-budget', str(config['thinking_budget'])]
        if config.get('reasoning_format'):
            args += ['--reasoning-format', str(config['reasoning_format'])]
        if config.get('thinking_budget_message'):
            args += ['--reasoning-budget-message', str(config['thinking_budget_message'])]
        if config.get('poll') is not None:
            args += ['--poll', str(config['poll'])]
    else:
        # Keep the existing Nomic capability on CPU, without taking VRAM from the cache.
        args += ['--embedding', '--pooling', 'mean', '--n-gpu-layers', '0',
                 '--device', 'none', '--fit', 'off', '--ctx-size', '2048',
                 '--batch-size', '2048', '--ubatch-size', '2048'] + thread_args(config)
    return args