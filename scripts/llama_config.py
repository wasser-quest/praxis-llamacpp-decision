"""GenerelSchwerz moe-cache settings; model choice per LLAMA_MODEL variant.

Every chat variant carries its own default geometry (context/batch/ubatch and
MoE cache size) plus optionally a separately pinned MTP draft head. Environment
overrides for the per-variant values stay bounded and fail closed.
"""
import json
import os
from pathlib import Path
import stat

from cpu_threads import physical_thread_count

BIN = Path('/opt/llama.cpp/bin')
DECISION_BIN = Path('/opt/llama.cpp-decision/bin')  # codacus parallel-decision (Jev /v1/decision)
# Codacus perf-Fork (qwen4exp-Routing-Profil-Expert-Cache, llama-moe-trace):
# LLAMA_CHAT_RECIPE=codacus laesst den Chat-Server mit dem ORIGINAL-Flash-Next-
# Rezept aus praxis-llamap-gpu laufen (moe-cache-profile/-slots statt GS-LRU).
CODACUS_BIN = Path('/opt/llama.cpp-codacus/bin')
CODACUS_REVISION = '27c54b4bbcefadedcec6397477cc2e866c1db716'
REVISION = '907a73da9a149faa8c42ccde890f1d575586810f'
MANIFEST = Path(__file__).with_name('llama-models.json')
RUNTIME_SETTINGS = Path('/workspace/llama.cpp/runtime-settings.json')
TUNING_KEYS = {'LLAMA_CONTEXT_SIZE', 'LLAMA_BATCH_SIZE', 'LLAMA_UBATCH_SIZE',
               'LLAMA_REASONING_EFFORT', 'LLAMA_THINKING_BUDGET', 'LLAMA_REASONING_FORMAT',
               'LLAMA_THINKING_BUDGET_MESSAGE', 'LLAMA_POLL',
               'LLAMA_MOE_CACHE_SIZE', 'LLAMA_MOE_CACHE_SLOTS', 'LLAMA_MOE_CACHE_ENABLED',
               'LLAMA_THREADS', 'LLAMA_DOWNLOAD_TIMEOUT_SECONDS'}


def models():
    return json.loads(MANIFEST.read_text())


def variant_name(env):
    name = str(env.get('LLAMA_MODEL', '') or models().get('default', 'qwen3.8-flash-next'))
    if name not in models()['variants']:
        raise ValueError('LLAMA_MODEL must name a pinned variant in llama-models.json: '
                         + ', '.join(sorted(models()['variants'])))
    return name


def variant(env):
    return models()['variants'][variant_name(env)]


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
        return absolute_path(env.get('LLAMA_MODEL_DIR',
                            f'/workspace/llama.cpp/models/{variant_name(env)}'))
    if kind == 'draft':
        return model_dir('chat', env) / 'MTP'
    if kind == 'embedding':
        return absolute_path(env.get('LLAMA_EMBEDDING_MODEL_DIR', '/workspace/llama.cpp/models/nomic-embed-text'))
    if kind == 'decision':
        return absolute_path(env.get('LLAMA_DECISION_MODEL_DIR',
                            f'/workspace/llama.cpp/models/decision-{decision_variant_name(env)}'))
    raise ValueError('Unknown model kind')


def decision_variant_name(env):
    name = str(env.get('LLAMA_DECISION_MODEL', '') or models().get('decision_default', 'qwen3-1.7b'))
    if name not in models().get('decisions', {}):
        raise ValueError('LLAMA_DECISION_MODEL must name a pinned decision variant: '
                         + ', '.join(sorted(models().get('decisions', {}))))
    return name


def model_spec(kind, env):
    """Pinned repository/revision/files for one servable/provisionable artifact."""
    if kind == 'decision':
        return models()['decisions'][decision_variant_name(env)]
    if kind == 'chat':
        return variant(env)
    if kind == 'draft':
        spec = variant(env).get('draft')
        if not isinstance(spec, dict):
            raise ValueError('Selected LLAMA_MODEL variant has no pinned MTP draft head')
        return spec
    if kind == 'embedding':
        return models()['embedding']
    raise ValueError('Unknown model kind')


def model_path(kind, env):
    return model_dir(kind, env) / Path(model_spec(kind, env)['files'][0]['path']).name


def profile_path(env):
    return absolute_path(env.get('LLAMA_MOE_CACHE_PROFILE',
                        '/workspace/llama.cpp/profiles/qwen3.8-flash-next-merged.csv'))


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
    recipe = variant(env)
    # Chat-Rezept (Binary + Placement): 'gs' (GenerelSchwerz LRU-Expert-Cache,
    # Default) oder 'codacus' (perf-Fork: Routing-Profil + --moe-cache-slots,
    # das ORIGINAL-Flash-Next-Setup aus praxis-llamap-gpu). Nur sinnvoll mit
    # dem 82-GB-Modell — alle Varianten laufen auf beiden Rezepten, aber nur
    # flash-next profitiert vom Profil-Cache.
    chat_recipe = str(env.get('LLAMA_CHAT_RECIPE', 'gs')).strip().lower()
    if chat_recipe not in ('gs', 'codacus'):
        raise ValueError("LLAMA_CHAT_RECIPE must be 'gs' or 'codacus'")
    # Under codacus the profile/slots/trace vars ARE the recipe (not legacy
    # conflicts); under gs they stay forbidden (silent-override guard).
    legacy = ({'LLAMA_MTP_MODEL_DIR', 'GGML_SCHED_PREFETCH_EXPERTS'} if chat_recipe == 'codacus'
              else {'LLAMA_MOE_CACHE_SLOTS', 'LLAMA_MOE_CACHE_PROFILE', 'LLAMA_MTP_MODEL_DIR',
                    'LLAMA_GENERATE_PROFILE', 'GGML_MOE_CACHE_PROFILE', 'GGML_MOE_CACHE_SLOTS',
                    'GGML_SCHED_PREFETCH_EXPERTS'})
    forbidden = [k for k, v in env.items() if v and (
        k in legacy or k.startswith('LLAMA_ARG_') or k.startswith('GGML_MOE_CACHE_') or k.startswith('OLLAMA_'))]
    if forbidden:
        raise ValueError('Remove conflicting legacy variables: ' + ', '.join(sorted(forbidden)))
    return {
        'variant': variant_name(env),
        'threads': thread_count(env),
        'context': number(env, 'LLAMA_CONTEXT_SIZE', recipe['context'], 512, 262144),
        'batch': number(env, 'LLAMA_BATCH_SIZE', recipe['batch'], 1, 8192),
        'ubatch': number(env, 'LLAMA_UBATCH_SIZE', recipe['ubatch'], 1, 8192),
        'cache_size': number(env, 'LLAMA_MOE_CACHE_SIZE', recipe['cache_size'], 0, 1024),
        'download': enabled(env, 'LLAMA_DOWNLOAD_MODELS'),
        'chat_recipe': chat_recipe,
        'moe_cache_slots': number(env, 'LLAMA_MOE_CACHE_SLOTS', 56, 1, 512),
        # Expert-Cache AUS (LLAMA_MOE_CACHE_ENABLED=0): codacus-Rezept ohne
        # --moe-cache-profile/-slots — reine Baseline-Placement (mmap/CPU),
        # kein Profil noetig, kein GPU-Cache.
        'moe_cache_enabled': enabled(env, 'LLAMA_MOE_CACHE_ENABLED', '1'),
        # Fehlendes Routing-Profil beim Start automatisch capturen (2 GPU-
        # Traces, ~1h, einmalig) statt abzusterben: nur mit diesem Flag.
        'generate_profile': enabled(env, 'LLAMA_GENERATE_PROFILE'),
        'embedding': enabled(env, 'LLAMA_EMBEDDING_ENABLED', '1'),
        'decision': enabled(env, 'LLAMA_DECISION_ENABLED', '1'),
        'download_timeout': number(env, 'LLAMA_DOWNLOAD_TIMEOUT_SECONDS', 21600, 60, 86400),
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
    # Everything in RAM, CUDA MoE expert cache on the GPU, F16 KV offloaded.
    # --fit does not account for the expert-cache pools.
    return ['--n-gpu-layers', '99', '--n-cpu-moe', '99', '--flash-attn', 'on',
            '--fit', 'off', '--load-mode', 'none', '--cache-type-k', 'f16',
            '--cache-type-v', 'f16', '--kv-offload'] + thread_args(config)


def decision_server_command(env):
    """Jev-Decision-Server (codacus parallel-decision): /v1/decision-Endpoint.

    Kleines denses Modell (Schema-Choices + Confidence, Parallel-Scoring aus
    einem gecachten Kontext). Läuft auf CPU (NGL=0) oder GPU — dieselbe Box
    wie der Chat-Server, eigener Port, eigene Binary/Backends.
    """
    name = decision_variant_name(env)
    spec = models()['decisions'][name]
    env = runtime_environment(env)
    ngl = number(env, 'LLAMA_DECISION_NGL', 999, 0, 999)
    port = number(env, 'LLAMA_DECISION_PORT', 11440, 1024, 65535)
    context = number(env, 'LLAMA_DECISION_CONTEXT', 8192, 512, 131072)
    threads = thread_count(env)
    args = [str(DECISION_BIN / 'llama-server'), '--model', str(model_path('decision', env)),
            '--alias', spec['alias'],
            '--host', '127.0.0.1', '--port', str(port),
            '--n-gpu-layers', str(ngl),
            '--ctx-size', str(context), '--jinja'] + thread_args({'threads': threads})
    return args


def server_command(kind, env):
    if kind == 'decision':
        return decision_server_command(env)
    if kind not in ('chat', 'embedding'):
        raise ValueError('Only chat, embedding and decision models can be served')
    config = settings(env)
    # Chat-Rezept waehlt die Binary: 'gs' (Default) oder 'codacus' (Original-
    # Flash-Next-Setup aus praxis-llamap-gpu: Routing-Profil-Expert-Cache).
    chat_bin = CODACUS_BIN if (kind == 'chat' and config['chat_recipe'] == 'codacus') else BIN
    args = [str(chat_bin / 'llama-server'), '--model', str(model_path(kind, env)),
            '--alias', models()['variants'][config['variant']]['alias'] if kind == 'chat' else models()['embedding']['alias'],
            '--host', '127.0.0.1',
            '--port', '11435' if kind == 'chat' else '11436',
            '--parallel', '1', '--no-slots', '--no-warmup']
    if kind == 'chat':
        if config['ubatch'] > config['batch']:
            raise ValueError('LLAMA_UBATCH_SIZE must not exceed LLAMA_BATCH_SIZE')
        if config['chat_recipe'] == 'codacus':
            # Original praxis-llamap-gpu Placement (perf-Fork, verifiziert
            # gegen common/arg.cpp@27c54b4): Experts via mmap auf CPU,
            # Profil-gesteuerter GPU-Cache statt GS-LRU, kein kv-offload.
            args += ['--n-gpu-layers', '99', '--n-cpu-moe', '99', '--flash-attn', 'on',
                     '--fit', 'off', '--load-mode', 'mmap', '--lazy-mode', 'on',
                     '--ctx-size', str(config['context']), '--batch-size', str(config['batch']),
                     '--ubatch-size', str(config['ubatch']), '--jinja', '--cache-ram', '0']
            if config['moe_cache_enabled']:
                args += ['--moe-cache-profile', str(profile_path(env)),
                         '--moe-cache-slots', str(config['moe_cache_slots'])]
            args += ['--verbosity', '4'] + thread_args(config)
        else:
            args += placement_args(config) + [
                '--ctx-size', str(config['context']), '--batch-size', str(config['batch']),
                '--ubatch-size', str(config['ubatch']), '--jinja', '--cache-ram', '0',
                '--moe-expert-cache-size', str(config['cache_size'])]
        if 'draft' in models()['variants'][config['variant']]:
            # Shared Q8 MTP head: borrows the target's embedding/output tensors.
            args += ['--spec-type', 'draft-mtp', '--spec-draft-model', str(model_path('draft', env)),
                     '--spec-draft-ngl', '99', '--spec-draft-n-max', '2',
                     '--spec-draft-threads', str(config['threads']),
                     '--spec-draft-threads-batch', str(config['threads'])]
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