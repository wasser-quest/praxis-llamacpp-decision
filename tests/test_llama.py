import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import llama_config as config
import provision_llama as provision
import run_llama as runner


class ConfigTests(unittest.TestCase):
    @patch.object(config, 'physical_thread_count', return_value=6)
    def test_default_flash_next_variant_uses_mtp_draft_and_original_geometry(self, detect):
        args = config.server_command('chat', {})
        detect.assert_called_once_with()
        for key, value in [('--threads', '6'), ('--threads-batch', '6'),
                           ('--spec-draft-threads', '6'), ('--spec-draft-threads-batch', '6'),
                           ('--n-cpu-moe', '99'), ('--n-gpu-layers', '99'),
                           ('--flash-attn', 'on'), ('--fit', 'off'), ('--load-mode', 'none'),
                           ('--cache-type-k', 'f16'), ('--cache-type-v', 'f16'),
                           ('--ctx-size', '4096'), ('--batch-size', '512'), ('--ubatch-size', '512'),
                           ('--cache-ram', '0'), ('--moe-expert-cache-size', '56'),
                           ('--spec-type', 'draft-mtp'), ('--spec-draft-ngl', '99'), ('--spec-draft-n-max', '2'),
                           ('--host', '127.0.0.1'), ('--port', '11435'), ('--parallel', '1')]:
            self.assertEqual(args[args.index(key) + 1], value, key)
        for key in ['--kv-offload', '--jinja', '--no-slots', '--no-warmup']:
            self.assertIn(key, args)
        for key in ['--moe-cache-profile', '--moe-cache-slots', '--verbosity', '--lazy-mode']:
            self.assertNotIn(key, args)
        self.assertEqual(args[args.index('--spec-draft-model') + 1],
                         '/workspace/llama.cpp/models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf')
        self.assertIn('Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf', ' '.join(args))

    @patch.object(config, 'physical_thread_count', return_value=6)
    def test_qwen36_16gb_variant_switches_geometry_and_drops_draft(self, detect):
        env = {'LLAMA_MODEL': 'qwen3.6-35b-a3b'}
        args = config.server_command('chat', env)
        detect.assert_called_once_with()
        for key, value in [('--ctx-size', '65536'), ('--batch-size', '8192'), ('--ubatch-size', '8192'),
                           ('--moe-expert-cache-size', '96')]:
            self.assertEqual(args[args.index(key) + 1], value, key)
        for key in ['--spec-type', '--spec-draft-model', '--spec-draft-ngl']:
            self.assertNotIn(key, args)
        self.assertIn('Qwen3.6-35B-A3B-Q4_K_M.gguf', ' '.join(args))
        self.assertEqual(config.model_dir('chat', env),
                         Path('/workspace/llama.cpp/models/qwen3.6-35b-a3b'))

    def test_unknown_variant_is_rejected(self):
        for env in [{'LLAMA_MODEL': 'gpt-5'}, {'LLAMA_MODEL_DIR': 'relative'},
                    {'LLAMA_MODEL_DIR': '/tmp/../secret'}]:
            with self.subTest(env=env), self.assertRaises(ValueError):
                config.server_command('chat', env)

    def test_cache_size_is_tunable_and_disable(self):
        args = config.server_command('chat', {'LLAMA_MOE_CACHE_SIZE': '48'})
        self.assertEqual(args[args.index('--moe-expert-cache-size') + 1], '48')
        args = config.server_command('chat', {'LLAMA_MOE_CACHE_SIZE': '0'})
        self.assertEqual(args[args.index('--moe-expert-cache-size') + 1], '0')
        with self.assertRaises(ValueError):
            config.server_command('mtp', {})

    def test_auto_and_unset_apply_detected_count_to_all_cpu_worker_pools(self):
        with patch.object(config, 'physical_thread_count', return_value=12):
            for env in [{}, {'LLAMA_THREADS': 'auto'}]:
                for kind in ['chat', 'embedding']:
                    args = config.server_command(kind, env)
                    keys = ['--threads', '--threads-batch']
                    if kind == 'chat':
                        keys += ['--spec-draft-threads', '--spec-draft-threads-batch']
                    for key in keys:
                        self.assertEqual(args[args.index(key) + 1], '12')

    def test_numeric_thread_override_does_not_require_topology(self):
        with patch.object(config, 'physical_thread_count', side_effect=AssertionError('must not detect')) as detect:
            for kind in ['chat', 'embedding']:
                args = config.server_command(kind, {'LLAMA_THREADS': '7'})
                self.assertEqual(args[args.index('--threads') + 1], '7')
                self.assertEqual(args[args.index('--threads-batch') + 1], '7')
            detect.assert_not_called()

    def test_boolean_zero_removes_presence_based_host_pinning(self):
        self.assertEqual(config.child_environment({})['GGML_CUDA_REGISTER_HOST'], '1')
        child = config.child_environment({'GGML_CUDA_REGISTER_HOST': '0'})
        self.assertNotIn('GGML_CUDA_REGISTER_HOST', child)

    def test_invalid_and_legacy_settings_rejected(self):
        for env in [{'LLAMA_THREADS': '0'}, {'LLAMA_THREADS': '1; echo nope'},
                    {'LLAMA_THREADS': ''}, {'LLAMA_THREADS': '513'}, {'LLAMA_THREADS': 'AUTO'},
                    {'LLAMA_MOE_CACHE_SIZE': '1025'}, {'LLAMA_DOWNLOAD_MODELS': 'yes'},
                    {'LLAMA_MODEL_DIR': '/tmp/../secret'},
                    # Old Codacus/Flash recipe variables must fail closed, never silently drift.
                    {'LLAMA_MOE_CACHE_SLOTS': '56'}, {'LLAMA_MOE_CACHE_PROFILE': 'merged.csv'},
                    {'LLAMA_MTP_MODEL_DIR': '/models/draft'}, {'LLAMA_GENERATE_PROFILE': '1'},
                    {'GGML_SCHED_PREFETCH_EXPERTS': '1'},
                    {'OLLAMA_MODEL': 'qwen3.8:27b'}, {'LLAMA_ARG_HOST': '0.0.0.0'},
                    {'GGML_MOE_CACHE_SLOTS': '112'}]:
            with self.subTest(env=env), self.assertRaises(ValueError):
                config.server_command('chat', env)

    def test_embedding_keeps_own_context_without_cache_or_drafting(self):
        args = config.server_command('embedding', {'LLAMA_CONTEXT_SIZE': '262144'})
        for key, value in [('--ctx-size', '2048'), ('--port', '11436'), ('--device', 'none'), ('--pooling', 'mean')]:
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn('--embedding', args)
        self.assertNotIn('--moe-expert-cache-size', args)
        self.assertNotIn('--spec-draft-model', args)

    def test_manifest_pins_both_variants_and_independent_nomic_weights(self):
        manifest = config.models()
        for name in ('qwen3.8-flash-next', 'qwen3.6-35b-a3b'):
            self.assertIn(name, manifest['variants'])
        flash = manifest['variants']['qwen3.8-flash-next']
        self.assertEqual(len(flash['files']), 3)
        self.assertEqual(sum(f['size'] for f in flash['files']), 81961823936)
        self.assertEqual(flash['draft']['revision'], flash['revision'])
        self.assertEqual(flash['draft']['files'][0]['size'], 2786568256)
        self.assertEqual(flash['draft']['files'][0]['sha256'],
                         '5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6')
        qwen36 = manifest['variants']['qwen3.6-35b-a3b']
        self.assertEqual(len(qwen36['files']), 1)
        self.assertEqual(qwen36['files'][0]['size'], 20419565568)
        self.assertEqual(qwen36['files'][0]['sha256'],
                         '671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7')
        self.assertNotIn('draft', qwen36)
        for name, variant in manifest['variants'].items():
            self.assertRegex(variant['revision'], r'^[0-9a-f]{40}$')
            for entry in variant['files']:
                self.assertRegex(entry['sha256'], r'^[0-9a-f]{64}$')
        self.assertRegex(manifest['embedding']['revision'], r'^[0-9a-f]{40}$')


class InstanceTuningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'runtime-settings.json'
        patcher = patch.object(config, 'RUNTIME_SETTINGS', self.path)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_persistent_cache_tuning_overrides_supervisor_env_and_keeps_embeddings_separate(self):
        env = {'LLAMA_MOE_CACHE_SIZE': '56'}
        self.assertEqual(config.settings(env)['cache_size'], 56)
        self.path.write_text('{"LLAMA_MOE_CACHE_SIZE":80}')
        self.assertEqual(config.settings(env)['cache_size'], 80)
        self.assertEqual(config.child_environment(env)['LLAMA_MOE_CACHE_SIZE'], '80')
        chat = config.server_command('chat', env)
        embedding = config.server_command('embedding', env)
        self.assertEqual(chat[chat.index('--moe-expert-cache-size') + 1], '80')
        self.assertEqual(embedding[embedding.index('--ctx-size') + 1], '2048')
        self.assertEqual(env, {'LLAMA_MOE_CACHE_SIZE': '56'})
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'STATE_DIR', Path(d)):
            receipt = {'pid': os.getpid(), 'process': runner.process_identity(os.getpid()),
                       'command': runner.command_digest(chat), 'cache_active': True}
            (Path(d) / 'chat.json').write_text(json.dumps(receipt))
            self.assertTrue(runner.runtime_ready('chat', env))
            self.path.write_text('{"LLAMA_MOE_CACHE_SIZE":48}')
            self.assertFalse(runner.runtime_ready('chat', env))

    def test_unknown_keys_unsafe_values_and_symlinks_fail_closed(self):
        for value in ['{"NB_SETUP_KEY":"not-a-key"}', '{"LLAMA_DOWNLOAD_MODELS":1}',
                      '{"LLAMA_MOE_CACHE_SIZE":true}', '{"LLAMA_MOE_CACHE_SIZE":-1}',
                      '{"LLAMA_MOE_CACHE_SIZE":1025}', '{"LLAMA_MOE_CACHE_SIZE":[]}',
                      '{"LLAMA_THREADS":"1; touch /tmp/not-executed"}', '[]', 'invalid', ' ' * 4097]:
            self.path.write_text(value)
            with self.subTest(value=value[:80]), self.assertRaises((ValueError, OSError)):
                config.settings({})
        self.path.unlink()
        target = self.path.with_name('other.json'); target.write_text('{}')
        self.path.symlink_to(target)
        with self.assertRaises(OSError):
            config.settings({})


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.env = {'LLAMA_MODEL_DIR': str(self.directory)}
        self.data = b'GGUFtest-fixture'
        self.entry = {'path': 'Qwen3.6-35B-A3B-Q4_K_M.gguf', 'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        self.model = {'repository': 'owner/model', 'revision': 'a'*40, 'files': [self.entry]}
        self.target = self.directory / self.entry['path']
        self.patcher = patch.object(provision, 'model_spec', return_value=self.model)
        self.patcher.start(); self.addCleanup(self.patcher.stop)

    def response(self, data=None):
        result = io.BytesIO(self.data if data is None else data)
        result.status = 200
        result.headers = {'Content-Length': str(len(self.data))}
        return result

    def test_download_is_opt_in(self):
        with patch.object(provision.OPENER, 'open') as request:
            with self.assertRaisesRegex(RuntimeError, 'Missing pinned'):
                provision.provision('chat', self.env)
            request.assert_not_called()

    def test_first_download_pinned_and_second_start_reuses_verified_files(self):
        with patch.object(provision.OPENER, 'open', return_value=self.response()) as request:
            provision.provision('chat', self.env, download=True)
            self.assertEqual(self.target.read_bytes(), self.data)
            self.assertIn('/resolve/' + 'a'*40 + '/Qwen3.6-35B-A3B-Q4_K_M.gguf', request.call_args.args[0])
        with patch.object(provision.OPENER, 'open') as request:
            provision.provision('chat', self.env, download=True)
            request.assert_not_called()
        self.assertFalse(self.target.with_suffix('.gguf.part').exists())

    def test_preexisting_weights_are_verified_without_network(self):
        self.target.write_bytes(self.data)
        with patch.object(provision.OPENER, 'open') as request:
            provision.provision('chat', self.env, download=True)
            request.assert_not_called()

    def test_modified_cached_file_and_symlink_are_not_replaced(self):
        self.target.write_bytes(self.data)
        provision.provision('chat', self.env)
        self.target.write_bytes(b'X' * len(self.data))
        with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
            provision.provision('chat', self.env, download=True)
        self.assertEqual(self.target.read_bytes(), b'X' * len(self.data))
        self.target.unlink(); self.target.symlink_to('/etc/passwd')
        with self.assertRaisesRegex(RuntimeError, 'regular file'):
            provision.provision('chat', self.env, download=True)

    def test_failed_download_retained_but_not_published_or_retried(self):
        with patch.object(provision.OPENER, 'open', return_value=self.response(b'X' * len(self.data))) as request:
            with self.assertRaisesRegex(RuntimeError, 'integrity'):
                provision.provision('chat', self.env, download=True)
            self.assertEqual(request.call_count, 1)
        self.assertFalse(self.target.exists())
        self.assertTrue(self.target.with_suffix('.gguf.part').exists())
        with patch.object(provision.OPENER, 'open') as request:
            with self.assertRaisesRegex(RuntimeError, 'interrupted download'):
                provision.provision('chat', self.env, download=True)
            request.assert_not_called()

    def test_short_oversized_and_timed_out_downloads_fail_closed(self):
        for data, timeout in [(b'short', 60), (self.data + b'extra', 60), (self.data, -1)]:
            with self.subTest(data=data, timeout=timeout):
                self.target.with_suffix('.gguf.part').unlink(missing_ok=True)
                with patch.object(provision.OPENER, 'open', return_value=self.response(data)):
                    with self.assertRaises((RuntimeError, TimeoutError)):
                        provision.download_file(self.model, self.entry, self.target, time.monotonic() + timeout)
                self.assertFalse(self.target.exists())

    def test_full_disk_and_lock_collision_do_not_download(self):
        with patch.object(provision.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 0})()):
            with self.assertRaisesRegex(RuntimeError, 'Insufficient'):
                provision.provision('chat', self.env, download=True)
        with provision.lock(self.directory / '.provision.lock'):
            with self.assertRaises(BlockingIOError):
                provision.provision('chat', self.env, download=True)


class RuntimeTests(unittest.TestCase):
    def test_cache_activation_requires_the_moe_cache_pool_line(self):
        self.assertTrue(runner.cache_active('moe-cache: CUDA_MoE_Cache_Pool[blk.0.ffn_up_chexps] = 42.00 MiB  (56 slots x 0.44 MiB)'))
        self.assertFalse(runner.cache_active('model loaded; no cache message'))
        self.assertFalse(runner.cache_active('expert cache: 40 layers x 56 slots, 8164.00 MiB uploaded to CUDA0'))

    def test_no_cuda_refuses_cpu_chat_fallback(self):
        with patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'Available devices:\n', '')):
            with self.assertRaisesRegex(RuntimeError, 'No CUDA device'): runner.require_cuda({})
        with patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'Available devices:\n  CUDA0: NVIDIA RTX 4060 Ti\n', '')):
            runner.require_cuda({})

    def test_startup_env_drives_download_and_launch_with_draft(self):
        from unittest.mock import MagicMock
        child = MagicMock()
        child.pid = os.getpid()
        child.stdout = io.StringIO(
            'load_model: loading model\n'
            'moe-cache: CUDA_MoE_Cache_Pool[blk.0.ffn_up_chexps] = 42.00 MiB  (56 slots x 0.44 MiB)\n'
            "common_speculative_init_result: loading draft model '/models/mtp.gguf'\n")
        child.wait.return_value = 0
        child.poll.return_value = 0
        env = {'LLAMA_DOWNLOAD_MODELS': '1'}
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'STATE_DIR', Path(d)), \
                patch.object(runner, 'require_cuda') as cuda, \
                patch.object(runner, 'provision') as download, \
                patch.object(runner.subprocess, 'Popen', return_value=child) as launch:
            self.assertEqual(runner.serve('chat', env), 0)
            cuda.assert_called_once_with(env)
            self.assertEqual(download.call_args_list,
                             [call('chat', env, download=True), call('draft', env, download=True)])
            self.assertIn('--spec-type', launch.call_args.args[0])
            self.assertIn('draft-mtp', launch.call_args.args[0])
            self.assertIn('--moe-expert-cache-size', launch.call_args.args[0])
            self.assertEqual(launch.call_args.kwargs['env']['GGML_CUDA_REGISTER_HOST'], '1')
            self.assertFalse((Path(d) / 'chat.json').exists())

    def test_draftless_variant_downloads_no_draft(self):
        from unittest.mock import MagicMock
        child = MagicMock()
        child.pid = os.getpid()
        child.stdout = io.StringIO(
            'moe-cache: CUDA_MoE_Cache_Pool[blk.0.ffn_up_chexps] = 42.00 MiB  (96 slots x 0.44 MiB)\n')
        child.wait.return_value = 0
        child.poll.return_value = 0
        env = {'LLAMA_DOWNLOAD_MODELS': '1', 'LLAMA_MODEL': 'qwen3.6-35b-a3b'}
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'STATE_DIR', Path(d)), \
                patch.object(runner, 'require_cuda'), \
                patch.object(runner, 'provision') as download, \
                patch.object(runner.subprocess, 'Popen', return_value=child) as launch:
            self.assertEqual(runner.serve('chat', env), 0)
            self.assertEqual(download.call_args_list, [call('chat', env, download=True)])
            self.assertNotIn('--spec-type', launch.call_args.args[0])

    def test_missing_cache_activation_fails_closed(self):
        from unittest.mock import MagicMock
        child = MagicMock(pid=os.getpid())
        child.stdout = io.StringIO('load_model: loading model\nmodel loaded\n')
        child.poll.return_value = None
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'STATE_DIR', Path(d)), \
                patch.object(runner, 'require_cuda'), patch.object(runner, 'provision'), \
                patch.object(runner.subprocess, 'Popen', return_value=child):
            with self.assertRaisesRegex(RuntimeError, 'No CUDA MoE expert-cache activation'):
                runner.serve('chat', {})
            child.terminate.assert_called_once()
            self.assertFalse((Path(d) / 'chat.json').exists())

    def test_draft_does_not_mask_missing_main_cache_activation(self):
        from unittest.mock import MagicMock
        child = MagicMock(pid=os.getpid())
        child.stdout = io.StringIO("common_speculative_init_result: loading draft model '/models/mtp.gguf'\n")
        child.poll.return_value = None
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'STATE_DIR', Path(d)), \
                patch.object(runner, 'require_cuda'), patch.object(runner, 'provision'), \
                patch.object(runner.subprocess, 'Popen', return_value=child):
            with self.assertRaisesRegex(RuntimeError, 'not activated before MTP drafting'):
                runner.serve('chat', {})
            child.terminate.assert_called_once()
            self.assertFalse((Path(d) / 'chat.json').exists())

    def test_missing_model_fails_before_server_without_fallback(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'STATE_DIR', Path(d)), \
                patch.object(runner, 'require_cuda'), \
                patch.object(runner, 'provision', side_effect=RuntimeError('Missing pinned GGUFs')) as download, \
                patch.object(runner.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'Missing pinned GGUFs'):
                runner.serve('chat', {})
            self.assertEqual(download.call_args_list, [call('chat', {}, download=False)])
            launch.assert_not_called()

    def test_disabled_embedding_does_not_download_or_launch(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'STATE_DIR', Path(d)), \
                patch.object(runner, 'provision') as download, patch.object(runner.subprocess, 'Popen') as launch:
            self.assertEqual(runner.serve('embedding', {'LLAMA_EMBEDDING_ENABLED': '0'}), 0)
            download.assert_not_called(); launch.assert_not_called()

    def test_readiness_is_bound_to_running_process_and_recipe(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'STATE_DIR', Path(d)):
            self.assertFalse(runner.runtime_ready('chat', {}))
            receipt = {'pid': os.getpid(), 'process': runner.process_identity(os.getpid()),
                       'command': runner.command_digest(config.server_command('chat', {})), 'cache_active': True}
            path = Path(d) / 'chat.json'; path.write_text(json.dumps(receipt))
            self.assertTrue(runner.runtime_ready('chat', {}))
            with patch.object(config, 'physical_thread_count', return_value=config.thread_count({}) % 512 + 1):
                self.assertFalse(runner.runtime_ready('chat', {}))
            self.assertFalse(runner.runtime_ready('chat', {'LLAMA_MODEL': 'qwen3.6-35b-a3b'}))
            receipt['cache_active'] = False; path.write_text(json.dumps(receipt))
            self.assertFalse(runner.runtime_ready('chat', {}))
            receipt['cache_active'] = True; receipt['process'] = ['stale', 'pid']; path.write_text(json.dumps(receipt))
            self.assertFalse(runner.runtime_ready('chat', {}))


if __name__ == '__main__':
    unittest.main()