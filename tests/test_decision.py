"""Offline contracts for the three backends; never download or run inference."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import healthcheck
import llama_config as config
import profile_llama as profile
import run_llama as runner


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(config, 'RUNTIME_SETTINGS', self.root / 'settings.json'))
        self.enterContext(patch.object(runner, 'STATE_DIR', self.root / 'run'))
        # A missing mock must fail, not reach a real GPU, daemon or network.
        self.enterContext(patch.object(subprocess, 'run', side_effect=AssertionError('unmocked subprocess')))
        self.enterContext(patch.object(subprocess, 'Popen', side_effect=AssertionError('unmocked process')))
        self.env = {'LLAMA_CHAT_RECIPE': 'codacus', 'LLAMA_THREADS': '2',
                    'LLAMA_MOE_CACHE_PROFILE': str(self.root / 'profile.csv')}

    def child(self, text=''):
        child = MagicMock(pid=os.getpid())
        child.stdout = io.StringIO(text)
        child.poll.return_value = 0
        child.wait.return_value = 0
        return child

    def receipt(self, kind, env, cache_active=False):
        path = runner.STATE_DIR / (kind + '.json')
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({'pid': os.getpid(), 'process': runner.process_identity(os.getpid()),
            'command': runner.command_digest(config.server_command(kind, env)), 'cache_active': cache_active}))
        return path

    def csv(self, path):
        path.write_text('0,0,0,1,2,3,4,5,6,7,8,9\n')

    def test_codacus_flags_and_profile_do_not_leak_into_gs(self):
        args = config.server_command('chat', self.env)
        self.assertEqual(args[0], str(config.CODACUS_BIN / 'llama-server'))
        for key, value in [('--load-mode', 'mmap'), ('--lazy-mode', 'on'),
                           ('--moe-cache-profile', self.env['LLAMA_MOE_CACHE_PROFILE']),
                           ('--moe-cache-slots', '56')]:
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertNotIn('--moe-expert-cache-size', args)
        self.assertEqual(args[args.index('--cache-ram') + 1], '0')
        self.assertIn('--metrics', args)
        self.assertTrue(config.cache_required('chat', config.settings(self.env)))
        gs = config.server_command('chat', {'LLAMA_THREADS': '2'})
        self.assertEqual(gs[0], str(config.BIN / 'llama-server'))
        self.assertNotIn('--moe-cache-profile', gs)

    def test_each_server_gets_its_own_libraries_without_cross_fork_fallback(self):
        inherited = ':'.join([str(config.BIN), str(config.CODACUS_BIN),
                              str(config.DECISION_BIN), '/usr/local/nvidia/lib64', ''])
        for kind, env, binary in [('chat', {}, config.BIN), ('chat', self.env, config.CODACUS_BIN),
                                 ('decision', {}, config.DECISION_BIN), ('embedding', {}, config.BIN)]:
            with self.subTest(kind=kind, binary=binary):
                env = dict(env, LD_LIBRARY_PATH=inherited)
                args = config.server_command(kind, env)
                child = config.child_environment(env, args[0])
                self.assertEqual(child['LD_LIBRARY_PATH'], str(binary) + ':/usr/local/nvidia/lib64')
                self.assertEqual(env['LD_LIBRARY_PATH'], inherited)
        with self.assertRaises(ValueError):
            config.child_environment({'GGML_BACKEND_PATH': '/unrelated/libggml.so'})
        with self.assertRaises(ValueError):
            config.child_environment({}, '/unrelated/llama-server')

    def test_cuda_probe_uses_the_selected_chat_binary_and_libraries(self):
        reply = subprocess.CompletedProcess([], 0, '  CUDA0: fake device\n', '')
        with patch.object(subprocess, 'run', return_value=reply) as run:
            runner.require_cuda(self.env)
            self.assertEqual(run.call_args.args[0], [str(config.CODACUS_BIN / 'llama-server'), '--list-devices'])
            self.assertEqual(run.call_args.kwargs['env']['LD_LIBRARY_PATH'], str(config.CODACUS_BIN))

    def test_cache_disabled_and_dense_chat_have_process_bound_readiness(self):
        for env in [{'LLAMA_MODEL': 'qwen3.8-27b'}, {'LLAMA_MOE_CACHE_SIZE': '0'},
                    dict(self.env, LLAMA_MOE_CACHE_ENABLED='0')]:
            with self.subTest(env=env):
                self.assertFalse(config.cache_required('chat', config.settings(env)))
                path = self.receipt('chat', env)
                self.assertTrue(runner.runtime_ready('chat', env))
                row = json.loads(path.read_text()); row['process'] = ['stale', '0']
                path.write_text(json.dumps(row))
                self.assertFalse(runner.runtime_ready('chat', env))
                child = self.child("common_speculative_init_result: loading draft model 'fake.gguf'\n")
                receipts = []
                def publish(file, data):
                    receipts.append(dict(data))
                with patch.object(runner, 'require_cuda'), patch.object(runner, 'provision'), \
                        patch.object(subprocess, 'Popen', return_value=child), \
                        patch.object(runner, 'atomic_json', side_effect=publish), \
                        patch.object(runner, 'ensure_profile') as trace:
                    self.assertEqual(runner.serve('chat', env), 0)
                    self.assertEqual(len(receipts), 1)
                    self.assertIs(receipts[0]['cache_active'], False)
                    trace.assert_not_called()

    def test_codacus_requires_engagement_before_draft(self):
        for lines, succeeds in [(
                'init_moe_expert_cache: expert cache: 40 layers x 112 slots, 100 MiB uploaded to CUDA0\n'
                "common_speculative_init_result: loading draft model 'fake.gguf'\n", True), (
                'WARNING: cache pack allocation failed\n'
                "common_speculative_init_result: loading draft model 'fake.gguf'\n", False)]:
            with self.subTest(succeeds=succeeds), patch.object(runner, 'require_cuda'), \
                    patch.object(runner, 'provision'), patch.object(runner, 'ensure_profile'), \
                    patch.object(subprocess, 'Popen', return_value=self.child(lines)):
                if succeeds:
                    self.assertEqual(runner.serve('chat', self.env), 0)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'did not engage before MTP'):
                        runner.serve('chat', self.env)
                self.assertFalse((runner.STATE_DIR / 'chat.json').exists())
        self.receipt('chat', self.env)
        self.assertFalse(runner.runtime_ready('chat', self.env))
        self.receipt('chat', self.env, cache_active=True)
        self.assertTrue(runner.runtime_ready('chat', self.env))

    def test_verified_weights_precede_opt_in_profile_generation(self):
        order = []
        env = dict(self.env, LLAMA_DOWNLOAD_MODELS='1', LLAMA_GENERATE_PROFILE='1')
        def provision(kind, env, download):
            self.assertTrue(download)
            order.append(kind)
        def trace(env, generate, adopt):
            self.assertTrue(generate); self.assertFalse(adopt)
            self.assertEqual(order, ['chat', 'draft'])
            order.append('profile')
        child = self.child('init_moe_expert_cache: expert cache: 40 layers, uploaded to CUDA0\n')
        with patch.object(runner, 'require_cuda'), patch.object(runner, 'provision', side_effect=provision), \
                patch.object(runner, 'ensure_profile', side_effect=trace), \
                patch.object(subprocess, 'Popen', return_value=child):
            self.assertEqual(runner.serve('chat', env), 0)
        self.assertEqual(order, ['chat', 'draft', 'profile'])

    def test_trace_timeout_is_present_and_bounded(self):
        self.assertEqual(config.settings(self.env)['trace_timeout'], 3600)
        self.assertEqual(config.settings(dict(self.env, LLAMA_TRACE_TIMEOUT_SECONDS='61'))['trace_timeout'], 61)
        for value in ('0', '59', '7201', 'nan', '-1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config.settings(dict(self.env, LLAMA_TRACE_TIMEOUT_SECONDS=value))

    def test_profile_identity_uses_the_supplied_env_not_the_parent_process(self):
        with patch.dict(os.environ, {'LLAMA_MODEL': 'qwen3.6-35b-a3b'}, clear=True):
            self.assertEqual(profile.recipe(self.env)['model'], config.models()['variants']['qwen3.8-flash-next'])

    def test_profile_is_explicit_and_existing_data_is_never_overwritten(self):
        target = Path(self.env['LLAMA_MOE_CACHE_PROFILE'])
        with self.assertRaisesRegex(RuntimeError, 'Missing routing profile'):
            profile.ensure_profile(self.env)
        self.csv(target)
        data = target.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'unverified'):
            profile.ensure_profile(self.env, generate=True)
        self.assertEqual(target.read_bytes(), data)
        self.assertEqual(profile.ensure_profile(self.env, adopt=True), target)
        self.assertEqual(profile.ensure_profile(self.env), target)
        target.write_text('0,0,10,1,2,3,4,5,6,7,8,9\n')
        with self.assertRaisesRegex(RuntimeError, 'unverified'):
            profile.ensure_profile(self.env, generate=True)
        self.assertTrue(target.read_text().startswith('0,0,10,'))

    def test_fake_trace_generation_is_bounded_and_reuses_verified_result(self):
        env = dict(self.env, LLAMA_TRACE_TIMEOUT_SECONDS='61')
        def fake(command, env, check, timeout):
            self.assertEqual(command[0], str(config.CODACUS_BIN / 'llama-moe-trace'))
            self.assertEqual(env['LD_LIBRARY_PATH'], str(config.CODACUS_BIN))
            self.assertTrue(check); self.assertEqual(timeout, 61)
            self.csv(Path(env['MOE_TRACE_OUT']))
        with patch.object(subprocess, 'run', side_effect=fake) as run:
            target = profile.ensure_profile(env, generate=True)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(len(target.read_text().splitlines()), 2)
            self.assertEqual(profile.ensure_profile(env, generate=True), target)
            self.assertEqual(run.call_count, 2)
            row = json.loads(target.with_suffix('.json').read_text())
            self.assertEqual(row['recipe'], profile.recipe(env))
            self.assertEqual(row['sha256'], profile.validate_profile(target))

    def test_failed_trace_keeps_staging_but_never_publishes_or_retries(self):
        def fail(command, env, **kwargs):
            self.csv(Path(env['MOE_TRACE_OUT']))
            raise subprocess.TimeoutExpired(command, 61)
        with patch.object(subprocess, 'run', side_effect=fail) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                profile.ensure_profile(self.env, generate=True)
            self.assertEqual(run.call_count, 1)
        self.assertFalse(Path(self.env['LLAMA_MOE_CACHE_PROFILE']).exists())
        self.assertEqual(len(list(self.root.glob('flash-trace-*/code.csv'))), 1)

    def test_profile_validation_rejects_wrong_shape_symlinks_and_unsupported_models(self):
        target = self.root / 'invalid.csv'
        for text in ('', '0,0,1\n', '0,0,0,0,2,3,4,5,6,7,8,9\n',
                     '-1,0,0,1,2,3,4,5,6,7,8,9\n', '0,0,512,1,2,3,4,5,6,7,8,9\n'):
            target.write_text(text)
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                profile.validate_profile(target)
        self.csv(target)
        link = self.root / 'link.csv'; link.symlink_to(target)
        with self.assertRaises(RuntimeError):
            profile.validate_profile(link)
        for model in ('qwen3.6-35b-a3b', 'qwen3.8-27b'):
            with self.assertRaisesRegex(ValueError, 'supports only'):
                config.settings(dict(self.env, LLAMA_MODEL=model))
            # An explicit non-profile-cache Codacus setup is not confused with failed activation.
            self.assertFalse(config.settings(dict(self.env, LLAMA_MODEL=model, LLAMA_MOE_CACHE_ENABLED='0'))['moe_cache_enabled'])

    def test_decision_and_embedding_are_independent_and_observable(self):
        args = config.server_command('decision', {})
        self.assertEqual(args[0], str(config.DECISION_BIN / 'llama-server'))
        for key, value in [('--port', '11440'), ('--ctx-size', '8192'), ('--n-gpu-layers', '999')]:
            self.assertEqual(args[args.index(key) + 1], value)
        for forbidden in ('--spec-type', '--moe-cache-profile', '--moe-expert-cache-size', '--props'):
            self.assertNotIn(forbidden, args)
        self.assertIn('--metrics', args)
        embedding = config.server_command('embedding', {})
        self.assertEqual(embedding[embedding.index('--n-gpu-layers') + 1], '0')
        self.assertIn('--metrics', embedding)
        with patch.object(runner, 'provision') as download, patch.object(subprocess, 'Popen') as launch:
            self.assertEqual(runner.serve('decision', {'LLAMA_DECISION_ENABLED': '0'}), 0)
            download.assert_not_called(); launch.assert_not_called()
        self.receipt('decision', {})
        self.assertTrue(runner.runtime_ready('decision', {}))
        self.assertFalse(runner.runtime_ready('decision', {'LLAMA_DECISION_CONTEXT': '4096'}))

    def test_container_health_includes_only_enabled_services_and_current_receipts(self):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = b'{"status":"ok"}'
        netbird = subprocess.CompletedProcess([], 0, json.dumps({'status': 'Connected'}), '')
        for env, kinds, ports in [({}, ['chat', 'embedding', 'decision'], ['11434/health', '11434/health/embedding', '11440/health']),
                                 ({'LLAMA_EMBEDDING_ENABLED': '0', 'LLAMA_DECISION_ENABLED': '0'}, ['chat'], ['11434/health']),
                                 ({'LLAMA_DECISION_PORT': '11441'}, ['chat', 'embedding', 'decision'], ['11434/health', '11434/health/embedding', '11441/health'])]:
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True), \
                    patch.object(healthcheck.urllib.request, 'build_opener', return_value=opener), \
                    patch.object(healthcheck, 'runtime_ready', return_value=True) as ready, \
                    patch.object(subprocess, 'run', return_value=netbird):
                opener.reset_mock()
                self.assertIsNone(healthcheck.main())
                self.assertEqual([c.args[0] for c in ready.call_args_list], kinds)
                self.assertEqual([c.args[0] for c in opener.open.call_args_list], ['http://127.0.0.1:' + p for p in ports])
                ready.side_effect = lambda kind, env: kind != kinds[-1]
                with self.assertRaisesRegex(RuntimeError, 'readiness receipt is missing/stale'):
                    healthcheck.main()


if __name__ == '__main__':
    unittest.main()
