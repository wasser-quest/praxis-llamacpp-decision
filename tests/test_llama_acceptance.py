import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import acceptance_llama as acceptance


def generation(text, calls=None):
    message = {'role': 'assistant', 'content': text}
    if calls is not None:
        message['tool_calls'] = calls
    return {'choices': [{'message': message, 'finish_reason': 'tool_calls' if calls else 'stop',
                         'logprobs': {'content': [{'logprob': -0.1}]}}],
            'timings': {}}


class AcceptanceTests(unittest.TestCase):
    def responses(self):
        call = {'id': 'test-call', 'type': 'function',
                'function': {'name': 'add_numbers', 'arguments': '{"a":17,"b":25}'}}
        return [generation('Paris'), generation('Berlin'),
                generation('', [call]), generation('Das Ergebnis ist 42.'),
                generation('Bergsee, 42.'), generation('56.')]

    def transport(self, completions):
        remaining = iter(completions)
        def request(url, payload=None, deadline=None):
            self.assertIsNotNone(deadline)
            if url.endswith('/health'):
                return {'status': 'ok'}
            if url.endswith('/v1/models'):
                return {'data': [{'id': 'qwen3.6-35b-a3b'}]}
            if url.endswith('/v1/embeddings'):
                return {'data': [{'embedding': [0.25] * 768}]}
            self.assertTrue(url.endswith('/v1/chat/completions'))
            self.assertEqual(payload['max_tokens'], 128)
            self.assertTrue(payload['logprobs'])
            return next(remaining)
        return request

    def test_multi_request_tool_and_long_prompt_acceptance(self):
        with patch.object(acceptance, 'request', side_effect=self.transport(self.responses())) as request:
            report = acceptance.run('http://127.0.0.1:11434')
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(report['embedding_dimensions'], 768)
        calls = [c.args[1] for c in request.call_args_list if c.args[0].endswith('/v1/chat/completions')]
        self.assertEqual(len(calls), 6)
        self.assertEqual(calls[3]['messages'][-1], {'role': 'tool', 'tool_call_id': 'test-call', 'content': '{"result":42}'})
        self.assertEqual(calls[4]['temperature'], 0.3)
        self.assertIn('Notiz 100:', calls[4]['messages'][0]['content'])

    def test_nonfinite_missing_and_unfinished_outputs_fail(self):
        for value in [None, float('nan'), float('inf'), True, '0']:
            result = generation('Paris')
            result['choices'][0]['logprobs']['content'][0]['logprob'] = value
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'probabilities'):
                acceptance.check_generation(result)
        result = generation('////')
        result['choices'][0]['finish_reason'] = 'length'
        with self.assertRaisesRegex(RuntimeError, 'token budget'):
            acceptance.check_generation(result)

    def test_failed_generation_is_not_retried(self):
        responses = self.responses()
        responses[0] = generation('////')
        report = {}
        with patch.object(acceptance, 'request', side_effect=self.transport(responses)) as request:
            with self.assertRaisesRegex(RuntimeError, 'unexpected answer'):
                acceptance.run('http://127.0.0.1:11434', report=report)
        self.assertEqual(request.call_count, 3)  # health, catalog, first generation
        self.assertEqual(len(report['steps']), 1)

    def test_public_or_credentialed_origins_are_rejected_before_requests(self):
        for base in ['https://example.com', 'http://8.8.8.8:11434', 'http://key@127.0.0.1:11434']:
            with patch.object(acceptance, 'request') as request:
                with self.assertRaises(ValueError):
                    acceptance.run(base)
                request.assert_not_called()

    def test_cli_requires_opt_in_and_never_overwrites_report(self):
        with patch.object(sys, 'argv', ['acceptance_llama.py']), patch.object(acceptance, 'run') as run, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                acceptance.main()
            run.assert_not_called()
        with tempfile.TemporaryDirectory() as d:
            report = Path(d) / 'report.json'; report.write_text('preserve me')
            with patch.object(sys, 'argv', ['acceptance_llama.py', '--run', '--output', str(report)]), patch.object(acceptance, 'run') as run:
                with self.assertRaises(FileExistsError):
                    acceptance.main()
                run.assert_not_called()
            self.assertEqual(report.read_text(), 'preserve me')


if __name__ == '__main__':
    unittest.main()
