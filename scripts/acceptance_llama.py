#!/usr/bin/env python3
"""Opt-in, bounded GPU API acceptance: repeated text, tools, long prompt and embeddings.

Run only against an idle, ready private server. No downloads, service restarts,
request retries, redirects or ambient proxies. Prompts are generic, not user data.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

from llama_config import model_spec, variant_name
from private_http import request, validate_base


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_generation(result):
    choices = result.get('choices', [])
    require(len(choices) == 1, 'Expected exactly one completion')
    choice = choices[0]
    require(choice.get('finish_reason') in ('stop', 'tool_calls'), 'Generation did not finish within the token budget')
    probabilities = (choice.get('logprobs') or {}).get('content', [])
    require(probabilities and all(finite(t.get('logprob')) for t in probabilities),
            'Missing/non-finite token probabilities: HTTP 200 is not proof of coherent inference')
    return choice['message']


def run(base, timeout=300, report=None):
    base = validate_base(base)
    if not 1 <= timeout <= 900:
        raise ValueError('Timeout must be 1–900 seconds')
    report = report if report is not None else {}
    alias = model_spec('chat', os.environ)['alias']
    report.update(status='running', model=alias, steps=[])
    deadline = time.monotonic() + timeout
    require(request(base + '/health', deadline=deadline).get('status') == 'ok', 'Chat is not ready')
    catalog = request(base + '/v1/models', deadline=deadline)
    require(alias in [m.get('id') for m in catalog.get('data', [])], 'Expected model alias is missing')
    common = {'model': alias, 'temperature': 0, 'max_tokens': 128,
              'chat_template_kwargs': {'enable_thinking': False}, 'logprobs': True, 'top_logprobs': 1}

    def chat(name, messages, expected=None, **extra):
        result = request(base + '/v1/chat/completions', dict(common, messages=messages, **extra), deadline)
        # Preserve failure evidence too, but never copy prompts or full embedding vectors.
        report['steps'].append({'name': name, 'message': result.get('choices', [{}])[0].get('message'),
                                'usage': result.get('usage'), 'timings': result.get('timings', {})})
        message = check_generation(result)
        if expected is not None:
            content = message.get('content') or ''
            require(all(word.casefold() in content.casefold() for word in expected), f'{name}: unexpected answer')
        return message

    for city, country in [('Paris', 'Frankreich'), ('Berlin', 'Deutschland')]:
        chat('plain-' + city, [{'role': 'user', 'content': f'Was ist die Hauptstadt von {country}? Antworte nur mit dem Namen.'}], [city])

    tool = {'type': 'function', 'function': {'name': 'add_numbers', 'description': 'Add two integers and return their sum.',
            'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                           'required': ['a', 'b'], 'additionalProperties': False}}}
    messages = [{'role': 'user', 'content': 'Benutze add_numbers, um 17 und 25 zu addieren. Nenne mir danach das Ergebnis auf Deutsch.'}]
    assistant = chat('tool-call', messages, tools=[tool], tool_choice='required')
    calls = assistant.get('tool_calls', [])
    require(len(calls) == 1, 'Expected exactly one tool call')
    call = calls[0]
    require(call.get('type') == 'function' and call.get('id') and call['function']['name'] == 'add_numbers', 'Unexpected tool')
    require(json.loads(call['function']['arguments']) == {'a': 17, 'b': 25}, 'Unexpected tool arguments')
    # Simulated arithmetic tool only. Never execute arbitrary model-generated code.
    messages += [assistant, {'role': 'tool', 'tool_call_id': call['id'], 'content': '{"result":42}'}]
    chat('tool-reply', messages, ['42'], tools=[tool], tool_choice='none')

    notes = '\n'.join(f'Notiz {i}: Ein zuverlässiger Dienst prüft Antworten und hält seine Daten privat.' for i in range(1, 101))
    prompt = ('Das geheime Testwort ist Bergsee. Merke es dir.\n' + notes +
              '\nNenne das Testwort vom Anfang und das Ergebnis von 17 + 25. Antworte nur mit einem Satz.')
    # A single short request missed the prefetch corruption. Exercise ~2k prompt
    # tokens, multiple prefill batches, non-greedy sampling, then another request.
    chat('long-prompt', [{'role': 'user', 'content': prompt}], ['Bergsee', '42'], temperature=0.3)
    chat('after-long-prompt', [{'role': 'user', 'content': 'Wie viel ist 7 mal 8? Antworte mit einem kurzen Satz.'}], ['56'])
    embedding = request(base + '/v1/embeddings', {'model': model_spec('embedding', os.environ)['alias'], 'input': 'search_query: Hallo Welt'}, deadline)
    vector = embedding['data'][0]['embedding']
    require(len(vector) == 768 and all(finite(x) for x in vector) and any(x != 0 for x in vector), 'Invalid embedding')
    require(request(base + '/health', deadline=deadline).get('status') == 'ok', 'Chat lost readiness')
    report.update(status='pass', embedding_dimensions=len(vector))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Authorize six bounded GPU generations and one CPU embedding request')
    parser.add_argument('--url', default='http://127.0.0.1:11434')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--output', type=Path, help='New private JSON report; existing files are never overwritten')
    args = parser.parse_args()
    if not args.run:
        parser.error('Real inference requires explicit --run')
    output = None
    if args.output:
        output = os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w')
    report = {}
    try:
        run(args.url, args.timeout, report)
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        data = json.dumps(report, ensure_ascii=False, indent=2)
        print(data)
        if output:
            with output:
                output.write(data + '\n')


if __name__ == '__main__':
    main()
