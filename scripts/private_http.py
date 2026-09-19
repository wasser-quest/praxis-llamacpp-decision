#!/usr/bin/env python3
"""Shared private-origin HTTP helpers for diagnostics and acceptance; no proxy environment."""
import ipaddress
import json
import time
import urllib.parse
import urllib.request

MAX_JSON = 2 * 1024 * 1024
MAX_FILE = 32 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def validate_base(base):
    parsed = urllib.parse.urlsplit(base)
    try:
        address = ipaddress.ip_address(parsed.hostname or '')
        parsed.port
    except ValueError as error:
        raise ValueError('Use a private literal-IP HTTP(S) origin') from error
    networks = ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '100.64.0.0/10', 'fc00::/7']
    private = address.is_loopback or any(address in ipaddress.ip_network(n) for n in networks)
    if (parsed.scheme not in ['http', 'https'] or not private or address.is_unspecified
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in ['', '/']):
        raise ValueError('Use a private literal-IP origin without credentials, path, query or fragment')
    return base.rstrip('/')


def download(url, maximum, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Diagnostic deadline exceeded')
    with OPENER.open(url, timeout=min(30, remaining)) as response:
        chunks, count = [], 0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic deadline exceeded')
            chunk = response.read1(min(65536, maximum + 1 - count))
            if not chunk:
                return b''.join(chunks)
            count += len(chunk)
            if count > maximum:
                raise ValueError('Response exceeds diagnostic size limit')
            chunks.append(chunk)


def request(url, payload=None, deadline=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    return json.loads(download(req, MAX_JSON, deadline or time.monotonic() + 30))