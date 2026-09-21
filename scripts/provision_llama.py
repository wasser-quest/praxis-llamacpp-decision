#!/usr/bin/env python3
"""Provision pinned GGUFs once; verify SHA256; never replace existing model files.

--download explicitly authorizes missing downloads after reviewing model licenses.
Without it this command only verifies existing files. No inference or retries.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
import urllib.parse
import urllib.request

from llama_config import model_dir, model_spec, settings

CHUNK = 4 * 1024 * 1024


@contextmanager
def lock(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError('Refusing non-regular lock file')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def fingerprint(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f'Expected a regular file: {path}')
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as out:
        temporary = Path(out.name)
        try:
            json.dump(value, out, sort_keys=True)
            out.flush(); os.fsync(out.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def read_receipt(path):
    try:
        if path.is_symlink() or path.stat().st_size > 65536:
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def verify_file(path, entry, receipt, rehash=False):
    before = fingerprint(path)
    if before[2] != entry['size']:
        raise RuntimeError(f'Model size mismatch; existing file preserved: {path}')
    # Coarse-timestamp filesystems can report identical stats for rapid rewrites.
    # Never trust a receipt for files modified within the last two seconds.
    settled = max(before[3], before[4]) < time.time_ns() - 2_000_000_000
    if not rehash and settled and receipt == {'sha256': entry['sha256'], 'stat': before}:
        return receipt
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    if before != fingerprint(path) or digest.hexdigest() != entry['sha256']:
        raise RuntimeError(f'Model SHA256 mismatch or concurrent change; file preserved: {path}')
    return {'sha256': entry['sha256'], 'stat': before}


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != 'https' or parsed.username is not None or parsed.password is not None:
            raise RuntimeError('Refusing an unsafe model-download redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# No ambient proxy credentials; public pinned model weights need no access token.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), HTTPSRedirect())


def download_file(model, entry, target, deadline):
    partial = target.with_name(target.name + '.part')
    if partial.exists() or partial.is_symlink():
        raise RuntimeError(f'Inspect/remove the interrupted download before retrying: {partial}')
    url = 'https://huggingface.co/{}/resolve/{}/{}'.format(
        model['repository'], model['revision'], urllib.parse.quote(entry['path'], safe='/'))
    digest, size = hashlib.sha256(), 0
    # Exclusive create: partials survive interruption for deliberate inspection.
    with partial.open('xb') as out:
        with OPENER.open(url, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError('Expected HTTP 200 for model download')
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != entry['size']:
                raise RuntimeError('Unexpected model download size')
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Model download deadline exceeded')
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > entry['size']:
                    raise RuntimeError('Model download exceeds pinned size')
                digest.update(chunk); out.write(chunk)
        out.flush(); os.fsync(out.fileno())
    if size != entry['size'] or digest.hexdigest() != entry['sha256']:
        raise RuntimeError(f'Download integrity check failed; partial preserved: {partial}')
    os.link(partial, target)  # publish without overwriting any concurrent file
    partial.unlink()
    return {'sha256': entry['sha256'], 'stat': fingerprint(target)}


def provision(kind, env, download=False, rehash=False):
    config, model = settings(env), model_spec(kind, env)
    directory = model_dir(kind, env)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock(directory / '.provision.lock'):
        receipt_path = directory / '.verified.json'
        receipt = read_receipt(receipt_path)
        if receipt.get('revision') != model['revision']:
            receipt = {}
        verified = receipt.get('files', {})
        if not isinstance(verified, dict):
            verified = {}
        missing = []
        for entry in model['files']:
            path = directory / Path(entry['path']).name
            if path.exists() or path.is_symlink():
                verified[path.name] = verify_file(path, entry, verified.get(path.name), rehash)
            else:
                missing.append((entry, path))
        if missing and not download:
            raise RuntimeError('Missing pinned GGUFs. Set LLAMA_DOWNLOAD_MODELS=1 after license review, '
                               'or provide all shards in ' + str(directory))
        if missing:
            needed = sum(entry['size'] for entry, _ in missing)
            if shutil.disk_usage(directory).free < needed + 1024**3:
                raise RuntimeError('Insufficient free disk for selected GGUFs plus 1 GiB headroom')
        deadline = time.monotonic() + config['download_timeout']
        for entry, path in missing:
            print(f'Downloading {model["repository"]} @ {model["revision"]}: {path.name}', flush=True)
            verified[path.name] = download_file(model, entry, path, deadline)
            atomic_json(receipt_path, {'revision': model['revision'], 'files': verified})
        atomic_json(receipt_path, {'revision': model['revision'], 'files': verified})
    print(f'{kind}: all pinned GGUFs verified (cached checksums reused if unchanged)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['chat', 'draft', 'embedding', 'decision'], default='chat')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--rehash', action='store_true', help='Read and SHA256 every byte, ignoring verification receipts')
    args = parser.parse_args()
    os.umask(0o077)
    provision(args.kind, os.environ, args.download, args.rehash)
