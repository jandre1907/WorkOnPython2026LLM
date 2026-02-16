"""Probe local API endpoints from the current Python environment.

Tries a set of candidate URLs (env override via THERMAL_API_URL), prints status
and a short snippet of the response body for each.
"""
import os
import urllib.request
import json
import socket

candidates = []
env = os.environ.get('THERMAL_API_URL')
if env:
    candidates.append(env)

# common localhost variants
candidates += [
    'http://127.0.0.1:8085/data.json',
    'http://localhost:8085/data.json',
    'http://[::1]:8085/data.json',
    'http://host.docker.internal:8085/data.json',
    'http://172.29.128.1:8085/data.json',
]

print('Probing API endpoints from Python runtime')
print('Hostname:', socket.gethostname())
print('Candidates:', candidates)

for url in candidates:
    try:
        print('\n==> Trying', url)
        with urllib.request.urlopen(url, timeout=3) as resp:
            status = resp.getcode()
            data = resp.read()
            text = data.decode('utf-8', errors='ignore')
            print('  Status:', status)
            # If JSON, pretty print first object or snippet
            try:
                obj = json.loads(text)
                s = json.dumps(obj, indent=2)[:1000]
                print('  JSON snippet:\n', s)
            except Exception:
                print('  Body snippet:\n', text[:1000])
    except Exception as e:
        print('  Error:', type(e).__name__, str(e))

print('\nProbe complete.')
