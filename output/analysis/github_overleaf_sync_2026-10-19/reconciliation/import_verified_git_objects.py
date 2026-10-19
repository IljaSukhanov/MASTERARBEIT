"""Import authenticated GitHub objects after checking their Git IDs, then fast-forward.

This is a one-time repair for this repository. It never reads credentials or
uses a network connection. Run without --apply to verify only.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import itertools
import json
import subprocess
import sys

ROOT = Path('/Users/iljasukhanov/MASTERARBEIT')
HERE = Path(__file__).resolve().parent
packet = json.loads((HERE / 'authenticated_git_objects.json').read_text())
proposal = json.loads((HERE / 'proposal.json').read_text())

def git(*args, data=None, check=True):
    return subprocess.run(['git', *args], cwd=ROOT, input=data,
                          capture_output=True, check=check)

def oid(kind, data):
    return hashlib.sha1(kind.encode() + b' ' + str(len(data)).encode()
                        + b'\0' + data).hexdigest()

def person(p, offset):
    timestamp = int(datetime.fromisoformat(p['date'].replace('Z', '+00:00')).timestamp())
    return f"{p['name']} <{p['email']}> {timestamp} {offset}"

def commit_bytes(c):
    verification = c.get('verification') or {}
    if verification.get('signature') and verification.get('payload'):
        headers, body = verification['payload'].split('\n\n', 1)
        signature = verification['signature'].replace('\n', '\n ')
        data = (headers + '\ngpgsig ' + signature + '\n\n' + body).encode()
        if oid('commit', data) == c['sha']:
            return data
        raise ValueError('Signed commit does not match its Git ID: ' + c['sha'])
    offsets = ['+0000', '+0200', '+0100']
    offsets += [f"{'+' if minutes >= 0 else '-'}{abs(minutes)//60:02}{abs(minutes)%60:02}"
                for minutes in range(-840, 841, 15)]
    offsets = list(dict.fromkeys(offsets))
    candidates = [(offset, offset) for offset in offsets]
    candidates += list(itertools.product(offsets[:3], repeat=2))
    prefix = 'tree ' + c['tree']['sha'] + '\n'
    prefix += ''.join('parent ' + p['sha'] + '\n' for p in c['parents'])
    messages = [c['message'], c['message'].rstrip('\n') + '\n']
    for author_offset, committer_offset in candidates:
        headers = (prefix + 'author ' + person(c['author'], author_offset) + '\n'
                   + 'committer ' + person(c['committer'], committer_offset) + '\n\n')
        for message in messages:
            data = (headers + message).encode()
            if oid('commit', data) == c['sha']:
                return data
    raise ValueError('Commit could not be reconstructed exactly: ' + c['sha'])

objects = {}
references = set()
for tree in packet['trees']:
    assert not tree.get('truncated'), 'Refusing a truncated tree'
    data = b''.join((entry['mode'].lstrip('0') + ' ' + entry['path']).encode()
                    + b'\0' + bytes.fromhex(entry['sha']) for entry in tree['tree'])
    assert oid('tree', data) == tree['sha'], 'Tree ID mismatch'
    objects[tree['sha']] = ('tree', data)
    references.update(entry['sha'] for entry in tree['tree'])
for commit in packet['commits']:
    data = commit_bytes(commit)
    objects[commit['sha']] = ('commit', data)
    references.add(commit['tree']['sha'])
    references.update(parent['sha'] for parent in commit['parents'])
missing = [sha for sha in references if sha not in objects
           and git('cat-file', '-e', sha, check=False).returncode]
assert not missing, 'Missing referenced objects: ' + ', '.join(missing)
assert git('rev-parse', 'HEAD').stdout.decode().strip() == packet['expected_local_head']
assert git('diff', '--quiet', check=False).returncode == 0, 'Working files changed'
assert git('diff', '--cached', '--quiet', check=False).returncode == 0, 'Staged files changed'
print(f'Verified {len(objects)} Git objects and all their references.')
if '--apply' not in sys.argv:
    sys.exit(0)
for sha, (kind, data) in objects.items():
    written = git('hash-object', '-w', '-t', kind, '--stdin', data=data).stdout.decode().strip()
    assert written == sha

target = packet['target']
before = packet['expected_local_head']
assert git('merge-base', '--is-ancestor', before, target, check=False).returncode == 0
changed = git('diff', '--name-only', before, target).stdout.decode().splitlines()
assert set(changed) == {change['path'] for change in proposal['changes']}, changed
for change in proposal['changes']:
    actual = git('rev-parse', target + ':' + change['path']).stdout.decode().strip()
    assert actual == change['new_blob'], 'Approved source blob mismatch'

backup = 'refs/heads/codex/before-overleaf-sync-2026-10-19'
existing = git('rev-parse', '--verify', backup, check=False)
if existing.returncode:
    git('update-ref', backup, before, '0' * 40)
else:
    assert existing.stdout.decode().strip() == before, 'Backup branch differs'
current_origin = git('rev-parse', 'refs/remotes/origin/main').stdout.decode().strip()
assert current_origin in {before, '3198f8351d93234a100888fc0f066b9a0f919fd5', target}
result = git('merge', '--ff-only', target)
print(result.stdout.decode())
if current_origin != target:
    git('update-ref', 'refs/remotes/origin/main', target, current_origin)
assert git('rev-parse', 'HEAD').stdout.decode().strip() == target
assert git('diff', '--quiet', check=False).returncode == 0
assert git('diff', '--cached', '--quiet', check=False).returncode == 0
assert git('rev-list', '--count', target + '..15e0ae519ea41d52cb9fae991283188d0e167bc5').stdout.strip() == b'0'
print('Local main now matches the verified GitHub repair; all unrelated files are preserved.')
