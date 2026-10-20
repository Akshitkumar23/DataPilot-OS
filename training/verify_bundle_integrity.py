"""Verify the uploaded bundle before using or copying training inputs."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Invalid or missing bundle member: {name}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'BUNDLE_HASH_MISMATCH: {name}')
    return len(manifest['files'])


if __name__ == '__main__':
    import sys
    print('Verified bundle files:', verify(sys.argv[1] if len(sys.argv) > 1 else '.'))
