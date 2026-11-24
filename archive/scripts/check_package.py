"""Verify packaged file hashes without touching the measurement dataset."""
from pathlib import Path
import hashlib,json


def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SystemExit('Missing/invalid package path: '+item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise SystemExit('Changed package file: '+item['path'])
    print(f"Verified {len(manifest['files'])} packaged file hashes. Data folder is supplied separately.")


if __name__=='__main__':main()
