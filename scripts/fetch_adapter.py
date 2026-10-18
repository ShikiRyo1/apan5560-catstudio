"""Install the published adapter after verifying its release checksum.

No account or inference API key is used. The manifest is created only after
actual training and identifies one immutable release asset by its SHA256.
"""
from __future__ import annotations
import json, shutil, tempfile, zipfile
from pathlib import Path
from download_assets import ROOT, download, sha256

MANIFEST = Path(__file__).resolve().parents[1] / 'config/adapter_release.json'
ALLOWED = {
    'pytorch_lora_weights.safetensors', 'ADAPTER_PROVENANCE.json',
    'MODEL_CARD.md', 'LICENSE_BASE_MODEL.txt', 'LICENSE_TRAINING_DATA.txt',
    'LICENSING.md', 'THIRD_PARTY_NOTICES.md'
}

def install_archive(archive: Path, target: Path, manifest: dict) -> dict:
    if sha256(archive) != manifest['archive_sha256']:
        raise ValueError('Adapter archive checksum mismatch; no model files installed')
    expected = manifest['files']
    if set(expected) != ALLOWED:
        raise ValueError('Release manifest must identify the expected seven adapter files')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='catstudio_adapter_', dir=target.parent) as tmp:
        staged = Path(tmp)
        with zipfile.ZipFile(archive) as package:
            if len(package.namelist()) != len(ALLOWED) or set(package.namelist()) != ALLOWED:
                raise ValueError('Unexpected or unsafe paths in adapter archive')
            for name in sorted(ALLOWED):
                info = package.getinfo(name)
                if info.file_size != expected[name]['bytes']:
                    raise ValueError(f'Unexpected uncompressed size: {name}')
                with package.open(name) as src, open(staged/name, 'wb') as dst:
                    shutil.copyfileobj(src, dst, 1024*1024)
                if sha256(staged/name) != expected[name]['sha256']:
                    raise ValueError(f'Adapter member checksum mismatch: {name}')
        # Verify every existing file before making any changes to the destination.
        for name in sorted(ALLOWED):
            old = target/name
            if old.exists() and sha256(old) != expected[name]['sha256']:
                raise FileExistsError(f'A different adapter file already exists: {old}; archive it before changing the release')
        target.mkdir(parents=True, exist_ok=True)
        for name in sorted(ALLOWED):
            if not (target/name).exists(): shutil.copy2(staged/name, target/name)
    return {'status':'verified_installed','target':str(target),
            'adapter_sha256':expected['pytorch_lora_weights.safetensors']['sha256'],
            'release_url':manifest['release_url']}

def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError('No trained release manifest is installed in this source checkout; obtain the completed release first.')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if not manifest['archive_url'].startswith('https://github.com/ShikiRyo1/apan5560-catstudio/releases/download/'):
        raise ValueError('Unexpected adapter release host/path')
    archive = ROOT/'05_Models/downloads'/manifest['archive_name']
    if Path(manifest['archive_name']).name != manifest['archive_name']:
        raise ValueError('Unsafe archive filename')
    download(manifest['archive_url'], archive)
    print(json.dumps(install_archive(archive, ROOT/'05_Models/cat_lora', manifest),indent=2))

if __name__ == '__main__':
    main()
