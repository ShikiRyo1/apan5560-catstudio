"""Download public, pinned model assets and official AFHQ v1 data (no API key)."""

from __future__ import annotations
import argparse, hashlib, json, os, time, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "05_Models/hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
REVISIONS = {
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    "Salesforce/blip-image-captioning-base": "82a37760796d32b1411fe092ab5d4e227313294b",
    "openai/clip-vit-base-patch32": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    if path.exists():
        return
    offset = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CatStudio-course-project/1.0",
            **({"Range": f"bytes={offset}-"} if offset else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        if offset and r.status != 206:
            offset = 0
        with open(part, "ab" if offset else "wb") as f:
            last = time.monotonic()
            count = offset
            while chunk := r.read(1024 * 1024):
                f.write(chunk)
                count += len(chunk)
                if time.monotonic() - last > 10:
                    print(f"{path.name}: {count/1e6:.1f} MB", flush=True)
                    last = time.monotonic()
    part.replace(path)


def data():
    url = "https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1"
    archive = ROOT / "04_Data/downloads/afhq_v1.zip"
    download(url, archive)
    archive_hash = sha256(archive)
    if archive_hash != "7f63dcc14ef58c0e849b59091287e1844da97016073aac20403ae6c6132b950f":
        raise ValueError("AFHQ v1 archive SHA256 mismatch")
    destination = ROOT / "04_Data/originals"
    count = {}
    with zipfile.ZipFile(archive) as z:
        cats = [
            n
            for n in z.namelist()
            if n.startswith(("afhq/train/cat/", "afhq/val/cat/"))
            and n.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        for name in cats:
            target = (destination / name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe ZIP path")
            if not target.exists():
                z.extract(name, destination)
            split = name.split("/")[1]
            count[split] = count.get(split, 0) + 1
    assert count == {"train": 5153, "val": 500}, count
    lic = ROOT / "04_Data/licenses"
    lic.mkdir(parents=True, exist_ok=True)
    download(
        "https://raw.githubusercontent.com/clovaai/stargan-v2/master/README.md",
        lic / "AFHQ_upstream_README.md",
    )
    download(
        "https://raw.githubusercontent.com/clovaai/stargan-v2/master/LICENSE",
        lic / "AFHQ_CC_BY_NC_4_LICENSE.txt",
    )
    report = {
        "dataset": "AFHQ v1 cats",
        "source": url,
        "sha256": archive_hash,
        "archive_bytes": archive.stat().st_size,
        "cat_counts": count,
        "license": "CC BY-NC 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "citation": "Choi et al. (2020). StarGAN v2: Diverse Image Synthesis for Multiple Domains. CVPR.",
        "extraction": "Only cat images extracted; other categories remain in original ZIP.",
    }
    (ROOT / "04_Data/download_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


def model(repo, folder, patterns):
    from huggingface_hub import snapshot_download

    revision = REVISIONS[repo]
    target = ROOT / "05_Models" / folder
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=str(target),
        allow_patterns=patterns,
        token=False,
        max_workers=3,
    )
    files = [
        {
            "path": str(p.relative_to(target)).replace("\\", "/"),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        }
        for p in target.rglob("*")
        if p.is_file() and ".cache" not in p.parts and p.name != "PROVENANCE.json"
    ]
    report = {
        "repo": repo,
        "revision": revision,
        "source": f"https://huggingface.co/{repo}",
        "files": files,
        "token_used": False,
    }
    expected_file = Path(__file__).with_name("asset_checksums.json")
    if expected_file.exists():
        expected = json.loads(expected_file.read_text(encoding="utf-8")).get(repo, {})
        missing = set(expected) - {item["path"] for item in files}
        if missing:
            raise ValueError(f"Missing expected asset files: {repo}: {sorted(missing)}")
        for item in files:
            if item["path"] in expected and item["sha256"] != expected[item["path"]]:
                raise ValueError(f'Checksum mismatch: {repo}/{item["path"]}')
    (target / "PROVENANCE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    licenses = {
        "stable-diffusion-v1-5/stable-diffusion-v1-5": (
            "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE",
            "SD_CreativeML_OpenRAIL_M_LICENSE.txt",
        ),
        "Salesforce/blip-image-captioning-base": (
            "https://raw.githubusercontent.com/salesforce/BLIP/main/LICENSE.txt",
            "BLIP_BSD3_LICENSE.txt",
        ),
        "openai/clip-vit-base-patch32": (
            "https://raw.githubusercontent.com/openai/CLIP/main/LICENSE",
            "CLIP_MIT_LICENSE.txt",
        ),
    }
    license_url, license_name = licenses[repo]
    download(license_url, ROOT / "05_Models/licenses" / license_name)
    print(f"Downloaded and hashed {repo}: {len(files)} files", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("asset", choices=["data", "sd15", "blip", "clip"])
    a = p.parse_args()
    if a.asset == "data":
        data()
    elif a.asset == "sd15":
        model(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "base_sd15",
            [
                "model_index.json",
                "README.md",
                "LICENSE*",
                "scheduler/*",
                "tokenizer/*",
                "feature_extractor/*",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "safety_checker/config.json",
                "safety_checker/model.safetensors",
            ],
        )
    elif a.asset == "blip":
        model(
            "Salesforce/blip-image-captioning-base",
            "blip_captioner",
            [
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
                "tokenizer.json",
                "pytorch_model.bin",
                "README.md",
                "LICENSE*",
            ],
        )
    else:
        model(
            "openai/clip-vit-base-patch32",
            "clip_evaluator",
            [
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "tokenizer.json",
                "pytorch_model.bin",
                "README.md",
                "LICENSE*",
            ],
        )
