"""AFHQ image integrity, group-aware partitions, and local BLIP captions."""

from __future__ import annotations
import hashlib, json, random, re, time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def image_inventory(seed=20261018):
    """Split original images before captions/latents; near duplicates share groups."""
    import imagehash

    paths = sorted((ROOT / "04_Data/originals/afhq").glob("*/cat/*"))
    records = []
    for i, p in enumerate(paths):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        with Image.open(p) as im:
            im.load()
            rgb = im.convert("RGB")
            phash = str(imagehash.phash(rgb))
            records.append(
                {
                    "id": f"{p.parts[-3]}_{p.stem}",
                    "path": p.relative_to(ROOT).as_posix(),
                    "source_split": p.parts[-3],
                    "width": im.width,
                    "height": im.height,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "phash": phash,
                }
            )
        if i % 1000 == 0:
            print("image audit", i, flush=True)
    n = len(records)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    hashes = [int(r["phash"], 16) for r in records]
    pairs = []
    for i in range(n):
        for j in range(i):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance <= 4:
                union(i, j)
                pairs.append({"a": records[i]["id"], "b": records[j]["id"], "hamming": distance})
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault(find(i), []).append(i)
    eligible = []
    cross = []
    for k, indices in groups.items():
        has_test = any(records[i]["source_split"] == "val" for i in indices)
        for i in indices:
            records[i]["duplicate_group"] = f"g{k:05d}"
            if has_test:
                records[i]["split"] = (
                    "test" if records[i]["source_split"] == "val" else "excluded_test_neighbor"
                )
                if records[i]["split"] == "excluded_test_neighbor":
                    cross.append(records[i]["id"])
        if not has_test:
            eligible.append(k)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    target = round(sum(len(groups[k]) for k in eligible) * 0.10)
    nv = 0
    for k in eligible:
        split = "validation" if nv < target else "train"
        for i in groups[k]:
            records[i]["split"] = split
        if split == "validation":
            nv += len(groups[k])
    out = ROOT / "04_Data/metadata"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "images.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = {
        "seed": seed,
        "total_decoded": n,
        "counts": dict(Counter(r["split"] for r in records)),
        "dimensions": dict(Counter(f'{r["width"]}x{r["height"]}' for r in records)),
        "duplicate_rule": "64-bit perceptual hash Hamming <= 4; transitive groups; exact SHA256 also recorded",
        "near_duplicate_pairs": pairs,
        "excluded_train_neighbors_of_test": cross,
        "limitation": "Perceptual-hash grouping is a conservative duplicate screen, not proof that all semantically similar images are unrelated.",
    }
    write_json(out / "split_audit.json", summary)
    print(json.dumps(summary), flush=True)
    return records


def conservative_caption(raw):
    raw = re.sub(r"\s+", " ", raw.lower().strip())
    if not re.search(r"\b(cat|kitten|kitty)\b", raw):
        return "a close-up portrait photograph of a cat", True
    # Only coat descriptors in the noun phrase before 'cat' are retained. BLIP
    # commonly invents furniture/actions/eye colors outside an AFHQ face crop.
    prefix = re.split(r"\b(cat|kitten|kitty)\b", raw, maxsplit=1)[0]
    colors = []
    for word in re.findall(r"\b(black|white|gray|grey|brown|orange|ginger)\b", prefix):
        word = {"grey": "gray", "ginger": "orange"}.get(word, word)
        if word not in colors:
            colors.append(word)
    descriptions = []
    if colors:
        descriptions.append(" and ".join(colors[:2]))
    for word in ("tabby", "calico", "striped", "fluffy"):
        if re.search(r"\b" + word + r"\b", prefix):
            descriptions.append(word)
    if re.search(r"long[- ]haired", prefix):
        descriptions.append("long-haired")
    adjective = (" ".join(descriptions) + " ") if descriptions else ""
    article = "an" if adjective.startswith("orange") else "a"
    return f"a portrait photograph of {article} {adjective}cat", False


def normalize_captions():
    path = ROOT / "04_Data/metadata/captions.jsonl"
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    overrides = json.loads(
        (ROOT / "03_Code/scripts/caption_overrides.json").read_text(encoding="utf-8")
    )
    for r in records:
        r["caption"], r["caption_fallback"] = conservative_caption(r["raw_blip_caption"])
        r["caption_cleaning"] = (
            "Keep only coat colors/pattern/long-haired/fluffy adjectives before cat noun; strip action, furniture, eyes, background, and speculative pose."
        )
        if r["id"] in overrides:
            r["caption"] = overrides[r["id"]]["caption"]
            r["caption_override_reason"] = overrides[r["id"]]["reason"]
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = {
        "count": len(records),
        "fallbacks": sum(r["caption_fallback"] for r in records),
        "unique_captions": len(set(r["caption"] for r in records)),
        "caption_counts": dict(Counter(r["caption"] for r in records)),
        "final_test_captioned": False,
        "quality_note": "Automated weak labels with an AI-assisted visual sample audit, not manually verified ground truth.",
    }
    write_json(ROOT / "04_Data/metadata/caption_summary.json", summary)
    caption_contact_sheet(records)
    print(json.dumps(summary), flush=True)


def caption_images(limit=None):
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    records = [
        json.loads(x)
        for x in (ROOT / "04_Data/metadata/images.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # Final test photos are not captioned or used to construct training prompts.
    records = [r for r in records if r["split"] in ("train", "validation")]
    if limit:
        records = records[:limit]
    out = ROOT / "04_Data/metadata/captions.jsonl"
    done = {}
    if out.exists():
        done = {r["id"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    todo = [r for r in records if r["id"] not in done]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    proc = BlipProcessor.from_pretrained(ROOT / "05_Models/blip_captioner", local_files_only=True)
    model = (
        BlipForConditionalGeneration.from_pretrained(
            ROOT / "05_Models/blip_captioner", local_files_only=True, torch_dtype=dtype
        )
        .to(device)
        .eval()
    )
    started = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), 8):
            batch = todo[start : start + 8]
            images = [Image.open(ROOT / r["path"]).convert("RGB") for r in batch]
            inputs = proc(images=images, return_tensors="pt").to(device, dtype)
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=28, num_beams=3)
            captions = proc.batch_decode(tokens, skip_special_tokens=True)
            for r, raw in zip(batch, captions):
                cap, fallback = conservative_caption(raw)
                result = {
                    **r,
                    "raw_blip_caption": raw,
                    "caption": cap,
                    "caption_fallback": fallback,
                    "caption_source": "Salesforce/blip-image-captioning-base; deterministic beam search, beams=3; generated label not human annotation",
                }
                f.write(json.dumps(result) + "\n")
                done[r["id"]] = result
            f.flush()
            if start % 80 == 0:
                print(
                    f"captioned {start+len(batch)}/{len(todo)}, seconds {time.time()-started:.1f}",
                    flush=True,
                )
    captions = list(done.values())
    write_json(
        ROOT / "04_Data/metadata/caption_summary.json",
        {
            "count": len(captions),
            "fallbacks": sum(r["caption_fallback"] for r in captions),
            "unique_captions": len(set(r["caption"] for r in captions)),
            "elapsed_seconds_this_run": time.time() - started,
            "final_test_captioned": False,
        },
    )
    caption_contact_sheet(captions)


def caption_contact_sheet(records, n=48):
    train = sorted(
        [r for r in records if r["split"] == "train"],
        key=lambda r: hashlib.sha256(("audit" + r["id"]).encode()).hexdigest(),
    )[:n]
    out = ROOT / "04_Data/caption_audit"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "audit_sample.json", train)
    for part in range((len(train) + 11) // 12):
        canvas = Image.new("RGB", (1400, 1260), "white")
        d = ImageDraw.Draw(canvas)
        for j, r in enumerate(train[part * 12 : part * 12 + 12]):
            x = (j % 4) * 350
            y = (j // 4) * 420
            im = Image.open(ROOT / r["path"]).convert("RGB").resize((300, 300))
            canvas.paste(im, (x + 10, y))
            text = r["id"] + "\nRAW: " + r["raw_blip_caption"] + "\nCLEAN: " + r["caption"]
            import textwrap

            lines = "\n".join(textwrap.fill(t, 50) for t in text.split("\n"))
            d.text((x + 10, y + 304), lines, fill="black")
        canvas.save(out / f"caption_audit_{part+1}.jpg", quality=92)
