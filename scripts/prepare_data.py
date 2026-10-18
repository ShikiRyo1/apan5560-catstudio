import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catstudio.data import image_inventory, caption_images, normalize_captions

p = argparse.ArgumentParser()
p.add_argument("phase", choices=["inventory", "caption", "normalize"])
p.add_argument("--limit", type=int)
a = p.parse_args()
if a.phase == "inventory":
    image_inventory()
elif a.phase == "caption":
    caption_images(a.limit)
else:
    normalize_captions()
