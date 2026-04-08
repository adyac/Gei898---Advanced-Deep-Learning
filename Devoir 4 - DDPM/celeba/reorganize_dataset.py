"""
Reorganize celeba_hq dataset into image/ and condition/ subfolders.

Before:
    dataset/celeba_hq/{split}/{gender}/*.jpg

After:
    dataset/celeba_hq/{split}/{gender}/image/*.jpg    (moved)
    dataset/celeba_hq/{split}/{gender}/condition/*.jpg (copied from image/)
"""

import shutil
from pathlib import Path

ROOT = Path("./dataset/celeba_hq")
SPLITS = ["train/female", "train/male", "val/female", "val/male"]

for split in SPLITS:
    src_dir = ROOT / split
    image_dir = src_dir / "image"
    cond_dir  = src_dir / "condition"

    image_dir.mkdir(exist_ok=True)
    cond_dir.mkdir(exist_ok=True)

    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in (".jpg", ".png") and p.is_file())
    print(f"{split}: {len(images)} images")

    for img_path in images:
        dst_image = image_dir / img_path.name
        dst_cond  = cond_dir  / img_path.name

        shutil.move(str(img_path), str(dst_image))
        shutil.copy2(str(dst_image), str(dst_cond))

    print(f"  -> image/    : {len(list(image_dir.iterdir()))} files")
    print(f"  -> condition/: {len(list(cond_dir.iterdir()))} files")

print("Done.")
