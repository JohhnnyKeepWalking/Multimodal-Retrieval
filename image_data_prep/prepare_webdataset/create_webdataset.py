"""
create_webdataset.py

Usage example:
python create_webdataset.py \
  --csv tcin_title_image_location.csv \
  --images_root /home/jovyan/two-tower-retrieval-datavol-1/data/clip_image_4M \
  --out_dir /home/jovyan/two-tower-retrieval-datavol-1/data/FastCLIP_Data/webdataset_4M \
  --samples_per_shard 2000 \
  --caption_field p_title \
  --image_field local_path \
  --tcin_field tcin \
  --p_title_field p_title \
  --image_url_field image_url \
  --shuffle \
  --val_frac 0.05
"""

import argparse
import os
import io
import tarfile
import json
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from PIL import Image

# ---------- Helpers ----------
def is_image_valid_bytes(b):
    try:
        im = Image.open(io.BytesIO(b))
        im.verify()  # quick integrity check
        return True
    except Exception:
        return False

def read_image_bytes(path):
    with open(path, "rb") as f:
        return f.read()

def safe_caption_text(caption):
    if caption is None:
        return ""
    if not isinstance(caption, str):
        caption = str(caption)
    return caption.strip()

def safe_metadata_value(v):
    # Ensure metadata values are serializable and non-null
    if pd.isna(v):
        return None
    if v is None:
        return None
    if isinstance(v, (int, float, bool, str)):
        return v
    return str(v)

# ---------- Core function ----------
def make_webdataset_from_csv(
    csv_path,
    images_root,
    out_dir,
    samples_per_shard=2000,
    caption_field="p_title",
    image_field="local_path",
    tcin_field="tcin",
    p_title_field="p_title",
    image_url_field="image_url",
    shuffle=True,
    val_frac=0.05,
    key_prefix="target_4m",
    skip_missing=True,
    max_samples=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, dtype=str)

    # Validate expected columns exist (caption & image are required)
    required = [caption_field, image_field]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")

    # Build list of records with metadata
    records = []
    for _, row in df.iterrows():
        img_loc = row[image_field]
        caption = row[caption_field] if caption_field in row else ""
        # skip rows missing image location
        if pd.isna(img_loc) or str(img_loc).strip() == "":
            continue

        # Resolve image path: prefer absolute if provided, else relative to images_root
        img_path = Path(img_loc)
        if not img_path.is_absolute():
            img_path = Path(images_root) / img_loc

        # Extract metadata values (allow missing metadata columns gracefully)
        meta = {
            "tcin": safe_metadata_value(row[tcin_field]) if tcin_field in df.columns else None,
            "p_title": safe_metadata_value(row[p_title_field]) if p_title_field in df.columns else None,
            "image_url": safe_metadata_value(row[image_url_field]) if image_url_field in df.columns else None,
        }

        records.append((str(img_path), safe_caption_text(caption), meta))

    if shuffle:
        import random
        random.shuffle(records)

    if max_samples:
        records = records[:max_samples]

    # train/val split
    num_total = len(records)
    num_val = int(num_total * val_frac)
    if num_val > 0:
        val_records = records[:num_val]
        train_records = records[num_val:]
    else:
        val_records, train_records = [], records

    def write_split(split_name, split_records):
        if len(split_records) == 0:
            print(f"No records for split {split_name}, skipping.")
            return

        shard_idx = 0
        sample_idx = 0
        cur_shard = out_dir / f"{key_prefix}-{split_name}-{shard_idx:05d}.tar"
        tar = tarfile.open(cur_shard, "w")  # use "w:gz" for gzipped tar
        written_in_shard = 0

        for img_path, caption, meta in tqdm(split_records, desc=f"Writing {split_name}", unit="samples"):
            # check file exists
            if not os.path.exists(img_path):
                if skip_missing:
                    # skip sample
                    continue
                else:
                    tar.close()
                    raise FileNotFoundError(f"Missing image: {img_path}")

            # read image bytes and quick validation
            try:
                img_bytes = read_image_bytes(img_path)
            except Exception as e:
                print(f"Warning: cannot read {img_path}: {e}")
                continue

            if not is_image_valid_bytes(img_bytes):
                print(f"Warning: invalid image (skipping): {img_path}")
                continue

            # create sample key (zero-padded global index for this split)
            key = f"{sample_idx:012d}"
            sample_idx += 1

            # --- write image ---
            img_name = f"{key}.jpg"  # stored name inside tar
            ti = tarfile.TarInfo(name=img_name)
            ti.size = len(img_bytes)
            tar.addfile(ti, io.BytesIO(img_bytes))

            # --- write caption (txt) ---
            text_bytes = caption.encode("utf-8")
            txt_name = f"{key}.txt"
            ti2 = tarfile.TarInfo(name=txt_name)
            ti2.size = len(text_bytes)
            tar.addfile(ti2, io.BytesIO(text_bytes))

            # --- write metadata (json) ---
            # Ensure JSON uses UTF-8 and preserve None as null
            meta = dict(meta)
            meta["key"] = key
            json_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
            json_name = f"{key}.json"
            ti3 = tarfile.TarInfo(name=json_name)
            ti3.size = len(json_bytes)
            tar.addfile(ti3, io.BytesIO(json_bytes))

            written_in_shard += 1

            # rotate shard by sample count
            if written_in_shard >= samples_per_shard:
                tar.close()
                shard_idx += 1
                cur_shard = out_dir / f"{key_prefix}-{split_name}-{shard_idx:05d}.tar"
                tar = tarfile.open(cur_shard, "w")
                written_in_shard = 0

        tar.close()
        print(f"Wrote {sample_idx} samples to {shard_idx+1} shards for split '{split_name}'")

    # Write splits
    write_split("train", train_records)
    if num_val > 0:
        write_split("val", val_records)

    # write simple manifest
    manifest = []
    for p in sorted(out_dir.glob(f"{key_prefix}-*.tar")):
        manifest.append(str(p.name))
    manifest_path = out_dir / f"{key_prefix}-manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for name in manifest:
            f.write(name + "\n")
    print(f"Manifest saved to {manifest_path}")
    print("Done.")

# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--samples_per_shard", type=int, default=2000)
    parser.add_argument("--caption_field", default="p_title")
    parser.add_argument("--image_field", default="local_path")
    parser.add_argument("--tcin_field", default="tcin")
    parser.add_argument("--p_title_field", default="p_title")
    parser.add_argument("--image_url_field", default="image_url")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--key_prefix", default="target_4m")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    make_webdataset_from_csv(
        csv_path=args.csv,
        images_root=args.images_root,
        out_dir=args.out_dir,
        samples_per_shard=args.samples_per_shard,
        caption_field=args.caption_field,
        image_field=args.image_field,
        tcin_field=args.tcin_field,
        p_title_field=args.p_title_field,
        image_url_field=args.image_url_field,
        shuffle=args.shuffle,
        val_frac=args.val_frac,
        key_prefix=args.key_prefix,
        skip_missing=args.skip_missing,
        max_samples=args.max_samples,
    )
