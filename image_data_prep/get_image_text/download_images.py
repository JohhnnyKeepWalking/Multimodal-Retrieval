#!/usr/bin/env python3
import argparse, os, mimetypes, time, io
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
from PIL import Image

# ---------- helpers ----------

def infer_ext(content_type: str, url: str) -> str:
    # Try header first
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            # Normalize common image types
            return {".jpe": ".jpg"}.get(ext, ext)
    # Fallback from URL path
    for cand in (".jpg", ".jpeg", ".png", ".webp"):
        if url.lower().endswith(cand):
            return cand if cand != ".jpeg" else ".jpg"
    return ".jpg"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def verify_image_bytes(b: bytes) -> bool:
    # Lightweight validity check
    try:
        im = Image.open(io.BytesIO(b))
        im.verify()  # does not decode full image
        return True
    except Exception:
        return False

def download_one(task):
    """
    Runs in a subprocess.
    task: dict with keys (idx, tcin, url, root, per_dir, retries, timeout, verify)
    """
    idx = task["idx"]
    tcin = str(task["tcin"])
    url = task["url"]
    root = Path(task["root"])
    per_dir = task["per_dir"]
    retries = task["retries"]
    timeout = task["timeout"]
    verify_flag = task["verify"]

    # Shard: folder index = idx // per_dir, zero-padded to 5 digits
    folder = root / f"{idx // per_dir:05d}"
    ensure_dir(folder)

    # We try to determine extension after first successful HEAD/GET
    # But if file already exists (any of the known extensions), we short-circuit
    for ext in (".jpg", ".png", ".webp", ".jpeg"):
        fp = folder / f"{tcin}{ext if ext!='.jpeg' else '.jpg'}"
        if fp.exists():
            return {"idx": idx, "tcin": tcin, "url": url, "local_path": str(fp), "status": "ok", "error": ""}

    last_err = ""
    for attempt in range(retries):
        try:
            # Use GET (not HEAD) because some CDNs don't give correct Content-Type on HEAD or block it
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "img-downloader/1.0"})
            ct = resp.headers.get("Content-Type", "")
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
                time.sleep(0.5 * (attempt + 1))
                continue
            data = resp.content
            if verify_flag and not verify_image_bytes(data):
                last_err = "invalid image bytes"
                time.sleep(0.5 * (attempt + 1))
                continue
            ext = infer_ext(ct, url)
            # Normalize jpeg
            if ext == ".jpeg": ext = ".jpg"
            out_path = folder / f"{tcin}{ext}"
            # Atomic-ish write
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, out_path)
            return {"idx": idx, "tcin": tcin, "url": url, "local_path": str(out_path), "status": "ok", "error": ""}
        except Exception as e:
            last_err = repr(e)
            time.sleep(0.5 * (attempt + 1))
    return {"idx": idx, "tcin": tcin, "url": url, "local_path": "", "status": "fail", "error": last_err}

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with columns: tcin,image_url")
    ap.add_argument("--outdir", required=True, help="Root folder to place images (subfolders of 10k)")
    ap.add_argument("--per_dir", type=int, default=40000, help="Max images assigned per folder")
    ap.add_argument("--workers", type=int, default=64, help="Process count")
    ap.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, default=3, help="Retries per URL")
    ap.add_argument("--verify", action="store_true", help="Verify image bytes using PIL")
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit rows for a small test")
    ap.add_argument("--out_map", default="image_location_map.csv", help="Output mapping table (CSV)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # normalize column names
    df.columns = [c.lower() for c in df.columns]
    if not {"tcin", "image_url"}.issubset(df.columns):
        raise SystemExit("CSV must have tcin and image_url columns")

    # limit for testing
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    outroot = Path(args.outdir); outroot.mkdir(parents=True, exist_ok=True)

    tasks, results = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        tcin, url = str(row.tcin), getattr(row, "image_url")
        if pd.isna(url) or not str(url).strip():
            # mark as fail immediately
            results.append({
                "idx": i, "tcin": tcin, "url": "", "local_path": "",
                "status": "fail", "error": "empty url"
            })
            continue
        tasks.append({
            "idx": i, "tcin": tcin, "url": str(url),
            "root": str(outroot),
            "per_dir": args.per_dir, "retries": args.retries,
            "timeout": args.timeout, "verify": bool(args.verify),
        })

    total = len(tasks) + len(results)
    print(f"Starting download: {total} rows ({len(tasks)} urls to fetch, {len(results)} skipped)")

    # multiprocessing download
    ok, fail = 0, len(results)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, t) for t in tasks]
        for n, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            if res["status"] == "ok": ok += 1
            else: fail += 1
            if n % 100 == 0 or n == len(futs):
                print(f"Progress: {n+fail-len(results)}/{total} done | ok={ok} | fail={fail} | pending={total - (ok+fail)}")

    # save map
    res_df = pd.DataFrame(results).sort_values("idx")
    res_df.rename(columns={"url": "image_url"}, inplace=True)
    res_df.to_csv(Path(args.out_map).with_suffix(".csv"), index=False)
    print(f"\nFinished: ok={ok} fail={fail}, mapping saved → {args.out_map}")

if __name__ == "__main__":
    main()
