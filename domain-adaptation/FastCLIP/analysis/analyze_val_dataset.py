import webdataset as wds

SHARDS = "/home/jovyan/two-tower-retrieval-datavol-1/data/FastCLIP_Data/webdataset_4M/target_4m-val-{00000..00009}.tar"
OUTPUT_FILE = "val_data_titles.txt"

count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sample in wds.WebDataset(SHARDS).decode():
        meta = sample.get("json")
        if not isinstance(meta, dict):
            continue

        title = meta.get("p_title")
        if not title:
            continue

        # ensure one title per line
        title = title.replace("\n", " ").strip()
        f.write(title + "\n")
        count += 1

print(f"Done. Wrote {count} titles to {OUTPUT_FILE}")
