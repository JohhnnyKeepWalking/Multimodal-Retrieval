#!/usr/bin/env python3
import argparse, os, tarfile, tempfile, shutil
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument("--val-tar", required=True)
parser.add_argument("--devkit", required=True)
parser.add_argument("--out", default=os.path.expanduser("~/datasets/imagenet"))
args = parser.parse_args()

out = args.out
os.makedirs(out, exist_ok=True)
work = tempfile.mkdtemp(prefix="imagenet_val_")
print("working in", work)

# extract val
print("Extracting val tar ...")
with tarfile.open(args.val_tar, "r") as tar:
    tar.extractall(path=work)
val_jpeg_dir = work  # contains ILSVRC2012_val_XXXX.JPEG files

# extract devkit
print("Extracting devkit ...")
with tarfile.open(args.devkit, "r:gz") as tar:
    tar.extractall(path=work)
devkit_dir = os.path.join(work, "ILSVRC2012_devkit_t12")

# map indices -> wnid
meta = loadmat(os.path.join(devkit_dir, "data", "meta.mat"))
synsets = meta["synsets"]
idx_to_synset = {}
for e in synsets:
    wnid = e[0][0][0]
    syn = e[0][1][0]
    idx_to_synset[int(wnid)] = syn

gt_file = os.path.join(devkit_dir, "data", "ILSVRC2012_validation_ground_truth.txt")
with open(gt_file, "r") as f:
    labels = [int(x.strip()) for x in f.readlines()]

imgs = sorted([f for f in os.listdir(val_jpeg_dir) if f.lower().endswith(".jpeg") or f.lower().endswith(".jpg")])
assert len(imgs) == len(labels), f"image count {len(imgs)} != labels {len(labels)}"

out_val = os.path.join(out, "val")
os.makedirs(out_val, exist_ok=True)

print("Moving files into ImageFolder structure (this may take a while)...")
for img_name, label in zip(imgs, labels):
    syn = idx_to_synset[label]
    dst_dir = os.path.join(out_val, syn)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.move(os.path.join(val_jpeg_dir, img_name), os.path.join(dst_dir, img_name))

# final verification
print("Verifying with torchvision.ImageFolder...")
from torchvision.datasets import ImageFolder
ds = ImageFolder(out_val)
print("Done. Images:", len(ds), "Classes:", len(ds.classes))
print("Output val dir:", out_val)

# cleanup temp
shutil.rmtree(work)
print("Temp cleaned up.")
