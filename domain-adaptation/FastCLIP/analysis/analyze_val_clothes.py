import re

INPUT_FILE = "val_data_titles.txt"

CLOTHING_KEYWORDS = [
    "shirt", "t-shirt", "tshirt", "tee",
    "hoodie", "sweater", "jacket", "coat", "blazer",
    "dress", "skirt",
    "pants", "trousers", "jeans", "leggings", "shorts",
    "bra", "underwear", "lingerie",
    "pajama", "sleepwear",
    "sock", "shoe", "sneaker", "boot", "sandal",
    "hat", "cap", "beanie", "scarf", "glove",
    "suit", "vest",
    "swimwear", "swimsuit", "bikini",
    "athletic", "sportswear", "activewear"
]

pattern = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CLOTHING_KEYWORDS) + r")\b",
    re.IGNORECASE
)

total = 0
clothing = 0
matched_titles = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        title = line.strip()
        if not title:
            continue

        total += 1
        if pattern.search(title):
            clothing += 1
            matched_titles.append(title)

print("=" * 60)
print(f"Total titles: {total}")
print(f"Clothing-related titles: {clothing}")
print(f"Percentage: {clothing / total * 100:.2f}%")
print("=" * 60)

# Optional: save matched titles
# with open("clothing_titles.txt", "w", encoding="utf-8") as f:
#     for t in matched_titles:
#         f.write(t + "\n")

# print("Saved clothing titles to clothing_titles.txt")
