"""One map task: count this HDFS split, then emit itemset and local count."""
import sys

CANDIDATES = []
with open("candidates.txt", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            CANDIDATES.append(int(line))

counts = [0] * len(CANDIDATES)
for line in sys.stdin:
    text = line.strip()
    if not text:
        continue
    mask = int(text)
    for i, cand in enumerate(CANDIDATES):
        if (mask & cand) == cand:
            counts[i] += 1

for cand, count in zip(CANDIDATES, counts):
    if count:
        sys.stdout.write(f"{cand}\t{count}\n")
