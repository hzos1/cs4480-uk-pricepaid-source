"""Sum mapper counts for one itemset. Hadoop Streaming sorts the keys."""
import sys

current = None
total = 0
for line in sys.stdin:
    key, _, value = line.strip().partition("\t")
    if not key:
        continue
    count = int(value)
    if key != current:
        if current is not None:
            sys.stdout.write(f"{current}\t{total}\n")
        current = key
        total = count
    else:
        total += count
if current is not None:
    sys.stdout.write(f"{current}\t{total}\n")
