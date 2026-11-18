# -*- coding: utf-8 -*-
"""
Check that a benchmarks.json you rebuilt with the notebook is the exact subset used in the paper.

  python scripts/check_subset.py path/to/benchmarks.json

Each item is hashed as SHA-256 of json.dumps(item, ensure_ascii=False, sort_keys=True).
The hashes are compared with data/subset_manifest.csv (item_sha256) position by position.
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "subset_manifest.csv"


def main(path: str) -> int:
    benches = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = defaultdict(list)
    with MANIFEST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expected[row["benchmark"]].append(row["item_sha256"])
    ok = True
    for bench, hashes in expected.items():
        got = [hashlib.sha256(json.dumps(q, ensure_ascii=False, sort_keys=True)
                              .encode("utf-8")).hexdigest() for q in benches.get(bench, [])]
        same = sum(a == b for a, b in zip(got, hashes))
        status = "OK" if same == len(hashes) == len(got) else "MISMATCH"
        ok &= status == "OK"
        print(f"{bench:<14} expected={len(hashes):5} got={len(got):5} matching={same:5}  {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
