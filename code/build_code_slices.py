"""Build code slices (hop 0/1/2) from data/code_doc_pairs.json; output is written under results/_regenerated/data/code_slices.json."""
import json
import sys
import os
import time

# Ensure the get_slices module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "slicing"))
import get_slices as gs

import config
INPUT = str(config.CODE_DOC_PAIRS)
OUTPUT = str(config.OUTPUT_DATA_DIR / "code_slices.json")

HOP_CONFIGS = [0, 1, 2]
DEDUP = "strict"


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    print(f"Loaded {total} entries from {INPUT}")
    print(f"Hop configs: {HOP_CONFIGS}, dedup_strategy: {DEDUP!r}")
    print()

    results = []
    t_start = time.time()

    for idx, entry in enumerate(dataset):
        entry_id = entry["id"]
        code = entry["code_formatted"]

        sliced_entry = {"id": entry_id}

        for hops in HOP_CONFIGS:
            key = f"hop_{hops}"
            try:
                r = gs.slice_java_method(code, max_hops=hops, dedup_strategy=DEDUP)
                sliced_entry[key] = r
            except Exception as exc:
                sliced_entry[key] = {
                    "is_effective": False, "is_fallback": False,
                    "slice": None, "stmt_retention": None, "token_retention": None,
                    "_error": str(exc),
                }

        results.append(sliced_entry)

        # Progress
        if (idx + 1) % 50 == 0 or idx == total - 1:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx + 1}/{total}]  {elapsed:.0f}s elapsed, "
                  f"~{rate:.1f} entries/s, ~{remaining:.0f}s remaining")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone. {total} entries written to {OUTPUT}")
    print(f"Total time: {elapsed:.0f}s ({elapsed / total:.1f}s per entry)")


if __name__ == "__main__":
    main()
