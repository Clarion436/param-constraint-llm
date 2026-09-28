"""
evaluate_wit.py
Evaluate WIT-extracted constraint expressions against documented constraints.
Compares wit_result (from results/wit/wit_result.json) with doc_constraint
(from data/code_doc_pairs.json) using strict/relaxed precision/recall/F1 metrics.
"""
import json
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "expression_similarity"))
from expr_sim import calculate_constraint_similarity, count_sub_expressions

import config

# ============================================================
# Configuration
# ============================================================
DOC_PATH = str(config.CODE_DOC_PAIRS)
WIT_PATH = str(config.resolve_wit_file("wit_result.json"))
OUTPUT_PATH = str(config.OUTPUT_WIT_DIR / "wit_metrics.csv")


def read_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_f1(p: float, r: float) -> float:
    """Harmonic mean (F1), returns 0.0 when p+r == 0."""
    denom = p + r
    return round(2 * p * r / denom, 4) if denom > 0 else 0.0


def evaluate_wit(doc_path: str, wit_path: str, output_path: str) -> None:
    # Load data
    doc_data = read_json(doc_path)
    wit_data = read_json(wit_path)

    # Index WIT data by id for O(1) lookup
    wit_by_id = {item["id"]: item for item in wit_data}

    rows = []
    error_count = 0
    empty_count = 0

    for item in doc_data:
        item_id = item["id"]
        doc_constraint = item.get("doc_constraint", "")

        # Match WIT entry by id
        wit_item = wit_by_id.get(item_id)
        if wit_item is None:
            print(f"WARNING: id={item_id} not found in WIT data, skipping.")
            continue

        wit_result = wit_item.get("wit_result", "")
        wit_paths = wit_item.get("wit_paths", [])
        miss_reason = wit_item.get("miss_reason", "")

        # Determine if WIT result is usable
        is_empty = (not wit_result
                    or wit_result.startswith("simplify_error")
                    or wit_result in ("true", "false"))
        if is_empty:
            empty_count += 1
            if wit_result.startswith("simplify_error"):
                error_count += 1

        # --- Similarity (0s if result is empty) ---
        if is_empty:
            strict_precision = strict_recall = strict_f1 = 0.0
            relaxed_precision = relaxed_recall = relaxed_f1 = 0.0
            code_subs = 0
        else:
            strict_precision = round(
                calculate_constraint_similarity(doc_constraint, wit_result,
                                                strict=True, normalize_mode="precision"), 4)
            strict_recall = round(
                calculate_constraint_similarity(doc_constraint, wit_result,
                                                strict=True, normalize_mode="recall"), 4)
            strict_f1 = safe_f1(strict_precision, strict_recall)

            relaxed_precision = round(
                calculate_constraint_similarity(doc_constraint, wit_result,
                                                strict=False, normalize_mode="precision"), 4)
            relaxed_recall = round(
                calculate_constraint_similarity(doc_constraint, wit_result,
                                                strict=False, normalize_mode="recall"), 4)
            relaxed_f1 = safe_f1(relaxed_precision, relaxed_recall)

            code_subs = count_sub_expressions(wit_result)

        # Exact match: strict_f1 == 1.0 (parse-failed/empty entries have F1 = 0, excluded)
        exact_match = 1 if strict_f1 == 1.0 else 0

        # None: WIT result missing (miss_reason set) → 1, otherwise 0
        none_pct = 1 if miss_reason else 0

        doc_subs = count_sub_expressions(doc_constraint)

        # Serialize wit_paths; truncate only if exceeds Excel cell limit.
        # Account for CSV double-quote escaping (~3K overhead for 100+ embedded quotes).
        if isinstance(wit_paths, list):
            wit_paths_str = json.dumps(wit_paths, ensure_ascii=False)
            if len(wit_paths_str) > 30000:
                wit_paths_str = wit_paths_str[:29997] + "..."
        else:
            wit_paths_str = str(wit_paths)

        rows.append([
            item_id,
            doc_constraint,
            wit_result,
            wit_paths_str,
            miss_reason,
            len(wit_paths) if isinstance(wit_paths, list) else 0,
            strict_precision,
            strict_recall,
            strict_f1,
            relaxed_precision,
            relaxed_recall,
            relaxed_f1,
            doc_subs,
            code_subs,
            none_pct,
            exact_match,
        ])

    # --- Append mean row ---
    # Numeric cols: path_count(5), sp(6), sr(7), sf1(8), rp(9), rr(10), rf1(11),
    #               doc_subs(12), code_subs(13), none_pct(14), exact_match(15)
    num_cols = list(range(5, 16))
    pct_cols = {14, 15}  # none_pct and exact_match are percentages
    means = []
    for col_idx in num_cols:
        vals = [float(row[col_idx]) for row in rows
                if row[col_idx] != "" and row[col_idx] is not None]
        if not vals:
            means.append("")
            continue
        if col_idx in pct_cols:
            means.append(round(sum(vals) / len(vals) * 100, 2))
        else:
            means.append(round(sum(vals) / len(vals), 4))

    mean_row = ["mean", "", "", "", ""] + means  # text cols: id, doc, wit_result, wit_paths, miss_reason
    rows.insert(0, mean_row)

    # Write CSV
    header = [
        "id",
        "doc_constraint",
        "wit_result",
        "wit_paths",
        "miss_reason",
        "wit_path_count",
        "strict_precision",
        "strict_recall",
        "strict_f1",
        "relaxed_precision",
        "relaxed_recall",
        "relaxed_f1",
        "doc_constraint_subs",
        "wit_result_subs",
        "none_pct",
        "exact_match",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Statistics
    total_docs = len(rows) - 1  # exclude mean row
    has_result = sum(1 for r in rows[1:] if r[2] and not r[2].startswith("simplify_error") and r[2] not in ("true", "false"))

    print(f"Total entries evaluated: {total_docs}")
    print(f"WIT results available:   {has_result}")
    print(f"WIT results empty/error: {empty_count} ({error_count} simplify errors)")
    print(f"Saved {len(rows)} rows (including mean) to {output_path}")


if __name__ == "__main__":
    config.ensure_output_dirs()
    evaluate_wit(DOC_PATH, WIT_PATH, OUTPUT_PATH)