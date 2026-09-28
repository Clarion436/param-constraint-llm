"""
significance_models.py
Statistical significance testing for comparing evaluation results across methods.

Supports two modes:
  - 2 groups only        → Wilcoxon signed-rank test (no correction needed)
  - >=3 groups           → Friedman test (global) + post-hoc Wilcoxon signed-rank
                           pairwise comparisons with 'fdr_bh' or 'holm' correction

API:
    compare_significance(data_dict, comparisons, correction_method='fdr_bh', metric_name='F1')
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import csv
import os
from itertools import combinations

import config

# ============================================================
# Paths
# ============================================================
WIT_EVAL_CSV = str(config.resolve_wit_file("wit_metrics.csv"))

# Prompt-type → CSV path parts resolution
COT_TYPES = {"unst", "woex", "expl", "exim"}
ALL_COT_TYPES = COT_TYPES
SLICE_TYPES = {"slextra", "slonly"}
MIX_TYPES = {"mix"}
FEWSHOT_STRATEGIES = {"rand", "sr"}


MODELS = [
    "qw3b",
    "qw7b",
    "qw14b",
    "qwc7b",
    "lma8b",
    "lma70b",
    "gpt4o",
    "gpt4om",
    "sonnet",
    "qw32b",
    "qw72b",
]

def _resolve_csv_path(model: str, config_name: str) -> str:
    """Resolve a config_name to the CSV file path produced by evaluate_llm.py.

    Parameters
    ----------
    model : str
        Model name, e.g. 'qw7b', 'lma8b'.
    config_name : str
        Configuration identifier. Recognised patterns:
          - 'vanilla'
          - CoT types: 'unst', 'woex', 'expl', 'exim'
          - Slice types: 'slextra', 'slonly'
          - Few-shot: 'rand_k{1..128}' or 'sr_k{1..128}'
          - 'wit'  → loads from wit_metrics.csv

    Returns
    -------
    str : path to the CSV file.
    """
    if config_name == "vanilla":
        return str(config.resolve_metrics_file(os.path.join("vanilla", f"{model}.csv")))

    if config_name in ALL_COT_TYPES:
        return str(config.resolve_metrics_file(os.path.join("cot", config_name, f"{model}.csv")))

    if config_name in SLICE_TYPES:
        return str(config.resolve_metrics_file(os.path.join("slice", config_name, f"{model}.csv")))

    if config_name in MIX_TYPES:
        return str(config.resolve_metrics_file(os.path.join("mix", f"{model}.csv")))

    # Few-shot: e.g. "rand_4", "sr_128"
    for strat in FEWSHOT_STRATEGIES:
        prefix = f"{strat}_"
        if config_name.startswith(prefix):
            k = config_name[len(prefix):]  # e.g. "4", "128"
            return str(config.resolve_metrics_file(os.path.join("fewshot", strat, f"{model}_k{k}.csv")))

    raise ValueError(f"Unknown config_name: {config_name!r}")


def load_config_f1(model: str, config_name: str, metric: str = "strict_f1") -> np.ndarray:
    """Load the array of per-sample F1 scores for a given model + config.

    Parameters
    ----------
    model : str
        Model name.
    config_name : str
        Configuration identifier (see _resolve_csv_path).
        Special value 'wit' loads from wit_metrics.csv (model is ignored).
    metric : str
        'strict_f1' or 'relaxed_f1'.

    Returns
    -------
    np.ndarray of shape (n_samples,) — the F1 scores (excluding the 'mean' row).
    """
    if config_name == "wit":
        csv_path = WIT_EVAL_CSV
        # WIT CSV column layout:
        # id, doc_constraint, wit_result, wit_paths, miss_reason, wit_path_count,
        # strict_precision, strict_recall, strict_f1,
        # relaxed_precision, relaxed_recall, relaxed_f1, ...
        col_map = {"strict_f1": 8, "relaxed_f1": 11}
    else:
        csv_path = _resolve_csv_path(model, config_name)
        # LLM CSV column layout:
        # id, doc_constraint, code_constraint, code_constraint_simplified,
        # parse_failed, strict_precision, strict_recall, strict_f1,
        # relaxed_precision, relaxed_recall, relaxed_f1, ...
        col_map = {"strict_f1": 7, "relaxed_f1": 10}

    if metric not in col_map:
        raise ValueError(f"Unknown metric {metric!r}. Choose 'strict_f1' or 'relaxed_f1'.")

    col_idx = col_map[metric]

    values = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader)  # skip header
        for row in reader:
            if row[0] == "mean":
                continue  # skip the summary row
            try:
                values.append(float(row[col_idx]))
            except (ValueError, IndexError):
                values.append(np.nan)

    arr = np.array(values, dtype=float)
    # Drop any NaN entries (shouldn't happen, but be safe)
    arr = arr[~np.isnan(arr)]
    return arr


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's Delta effect size (x relative to y).

    Range [-1, 1].  Positive → x tends to be larger than y.
    """
    x, y = np.asarray(x), np.asarray(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    diff = x[:, None] - y
    dominations = np.sum(diff > 0) - np.sum(diff < 0)
    return dominations / (n_x * n_y)


def _effect_size_label(abs_delta: float) -> str:
    """Map absolute Cliff's Delta to a semantic label."""
    if abs_delta < 0.147:
        return "Negligible"
    if abs_delta < 0.33:
        return "Small"
    if abs_delta < 0.474:
        return "Medium"
    return "Large"


def compare_significance(
    data_dict: dict,
    comparisons: list,
    correction_method: str = "fdr_bh",
    metric_name: str = "F1",
) -> tuple[pd.DataFrame | None, dict]:
    """Statistical significance comparison across methods.

    When ``data_dict`` has **2 keys**: performs a single Wilcoxon signed-rank test
    (no global test or correction).

    When ``data_dict`` has **>=3 keys**: performs Friedman test as a global
    omnibus check, then post-hoc Wilcoxon signed-rank tests on the specified
    ``comparisons`` pairs with the chosen ``correction_method``.

    Parameters
    ----------
    data_dict : dict[str, np.ndarray]
        Mapping from method/configuration name to its per-sample F1 scores.
        All arrays must have the same length and be pairwise aligned.
    comparisons : list[tuple[str, str]]
        List of (method_A, method_B) pairs to test.
        For 2-group mode this list is ignored (the two keys are compared);
        for >=3 groups only the pairs listed here are tested post-hoc.
    correction_method : str
        'fdr_bh' (Benjamini-Hochberg) or 'holm' (Holm-Bonferroni).
        Only used when there are >=3 groups.
    metric_name : str
        Label used in column headers (e.g. 'Strict F1', 'Relaxed F1').

    Returns
    -------
    (DataFrame | None, dict)
        DataFrame with pairwise results (or None if global test not significant),
        and a dict with global test details.
    """
    names = list(data_dict.keys())
    arrays = list(data_dict.values())
    n_groups = len(names)
    n_samples = len(arrays[0])

    # Validate equal lengths
    for name, arr in zip(names, arrays):
        if len(arr) != n_samples:
            raise ValueError(
                f"Array length mismatch: '{name}' has {len(arr)} samples, "
                f"expected {n_samples}."
            )

    print(f"--- Metric: {metric_name} ---")
    print(f"--- Groups: {n_groups}  |  Samples (N): {n_samples} ---")

    # ============================================================
    # 2-group mode: Wilcoxon only, no correction
    # ============================================================
    if n_groups == 2:
        print("--- Mode: Wilcoxon signed-rank test (2 groups, no global test) ---\n")
        a_name, b_name = names[0], names[1]
        data_a, data_b = arrays[0], arrays[1]

        _, p_w = stats.wilcoxon(data_a, data_b, zero_method="zsplit")
        delta = cliffs_delta(data_a, data_b)

        result = {
            "Comparison": f"{a_name} vs {b_name}",
            f"Mean {metric_name} (A)": round(np.mean(data_a), 4),
            f"Mean {metric_name} (B)": round(np.mean(data_b), 4),
            "Raw p-value": p_w,
            "Corrected p-value": p_w,  # no correction needed
            "Significant": p_w < 0.05,
            "Cliff's Delta": round(delta, 4),
            "Effect Size": _effect_size_label(abs(delta)),
        }
        df = pd.DataFrame([result])
        print(f"Wilcoxon p = {p_w:.4e}  |  Significant: {p_w < 0.05}")
        global_info = {
            "test": "Wilcoxon signed-rank",
            "n_groups": 2,
            "n_samples": n_samples,
        }
        return df, global_info

    # ============================================================
    # >=3 group mode: Friedman global + post-hoc Wilcoxon
    # ============================================================
    print(f"--- Comparisons: {len(comparisons)} planned pairs ---")
    print(f"--- Correction: {correction_method.upper()} ---\n")

    # Phase 1: Friedman global test
    stat_f, p_friedman = stats.friedmanchisquare(*arrays)
    print(f"[Global] Friedman chi2 = {stat_f:.2f}, p = {p_friedman:.4e}")

    if p_friedman >= 0.05:
        print("Friedman test NOT significant — but still running post-hoc Wilcoxon tests.\n")
    else:
        print("Friedman test significant → proceeding with post-hoc Wilcoxon tests.\n")

    # Phase 2: Post-hoc Wilcoxon
    results = []
    p_values_raw = []

    for method_a, method_b in comparisons:
        if method_a not in data_dict:
            raise KeyError(f"'{method_a}' not in data_dict. Available: {list(data_dict.keys())}")
        if method_b not in data_dict:
            raise KeyError(f"'{method_b}' not in data_dict. Available: {list(data_dict.keys())}")

        data_a = data_dict[method_a]
        data_b = data_dict[method_b]

        _, p_w = stats.wilcoxon(data_a, data_b, zero_method="zsplit")
        delta = cliffs_delta(data_a, data_b)

        p_values_raw.append(p_w)
        results.append({
            "Comparison": f"{method_a} vs {method_b}",
            f"Mean {metric_name} (A)": round(np.mean(data_a), 4),
            f"Mean {metric_name} (B)": round(np.mean(data_b), 4),
            "Raw p-value": p_w,
            "Cliff's Delta": round(delta, 4),
        })

    # Phase 3: Multiple-comparison correction
    reject, pvals_corrected, _, _ = multipletests(
        p_values_raw, alpha=0.05, method=correction_method,
    )

    for i, res in enumerate(results):
        res["Corrected p-value"] = pvals_corrected[i]
        res["Significant"] = reject[i]
        res["Effect Size"] = _effect_size_label(abs(res["Cliff's Delta"]))

    df = pd.DataFrame(results)
    column_order = [
        "Comparison",
        f"Mean {metric_name} (A)", f"Mean {metric_name} (B)",
        "Raw p-value", "Corrected p-value", "Significant",
        "Cliff's Delta", "Effect Size",
    ]
    global_info = {
        "test": "Friedman",
        "n_groups": n_groups,
        "n_samples": n_samples,
        "chi2": round(stat_f, 4),
        "p_value": p_friedman,
        "significant": p_friedman < 0.05,
        "correction": correction_method.upper(),
    }
    return df[column_order], global_info


# ============================================================
# Comparison group definitions
# ============================================================
# Each group is a dict with:
#   name        : str  — label for console output and CSV section header
#   global      : list[str] — config names for the global (Friedman) test
#   comparisons : list[tuple[str,str]] — pairs for post-hoc Wilcoxon tests
#
# Use  list(combinations(global, 2))  for all pairwise, or a hand-picked list.


# Main comparison uses exim / mix
ANALYSES = [
    {
        "name": "Main comparison",
        "global": ["vanilla", "sr_16", "exim", "slextra", "mix"],
        "comparisons": [
            ("vanilla", "sr_16"),
            ("vanilla", "exim"),
            ("vanilla", "slextra"),
            ("vanilla", "mix"),
            # ("sr_16", "exim"),
            # ("sr_16", "slextra"),
            # ("exim", "slextra"),
            ],
    },
    {
        "name": "Vanilla vs WIT",
        "global": ["vanilla", "wit"],
        "comparisons": [("vanilla", "wit")],
    },
]


def run_comparison_group(
    group: dict,
    model: str,
    metric: str,
    correction_method: str,
) -> tuple[pd.DataFrame | None, dict]:
    """Run a single comparison group.

    Returns (pairwise_df, global_info_dict).
    """
    name = group["name"]
    global_configs = group["global"]
    comparisons = group["comparisons"]
    metric_label = "Strict F1" if metric == "strict_f1" else "Relaxed F1"

    print(f"{'='*65}")
    print(f"Group: {name}  |  Model: {model}  |  Metric: {metric}")
    print(f"{'='*65}\n")

    data_dict = {}
    for config in global_configs:
        data_dict[config] = load_config_f1(model, config, metric=metric)
        print(f"  Loaded {config:10s} → mean {metric} = {np.mean(data_dict[config]):.4f}")

    print()
    df, global_info = compare_significance(
        data_dict=data_dict,
        comparisons=comparisons,
        correction_method=correction_method,
        metric_name=metric_label,
    )

    if df is not None:
        print(f"\n[{name} — {model}]")
        print(df.to_string(formatters={
            f"Mean {metric_label} (A)": "{:.4f}".format,
            f"Mean {metric_label} (B)": "{:.4f}".format,
            "Raw p-value": "{:.4e}".format,
            "Corrected p-value": "{:.4e}".format,
            "Cliff's Delta": "{:.4f}".format,
        }, index=False))
    return df, global_info


# ============================================================
# Main
# ============================================================
COLUMN_ORDER = [
    "Model", "Analysis", "N", "Groups", "Correction", "Metric", "Comparison",
    "Mean F1 (A)", "Mean F1 (B)",
    "Raw p-value", "Corrected p-value", "Significant",
    "Cliff's Delta", "Effect Size", "Global chi2",
]


def _build_result_rows(
    group: dict, model: str, metric: str, correction: str,
) -> list[pd.DataFrame]:
    """Return [global_row, pairwise_df] for one metric in one group."""
    df, global_info = run_comparison_group(
        group=group, model=model, metric=metric, correction_method=correction,
    )
    metric_label = "Strict F1" if metric == "strict_f1" else "Relaxed F1"

    g = global_info
    global_row = pd.DataFrame([{
        "Model": "",
        "Analysis": "",
        "N": "",
        "Groups": "",
        "Correction": "",
        "Metric": metric_label,
        "Comparison": f"{g['test']} (global)",
        "Mean F1 (A)": "",
        "Mean F1 (B)": "",
        "Raw p-value": g.get("p_value", ""),
        "Corrected p-value": "",
        "Significant": g.get("significant", ""),
        "Cliff's Delta": "",
        "Effect Size": "",
        "Global chi2": g.get("chi2", ""),
    }])

    if df is not None:
        sig_count = df["Significant"].sum()
        print(f"\n  -> {sig_count}/{len(df)} comparisons significant after correction.")

        mean_a_col = f"Mean {metric_label} (A)"
        mean_b_col = f"Mean {metric_label} (B)"
        df = df.rename(columns={mean_a_col: "Mean F1 (A)", mean_b_col: "Mean F1 (B)"})

        df.insert(0, "Analysis", "")
        df.insert(1, "N", "")
        df.insert(2, "Groups", "")
        df.insert(3, "Correction", "")
        df.insert(4, "Metric", metric_label)
        df["Model"] = ""
        df["Global chi2"] = ""
        return [global_row, df]
    else:
        return [global_row]


if __name__ == "__main__":
    
    CORRECTION = "holm"

    for model in MODELS:
        print(f"\n{'#'*65}")
        print(f"#  MODEL: {model}  |  Correction: {CORRECTION.upper()}")
        print(f"{'#'*65}\n")

        all_parts = []
        OUTPUT_DIR = str(config.OUTPUT_SIGNIFICANCE_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        OUTPUT_CSV = f"{OUTPUT_DIR}/significance_{model}_{CORRECTION}.csv"

        for group in ANALYSES:
            group_name = group["name"]

            # Determine N from first config in the group (use strict_f1 to probe)
            try:
                probe = load_config_f1(model, group["global"][0], metric="strict_f1")
                n_samples = len(probe)
            except Exception:
                n_samples = "?"

            # Section header row
            header_row = pd.DataFrame([{
                "Model": model,
                "Analysis": group_name,
                "N": n_samples,
                "Groups": len(group["global"]),
                "Correction": CORRECTION.upper(),
                "Metric": "", "Comparison": "",
                "Mean F1 (A)": "", "Mean F1 (B)": "",
                "Raw p-value": "", "Corrected p-value": "", "Significant": "",
                "Cliff's Delta": "", "Effect Size": "", "Global chi2": "",
            }])

            strict_parts = _build_result_rows(group, model, "strict_f1", CORRECTION)
            relaxed_parts = _build_result_rows(group, model, "relaxed_f1", CORRECTION)
            print("\n")

            blank = pd.DataFrame([{c: "" for c in header_row.columns}])
            all_parts.append(header_row)
            all_parts.extend(strict_parts)
            all_parts.append(blank)
            all_parts.extend(relaxed_parts)
            all_parts.append(blank)

        if all_parts:
            combined = pd.concat(all_parts, ignore_index=True)
            combined = combined[COLUMN_ORDER]
            combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            print(f"Results saved to {OUTPUT_CSV}")
