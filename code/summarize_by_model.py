"""
summarize_by_model.py
Aggregate a single model's evaluation metrics across all prompt types into a CSV.
"""
import csv
import os

import config

MODEL_NAMES = [
    "lma8b",
    "qw3b",
    "qw7b",
    "qw14b",
    "qwc7b",
    "gpt4o",
    "gpt4om",
    "sonnet",
    "lma70b",
    "qw32b",
    "qw72b",
]


configs1 = [
    ('vanilla', None, None),
] + [
    ('few-shot', pt, k) for pt in ['rand', 'sr'] for k in [1, 2, 4, 8, 16, 32, 64, 128]
] + [
    ('CoT', ct, None) for ct in ['unst', 'woex', 'expl', 'exim']
] + [
    ('slice', st, None) for st in ['slextra', 'slonly']
] + [
    ('mix', 'mix', None),
]


configs2 = [
    ('vanilla', None, None),
] + [
    ('few-shot', pt, k) for pt in ['sr'] for k in [16]
] + [
    ('CoT', ct, None) for ct in ['exim']
] + [
    ('slice', st, None) for st in ['slextra']
] + [
    ('mix', 'mix', None),
]

def build_label(cat: str, pt: str, k: int) -> str:
    if cat == 'vanilla':
        return 'vanilla'
    if cat == 'few-shot':
        return f'{pt} k={k}'
    if cat == 'CoT':
        return f'cot-{pt}'
    if cat == 'mix':
        return 'mix'
    if cat == 'slice':
        return f'slice-{pt}'
    return ''


def build_path(cat: str, pt: str, k: int, model: str) -> str:
    if cat == 'vanilla':
        rel = os.path.join('vanilla', f'{model}.csv')
    elif cat == 'few-shot':
        rel = os.path.join('fewshot', pt, f'{model}_k{k}.csv')
    elif cat == 'CoT':
        rel = os.path.join('cot', pt, f'{model}.csv')
    elif cat == 'mix':
        rel = os.path.join('mix', f'{model}.csv')
    elif cat == 'slice':
        rel = os.path.join('slice', pt, f'{model}.csv')
    else:
        return ''
    return str(config.resolve_metrics_file(rel))


header = [
    'prompt_type', 'strict_precision', 'strict_recall', 'strict_f1',
    'relaxed_precision', 'relaxed_recall', 'relaxed_f1',
    'doc_constraint_subs', 'code_constraint_subs',
    'input_tokens', 'output_tokens', 'elapsed_seconds',
    'parse_fail_pct', 'none_pct', 'exact_match_pct',
]

for model in MODEL_NAMES:
    rows = []
    if model == "qw7b":
        configs = configs1
    else:
        configs = configs2
    for cat, pt, k in configs:
        path = build_path(cat, pt, k, model)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = list(csv.reader(f))
        except FileNotFoundError:
            print(f'MISSING: {path}')
            continue

        mean = data[1]
        items = data[2:]
        total = len(items)

        none_pct = round(sum(1 for r in items if r[3].strip() == '') / total * 100, 2)
        pf_pct = round(sum(1 for r in items if r[4] == '1') / total * 100, 2)
        # exact_match: entries with strict F1 == 1 (parse-failed entries have F1 0 and are not counted)
        em_pct = round(sum(1 for r in items if r[7] and float(r[7]) == 1.0) / total * 100, 2)

        rows.append([
            build_label(cat, pt, k),
            mean[5],   # strict_precision
            mean[6],   # strict_recall
            mean[7],   # strict_f1
            mean[8],   # relaxed_precision
            mean[9],   # relaxed_recall
            mean[10],  # relaxed_f1
            mean[11],  # doc_constraint_subs
            mean[12],  # code_constraint_subs
            mean[13],  # input_tokens
            mean[14],  # output_tokens
            mean[15],  # elapsed_seconds
            pf_pct,    # parse_fail_pct
            none_pct,  # none_pct
            em_pct,    # exact_match_pct
        ])

    out_dir = str(config.OUTPUT_SUMMARY_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/{model}_all_results.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'{model}: {len(rows)} rows -> {out_path}')
