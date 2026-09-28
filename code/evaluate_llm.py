"""
evaluate_llm.py
Evaluate the similarity between LLM-generated code constraint expressions and the
documented constraint expressions.

The output CSV reports strict/relaxed precision/recall/F1, plus sub-constraint
counts and token statistics.
"""
import json
import csv
import re
import sys
import os

# Determinism guard: SymPy's boolean simplification is sensitive to Python's
# hash randomization, so a random PYTHONHASHSEED would make the simplified
# expressions (and thus the shipped results) non-reproducible. PYTHONHASHSEED
# is only read at interpreter startup, so re-exec once with it fixed to 0.
# Set REPRO_KEEP_HASHSEED=1 to opt out of the automatic re-exec.
if (__name__ == "__main__"
        and os.environ.get("PYTHONHASHSEED") != "0"
        and os.environ.get("REPRO_KEEP_HASHSEED") != "1"):
    os.environ["PYTHONHASHSEED"] = "0"
    print("[evaluate_llm] PYTHONHASHSEED != 0; re-executing with "
          "PYTHONHASHSEED=0 for deterministic results "
          "(set REPRO_KEEP_HASHSEED=1 to disable).")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, *sys.argv])

# Add expression_similarity to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "expression_similarity"))
from expr_sim import calculate_constraint_similarity, count_sub_expressions
from expr_process import simplify_java_bool_expr

# ============================================================
# Configuration
# ============================================================
import config
DOC_PATH = str(config.CODE_DOC_PAIRS)

# Prompt types
PROMPT_TYPES = [
    "vanilla",
    "rand",
    "sr",
    "unst",
    "woex",
    "expl",
    "exim",
    "slextra",
    "slonly",
    "mix",
]

# allconfigs mode: qw7b runs all prompt types (few-shot with all k values)
MODELS_ALLCONFIGS = ["qw7b"]
FEWSHOT_COUNTS_ALL = [1, 2, 4, 8, 16, 32, 64, 128]
# FEWSHOT_COUNTS_ALL = [128]

# allmodels mode: all models run the 5 specified configs
MODELS_ALLMODELS = [
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
# (prompt_type, fewshot_counts): fewshot_counts is the k-list for rand/sr, None otherwise
ALLMODELS_CONFIGS = [
    ("vanilla", None),
    ("sr", [16]),   # sr-16: only k=16
    ("exim", None),
    ("slextra", None),
    ("mix", None),
]

RUN_CONFIGS = [
    "allconfigs",
    "allmodels",
]


def read_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_f1(p: float, r: float) -> float:
    """Harmonic mean (F1), returns 0.0 when p+r == 0."""
    denom = p + r
    return round(2 * p * r / denom, 5) if denom > 0 else 0.0


# Matches a fenced markdown code block (optionally with a language tag).
_FENCE_RE = re.compile(r'```[^\n]*\n(?P<code>.*?)\n```', re.DOTALL)

# Flexion marker: covers "Final Expression:", "final expression is:", etc.
_MARKER_RE = re.compile(r'(?i)final\s+expression\s*(?:is)?\s*:')

# Inline code span: `expr` or ``expr``
_INLINE_CODE_RE = re.compile(r'`+([^`]+)`+')

# A bare '=' that is NOT part of ==, !=, >=, <= (i.e. an assignment statement).
_BARE_ASSIGN_RE = re.compile(r'(?<![<>=!])=(?!=)')

# Java declaration/assignment statements we want to reject (e.g. `int d = Math.max(...)`).
_DECL_STMT_RE = re.compile(
    r'^\s*\b(?:int|long|double|float|boolean|char|byte|short|String|Object|final|void)\b\s[\w\[\]<>]+\s*='
)

# Multi-sentence prose: a period followed by a space and a word (sentence boundary).
_SENTENCE_RE = re.compile(r'\.\s+[A-Za-z]')

# Leading explanation/bullet markers that indicate prose rather than a bare expression.
_PROSE_MARK_RE = re.compile(r'^\s*(?:-\s|\*\s|\*\*|#>?|[A-Za-z]+:\s|\d+[\.\):]\s)')

# Expression markers: comparison / boolean operators, negations, `instanceof`,
# and method-call predicates (e.g. `.equals(...)`, `Files.isDirectory(...)`).
# NOTE: bare literals (true/false/null/none) are NOT included — the word
# "null" appears in prose ("is not null."), and single-token literals are
# already accepted via the no-whitespace rule below.
_EXPR_MARK_RE = re.compile(
    r'(?:&&|\|\||[<>=!]=?|[<>]|\binstanceof\b|[\w$.]+\s*\()'
)


def _is_rejected_candidate(text: str) -> bool:
    """Reject a candidate that is clearly NOT a scalar boolean expression.

    This is deliberately narrow: we only exclude *statements* and *prose*, and
    accept single boolean variables (e.g. `isPositive`), method calls (`.equals()`,
    `Files.isDirectory(...)`), casts (`(int) x > 0`) and operator expressions.
    """
    s = text.strip()
    if not s:
        return True

    # Quoted literals are stripped so an '=' inside a string (e.g.
    # `.contains("=")`) is not mistaken for an assignment, and sentence
    # boundaries / labels inside messages are not mistaken for prose.
    s_no_str = re.sub(r'"[^"]*"', '', s)

    # Statements: terminated by ';', or an assignment ('=' not part of a comparison),
    # or a `throw` statement (never valid inside a boolean expression).
    if s.endswith(';'):
        return True
    if re.search(r'\bthrow\b', s):
        return True
    if _BARE_ASSIGN_RE.search(s_no_str):
        return True
    if _DECL_STMT_RE.search(s):
        return True

    # Multi-sentence prose.
    if _SENTENCE_RE.search(s_no_str):
        return True

    # A boolean expression never ends with a sentence-terminating period
    # (e.g. "This expression represents the overall valid state...").
    if s.endswith('.'):
        return True

    # ... nor with a colon (prose labels like
    # "Combining all relevant constraints using logical AND (`&&`), we get:").
    if s.endswith(':'):
        return True

    # Leading bullets / labels / numbered items indicate prose.
    if _PROSE_MARK_RE.search(s_no_str):
        return True

    # Space-bearing text with no expression operator / call / literal is prose
    # (e.g. "This checks the id is not null."). Single-token bare identifiers
    # (e.g. `isPositive`) are accepted as valid boolean variables.
    if re.search(r'\s', s) and not _EXPR_MARK_RE.search(s):
        return True

    return False


def _cleanup_candidate(text: str) -> str:
    """Strip outer fence, leading junk lines, cut at paragraph break, strip backticks."""
    stripped = text.strip()
    if not stripped:
        return ""

    # Strip an outer markdown fence wrapping the whole candidate
    stripped = re.sub(r'^```[^\n]*\s*\n', '', stripped)
    stripped = re.sub(r'\n```\s*$', '', stripped)
    stripped = stripped.strip()

    # Skip leading junk lines: **, ---, markdown headers, empty lines
    lines = stripped.split('\n')
    cleaned_lines = []
    for line in lines:
        s = line.strip()
        if not s or s in ('**', '---', '***', '*') or s.startswith('#') or re.fullmatch(r'[*_\-]{2,}', s):
            if cleaned_lines:   # only skip leading junk; keep interior blank lines for now
                break
            continue
        cleaned_lines.append(line)
    if not cleaned_lines:
        return ""
    stripped = '\n'.join(cleaned_lines).strip()

    # Cut at the first paragraph break (discard trailing explanation,
    # but preserve multi-line expressions that only have single \n)
    if '\n\n' in stripped:
        stripped = stripped.split('\n\n')[0].strip()

    # Strip redundant leading anchor/trailing backticks
    if stripped.startswith('`') and stripped.endswith('`'):
        inner = stripped.strip('`').strip()
        if inner:
            stripped = inner

    return stripped.strip()


def _enumerate_candidates(text: str) -> list[str]:
    """Enumerate candidate expression fragments in priority order."""
    raw = text
    candidates = []

    # 1. marker pass: text after the LAST "Final Expression:"-style marker
    marker_matches = list(_MARKER_RE.finditer(raw))
    if marker_matches:
        candidates.append(raw[marker_matches[-1].end():].strip())

    # 2. fenced code blocks (any position), last -> first
    fenced = [m.group('code') for m in _FENCE_RE.finditer(raw)]
    candidates.extend(fenced[::-1])

    # 3. inline code spans, last -> first
    inline = [m.group(1) for m in _INLINE_CODE_RE.finditer(raw)]
    candidates.extend(inline[::-1])

    # 4. paragraphs (split on \n\n), last -> first (fallback only; prose is
    #    rejected later by _is_rejected_candidate)
    paragraphs = [p.strip() for p in raw.split('\n\n') if p.strip()]
    candidates.extend(paragraphs[::-1])

    return candidates


def _collapse_multiline_expression(text: str) -> str | None:
    """Fold a multi-line fragment into a single-line expression string.

    Lines continue the expression if the previous collected line ends with a
    continuation operator (e.g. `&&`, `||`, `(`, `,`) or the next line starts
    with one. Stops at the first prose line. Returns None if the result has a
    dangling leading/trailing binary operator.
    """
    s = text.strip()
    if not s:
        return None
    if '\n' not in s:
        s = s.strip()
        if re.search(r'\s*(?:&&|\|\||[=!<>]=?)\s*$', s):
            return None
        return s

    lines = [ln.strip() for ln in s.split('\n') if ln.strip()]
    if not lines:
        return None

    acc = [lines[0]]
    for ln in lines[1:]:
        prev = acc[-1]
        if (re.search(r'(?:&&|\|\||[=!<>]=?|[,(\[])\s*$', prev)
                or re.match(r'^\s*(?:&&|\|\||\)|[,)\]])', ln)):
            acc.append(ln)
        else:
            break

    collapsed = ' '.join(acc).strip()
    if re.match(r'^\s*(?:&&|\|\||[=<>])', collapsed):
        return None
    if re.search(r'\s*(?:&&|\|\||[=!<>]=?)\s*$', collapsed):
        return None
    return collapsed


def _isolate_single_expression(text: str) -> str | None:
    """Reduce a (possibly messy) candidate to exactly ONE clean expression.

    Splits on embedded code-fence markers and re-joins segments that clearly
    continue each other (so an expression interrupted by a stray ``` is
    reconstructed), folds multi-line continuations, then returns the last clean
    expression. Returns None when no fragment is clean (so it becomes a
    parse_failed rather than emitting garbage).
    """
    s = text.strip()
    if not s:
        return None

    # Strip wrapping backticks
    if s.startswith('`') and s.endswith('`'):
        inner = s.strip('`').strip()
        if inner:
            s = inner
        s = s.strip()

    # Split on embedded fence markers: ```lang or ```
    segments = re.split(r'```[^\n]*\n?|\n?```', s)

    # Re-join segments that continue each other across a stray fence.
    groups = []
    cur = None
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if cur is None:
            cur = seg
        elif (re.search(r'(?:&&|\|\||[=!<>]=?|[,(\[])\s*$', cur)
                or re.match(r'^\s*(?:&&|\|\||\)|[,)\]])', seg)):
            cur = cur + ' ' + seg
        else:
            groups.append(cur)
            cur = seg
    if cur is not None:
        groups.append(cur)

    clean = []
    for g in groups:
        collapsed = _collapse_multiline_expression(g)
        if collapsed is None:
            continue
        if _is_rejected_candidate(collapsed):
            continue
        clean.append(collapsed)

    if not clean:
        return None
    return clean[-1]


def validate_format(text: str) -> tuple[int, str]:
    """Minimal syntax pre-checks (bracket balance, trailing dangling operator).

    Returns (parse_failed, sanitized_text) where:
      parse_failed = 0  →  text looks syntactically viable for simplification
      parse_failed = 1  →  obvious syntax violation, skip further processing
    """
    stripped = text.strip()
    if not stripped:
        return 0, ""

    # Bracket balance check
    depth = 0
    for ch in stripped:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth < 0:
            return 1, "parse_error: mismatched parentheses"
    if depth != 0:
        return 1, "parse_error: unmatched opening parenthesis"

    # Trailing dangling operator
    if re.search(r'\s*(?:&&|\|\||[=!]=?|\s[<>]=?)\s*$', stripped):
        return 1, "parse_error: trailing dangling operator"

    return 0, stripped


def simplify_expression(text: str) -> tuple[int, str]:
    """SymPy-based semantic simplification of a *format-validated* expression.

    Returns (parse_failed, simplified_expression).
    """
    stripped = text.strip()

    if not stripped:
        return 0, ""

    # Recognize fallback sentinels
    if stripped.lower() in ("none", "(none)", "true", "(true)", "false", "(false)"):
        return 0, ""

    # Strip backtick wrapping: `expr` or ``expr`` → expr
    if stripped.startswith("`") and stripped.endswith("`"):
        inner = stripped.strip("`").strip()
        if inner:
            stripped = inner

    # SymPy simplification
    try:
        simplified = simplify_java_bool_expr(stripped)
    except Exception as e:
        err_msg = str(e).replace("\n", " ")
        if len(err_msg) > 120:
            err_msg = err_msg[:117] + "..."
        return 1, f"parse_error: {err_msg}"

    return 0, simplified


def try_simplify(text: str) -> tuple[int, str]:
    """Minimal syntax validation → semantic simplification.

    Returns (parse_failed, simplified) where:
      parse_failed = 0  →  simplification succeeded
      parse_failed = 1  →  simplification failed (malformed syntax, etc.)
    """
    pf, sanitized = validate_format(text)
    if pf:
        return pf, sanitized
    return simplify_expression(sanitized)


def extract_expression(text: str) -> tuple[int, str]:
    """Extract the constraint expression: try candidates in priority order and
    return the first one that passes cleanup → validation → simplification.

    Returns (parse_failed, simplified_expression).
    """
    raw = (text or "").strip()
    if not raw:
        return 0, ""

    # Short-circuit the empty-sentinel fallback that some models emit.
    if raw.lower() in ("none", "(none)", "true", "(true)", "false", "(false)"):
        return 0, ""

    last_err = "parse_error: no extractable expression"
    for cand in _enumerate_candidates(raw):
        cleaned = _cleanup_candidate(cand)
        iso = _isolate_single_expression(cleaned) if cleaned else None
        if iso is None:
            # Could not isolate a single clean expression; skip to next candidate.
            continue
        pf, simplified = try_simplify(iso)
        if pf:
            last_err = simplified
            continue
        return 0, simplified

    return 1, last_err


# ============================================================
# Evaluation
# ============================================================

def evaluate(doc_path: str, infer_path: str, output_path: str) -> None:
    # Load data
    doc_data = read_json(doc_path)
    infer_data = read_json(infer_path)

    # Index infer data by id for O(1) lookup
    infer_by_id = {str(item["id"]): item for item in infer_data}

    if len(doc_data) != len(infer_data):
        print(f"WARNING: doc_data ({len(doc_data)}) and infer_data ({len(infer_data)}) "
              f"have different lengths. Proceeding with doc_data ids.")

    rows = []
    parse_fail_count = 0
    for item in doc_data:
        item_id = str(item["id"])
        doc_constraint = item.get("doc_constraint", "")

        # Match infer entry by id
        infer_item = infer_by_id.get(item_id)
        if infer_item is None:
            print(f"WARNING: id={item_id} not found in infer data, skipping.")
            continue

        code_constraint = infer_item.get("text", "")

        parse_failed, code_constraint_simplified = extract_expression(code_constraint)
        if parse_failed:
            parse_fail_count += 1
            print(f"WARNING: parse failed for id={item_id}: {code_constraint_simplified}")

        # --- Similarity (0s if parse failed) ---
        if parse_failed:
            strict_precision = strict_recall = strict_f1 = 0.0
            relaxed_precision = relaxed_recall = relaxed_f1 = 0.0
            code_subs = 0
        else:
            strict_precision = round(
                calculate_constraint_similarity(doc_constraint, code_constraint_simplified,
                                                strict=True, normalize_mode="precision"), 5)
            strict_recall = round(
                calculate_constraint_similarity(doc_constraint, code_constraint_simplified,
                                                strict=True, normalize_mode="recall"), 5)
            strict_f1 = safe_f1(strict_precision, strict_recall)

            relaxed_precision = round(
                calculate_constraint_similarity(doc_constraint, code_constraint_simplified,
                                                strict=False, normalize_mode="precision"), 5)
            relaxed_recall = round(
                calculate_constraint_similarity(doc_constraint, code_constraint_simplified,
                                                strict=False, normalize_mode="recall"), 5)
            relaxed_f1 = safe_f1(relaxed_precision, relaxed_recall)

            code_subs = count_sub_expressions(code_constraint_simplified)

        doc_subs = count_sub_expressions(doc_constraint)

        # --- Token stats ---
        input_tokens = infer_item.get("input_tokens", "")
        output_tokens = infer_item.get("output_tokens", "")
        elapsed_seconds = infer_item.get("elapsed_seconds", "")

        rows.append([
            item_id,
            doc_constraint,
            code_constraint,
            code_constraint_simplified,
            parse_failed,
            strict_precision,
            strict_recall,
            strict_f1,
            relaxed_precision,
            relaxed_recall,
            relaxed_f1,
            doc_subs,
            code_subs,
            input_tokens,
            output_tokens,
            elapsed_seconds,
        ])

    # --- Append mean row ---
    # Text cols: id(0), doc_constraint(1), code_constraint(2), code_constraint_simplified(3)
    # Numeric cols: parse_failed(4), sp(5), sr(6), sf1(7), rp(8), rr(9), rf1(10),
    #               doc_subs(11), code_subs(12), itok(13), otok(14), elap(15)
    num_cols = list(range(4, 16))
    means = []
    for col_idx in num_cols:
        vals = [float(row[col_idx]) for row in rows
                if row[col_idx] != "" and row[col_idx] is not None]
        means.append(round(sum(vals) / len(vals), 3) if vals else "")

    mean_row = ["mean", "", "", ""] + means  # id, doc, code, simp (all text)
    rows.insert(0, mean_row)

    # Data-row numeric columns rendered to 5 decimals (precision/recall/F1).
    # parse_failed(4)/subs(11,12)/tokens(13,14) stay as integers; elapsed_seconds(15)
    # is already 3 decimals from source and is left untouched.
    frac_cols = [5, 6, 7, 8, 9, 10]

    def _fmt_cell(val: str, is_mean: bool, col_idx: int) -> str:
        if val == "" or val is None:
            return val
        if is_mean:
            return f"{float(val):.3f}" if col_idx in num_cols else val
        return f"{float(val):.5f}" if col_idx in frac_cols else val

    # Write CSV
    header = [
        "id",
        "doc_constraint",
        "code_constraint",
        "code_constraint_simplified",
        "parse_failed",
        "strict_precision",
        "strict_recall",
        "strict_f1",
        "relaxed_precision",
        "relaxed_recall",
        "relaxed_f1",
        "doc_constraint_subs",
        "code_constraint_subs",
        "input_tokens",
        "output_tokens",
        "elapsed_seconds",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            is_mean = (row[0] == "mean")
            writer.writerow([_fmt_cell(c, is_mean, ci) for ci, c in enumerate(row)])

    print(f"Parse failures: {parse_fail_count}/{len(rows)}")
    print(f"Saved {len(rows)} rows (including mean) to {output_path}")


# ============================================================
# Dispatch
# ============================================================

def run_config(prompt_type: str, fewshot_counts: list[int] | None, model_names: list[str]) -> None:
    """Run evaluate() for one prompt_type across the given models.

    fewshot_counts is only used for rand/sr; None for other types.
    """
    out_root = config.OUTPUT_METRICS_DIR

    if prompt_type in ("rand", "sr"):
        rel = os.path.join("fewshot", prompt_type)
        for fewshot_count in fewshot_counts:
            for model_name in model_names:
                infer_path = str(config.resolve_raw_file(os.path.join(
                    rel, f"{model_name}_k{fewshot_count}.json")))
                output_path = os.path.join(out_root, rel,
                    f"{model_name}_k{fewshot_count}.csv")
                print(f"\n{'='*60}")
                print(f"Prompt: {prompt_type}  |  k={fewshot_count}  |  Model: {model_name}")
                print(f"{'='*60}")
                evaluate(DOC_PATH, infer_path, output_path)
    elif prompt_type in ("unst", "woex", "expl", "exim"):
        rel = os.path.join("cot", prompt_type)
        for model_name in model_names:
            infer_path = str(config.resolve_raw_file(os.path.join(rel, f"{model_name}.json")))
            output_path = os.path.join(out_root, rel, f"{model_name}.csv")
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt_type} (CoT)  |  Model: {model_name}")
            print(f"{'='*60}")
            evaluate(DOC_PATH, infer_path, output_path)
    elif prompt_type == "mix":
        rel = "mix"
        for model_name in model_names:
            infer_path = str(config.resolve_raw_file(os.path.join(rel, f"{model_name}.json")))
            output_path = os.path.join(out_root, rel, f"{model_name}.csv")
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt_type} (Mix)  |  Model: {model_name}")
            print(f"{'='*60}")
            evaluate(DOC_PATH, infer_path, output_path)
    elif prompt_type in ("slextra", "slonly"):
        rel = os.path.join("slice", prompt_type)
        for model_name in model_names:
            infer_path = str(config.resolve_raw_file(os.path.join(rel, f"{model_name}.json")))
            output_path = os.path.join(out_root, rel, f"{model_name}.csv")
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt_type} (Slice)  |  Model: {model_name}")
            print(f"{'='*60}")
            evaluate(DOC_PATH, infer_path, output_path)
    else:
        # Vanilla
        rel = prompt_type
        for model_name in model_names:
            infer_path = str(config.resolve_raw_file(os.path.join(rel, f"{model_name}.json")))
            output_path = os.path.join(out_root, rel, f"{model_name}.csv")
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt_type}  |  Model: {model_name}")
            print(f"{'='*60}")
            evaluate(DOC_PATH, infer_path, output_path)


if __name__ == "__main__":
    config.ensure_output_dirs()
    for config_name in RUN_CONFIGS:
        print(f"\n{'#'*70}")
        print(f"RUN CONFIG: {config_name}")
        print(f"{'#'*70}")

        if config_name == "allconfigs":
            for prompt_type in PROMPT_TYPES:
                fewshot_counts = (FEWSHOT_COUNTS_ALL if prompt_type in ("rand", "sr") else None)
                run_config(prompt_type, fewshot_counts, MODELS_ALLCONFIGS)
        elif config_name == "allmodels":
            for prompt_type, fewshot_counts in ALLMODELS_CONFIGS:
                run_config(prompt_type, fewshot_counts, MODELS_ALLMODELS)
        else:
            raise ValueError(f"Unknown run config: {config_name}")
