"""
expr_sim.py — expression-similarity utility

Semantic preprocessing + atom matching + hierarchical set matching:
  Part 1: Java semantic conversion (.isEmpty->LEN, .equals->==, etc.)
  Part 2: atomic-expression similarity (operand edit distance + operator Jaccard + flip matching)
  Part 3: constraint-set hierarchical matching (recursive &&/|| parsing -> Hungarian algorithm -> Recall/Precision)

Note: input expressions should already be simplified by expr_process.simplify_java_bool_expr.
"""

import re
import numpy as np
from scipy.optimize import linear_sum_assignment



# ============================================================
# Utility: strip parentheses
# ============================================================

def _strip_outer_parentheses(text: str) -> str:
    """Safely strip redundant outermost parentheses from an expression."""
    text = text.strip()
    while text.startswith('(') and text.endswith(')'):
        paren_level = 0
        is_wrapper = True
        for char in text[1:-1]:
            if char == '(':
                paren_level += 1
            elif char == ')':
                paren_level -= 1
            if paren_level < 0:
                is_wrapper = False
                break
        if is_wrapper and paren_level == 0:
            text = text[1:-1].strip()
        else:
            break
    return text


# ============================================================
# Part 1: expression preprocessing and normalization
# ============================================================

def _convert_isEmpty(expr: str) -> str:
    """Convert .isEmpty() to (var.LEN == 0)."""
    pattern = re.compile(r"""
        (?P<var>
            [\w\.]+
            (?:
                \s* \. [\w]+
                (?: \s* \( \s* \) )?
            )* )
        \.isEmpty\(\)
    """, re.VERBOSE)
    return pattern.sub(r'(\g<var>.LEN == 0)', expr)


def _convert_isBlank(expr: str) -> str:
    """Handle .isBlank(): replace with .trim().isEmpty(), then reuse _convert_isEmpty."""
    preprocessed = re.sub(r'\.isBlank\(\)', '.trim().isEmpty()', expr)
    return _convert_isEmpty(preprocessed)


def _convert_length_size(expr: str) -> str:
    """Convert .length, .length(), .size() uniformly to the abstract property .LEN."""
    pattern = re.compile(r"""
        (?P<var>
            [\w\.]+
            (?:
                \s* \. [\w]+
                (?: \s* \( \s* \) )?
            )* )
        (?P<accessor>
            \.length\(\)   |
            \.size\(\)     |
            \.length\b
        )
    """, re.VERBOSE)
    return pattern.sub(r'\g<var>.LEN', expr)


def _convert_equals(expr: str) -> str:
    """Convert .equals() to (A == B)."""
    head_pattern = re.compile(r"""
        (?P<var>
            (?:
                "(?:\\.|[^"\\])*"
                |
                [\w\.]+(?:\s*\(\s*\)|\.\w+)*
            )
        )
        \.equals\s*\(
    """, re.VERBOSE)

    replacements = []
    for match in head_pattern.finditer(expr):
        start_arg = match.end()
        paren_level = 1
        in_quote = False
        quote_char = None
        end_arg = -1

        current_idx = start_arg
        while current_idx < len(expr):
            char = expr[current_idx]
            if char in ('"', "'"):
                if current_idx > 0 and expr[current_idx - 1] != '\\':
                    if not in_quote:
                        in_quote, quote_char = True, char
                    elif char == quote_char:
                        in_quote = False

            if not in_quote:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                    if paren_level == 0:
                        end_arg = current_idx
                        break
            current_idx += 1

        if end_arg == -1:
            continue

        argument = expr[start_arg:end_arg].strip()
        variable = match.group('var')
        new_code = f"({variable} == {argument})"
        replacements.append((match.start(), end_arg + 1, new_code))

    new_expr_list = list(expr)
    for start, end, txt in reversed(replacements):
        new_expr_list[start:end] = list(txt)
    return "".join(new_expr_list)


def _convert_instanceof(expr: str) -> str:
    """Convert the instanceof operator to the form .TYPE == ClassName."""
    pattern = re.compile(r"""
        (?P<var>
            [\w\.]+
            (?: \s* \( \s* \) )?
            (?:
                \s* \. [\w]+
                (?: \s* \( \s* \) )?
            )* )
        \s+ instanceof \s+
        (?P<cls>[\w\.]+)
    """, re.VERBOSE)
    return pattern.sub(r'\g<var>.TYPE == \g<cls>', expr)


def _convert_commons_utils(expr: str) -> str:
    """Convert Apache Commons static utility calls to equivalent standard forms.

    StringUtils.isEmpty(str)     → (str == null || str.isEmpty())
    StringUtils.isBlank(str)     → (str == null || str.isBlank())
    CollectionUtils.isEmpty(c)   → (c == null || c.isEmpty())
    ArrayUtils.isEmpty(arr)      → (arr == null || arr.length == 0)
    ObjectUtils.isEmpty(obj)     → (obj == null || obj.isEmpty())
    """
    # Match: (UtilsClass.isEmpty|isBlank)(var)
    pattern = re.compile(r"""
        (?P<cls>StringUtils|CollectionUtils|ArrayUtils|ObjectUtils)
        \.
        (?P<method>is(?:Empty|Blank))
        \s*\(\s*
        (?P<var>
            [\w\.]+
            (?: \s* \( \s* \) )?
            (?:
                \s* \. [\w]+
                (?: \s* \( \s* \) )?
            )*
        )
        \s*\)
    """, re.VERBOSE)

    def _replacer(match):
        cls = match.group('cls')
        method = match.group('method')
        var = match.group('var')

        if cls == 'ObjectUtils' and method == 'isEmpty':
            return f"({var} == null || {var}.isEmpty())"
        if cls == 'StringUtils':
            if method == 'isEmpty':
                return f"({var} == null || {var}.isEmpty())"
            if method == 'isBlank':
                return f"({var} == null || {var}.isBlank())"
        if cls == 'CollectionUtils' and method == 'isEmpty':
            return f"({var} == null || {var}.isEmpty())"
        if cls == 'ArrayUtils' and method == 'isEmpty':
            return f"({var} == null || {var}.length == 0)"
        return match.group(0)  # unrecognised → keep original

    return pattern.sub(_replacer, expr)


def _convert_bare_this_methods(expr: str) -> str:
    """Add the `this.` prefix to bare instance-method calls.

    Only methods known to appear bare in constraint expressions are covered, to avoid over-prefixing.
    """
    bare_methods = ['length', 'size', 'isEmpty', 'isBlank']
    for m in bare_methods:
        # Match: (not at a word boundary) method() — i.e. a bare call with no preceding .
        # \b must not be preceded by .
        expr = re.sub(rf'(?<!\.)\b{m}\(\)', f'this.{m}()', expr)
    return expr


def expression_preConvert(expr: str) -> str:
    """Main conversion function: run all preprocessing steps in order."""
    expr = _convert_commons_utils(expr)
    expr = _convert_bare_this_methods(expr)
    expr = _convert_equals(expr)
    expr = _convert_instanceof(expr)
    expr = _convert_length_size(expr)
    expr = _convert_isBlank(expr)
    expr = _convert_isEmpty(expr)
    return expr


def _normalize_sentinel(expr: str) -> str:
    """Convert boolean literal sentinels ('true', 'false', 'None') to '' (no constraints).

    - 'true' / 'false': legacy fallbacks, or SymPy-simplified contradictions.
    - 'None': current prompt's fallback when no constraints are found.
    """
    stripped = _strip_outer_parentheses(expr)
    if stripped.lower() in ("true", "false", "none"):
        return ""
    return expr


def normalize_expression(expr: str) -> str:
    """Normalization pipeline: handle sentinel literals, then Java→canonical form."""
    expr = _normalize_sentinel(expr)
    return expression_preConvert(expr)




# ============================================================
# Part 2: atomic-expression similarity
# ============================================================


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein edit distance."""
    dp = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(s1)][len(s2)]


def _normalized_levenshtein_similarity(str1: str, str2: str) -> float:
    """Normalized edit-distance similarity."""
    lev_distance = _levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 0.0
    return 1 - lev_distance / max_len

# Operator semantic sets — decompose comparison operators into atomic relation elements {<, =, >}
operator_sets = {
    "==": frozenset(["="]),
    "!=": frozenset(["<", ">"]),
    "<=": frozenset(["<", "="]),
    ">=": frozenset([">", "="]),
    "<":  frozenset(["<"]),
    ">":  frozenset([">"]),
}

# Weights of operator atomic elements: direction features outweigh boundary features, so direction mismatches are penalized more and boundary differences are tolerated more
# Currently set to 3x; < vs <= : 0.75
"""
        ==      !=       <      <=       >      >=
  ==  1.0000  0.0000  0.0000  0.2500  0.0000  0.2500
  !=  0.0000  1.0000  0.5000  0.4286  0.5000  0.4286
  <   0.0000  0.5000  1.0000  0.7500  0.0000  0.0000
  <=  0.2500  0.4286  0.7500  1.0000  0.0000  0.1429
  >   0.0000  0.5000  0.0000  0.0000  1.0000  0.7500
  >=  0.2500  0.4286  0.0000  0.1429  0.7500  1.0000
"""

ELEMENT_WEIGHTS = {
    "<": 3.0,
    ">": 3.0,
    "=": 1.0,
}


def _weighted_jaccard_similarity_for_operators(op1: str, op2: str) -> float:
    """Compute the semantic similarity of two operators using weighted Jaccard similarity."""
    set1 = operator_sets.get(op1)
    set2 = operator_sets.get(op2)
    if set1 is None or set2 is None:
        return 0.0 if op1 != op2 else 1.0

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return 1.0

    weighted_intersection = sum(ELEMENT_WEIGHTS[e] for e in intersection)
    weighted_union = sum(ELEMENT_WEIGHTS[e] for e in union)

    return weighted_intersection / weighted_union


def _split_to_left_op_right(expr: str) -> tuple[str, str, str]:
    """Parse an atomic expression into a (left operand, operator, right operand) triple.
    Operators inside string literals are handled correctly (not used as split points).
    """
    operators = sorted(operator_sets.keys(), key=len, reverse=True)
    n = len(expr)
    i = 0
    in_single = False
    in_double = False
    escape = False

    while i < n:
        ch = expr[i]

        if escape:
            escape = False
            i += 1
            continue

        if ch == '\\':
            escape = True
            i += 1
            continue

        if in_single:
            if ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == "'":
            in_single = True
            i += 1
            continue

        if ch == '"':
            in_double = True
            i += 1
            continue

        # Skip shift operators << >> >>> so they are not mis-matched by < >
        if ch in ('<', '>') and i + 1 < n and expr[i + 1] in ('<', '>'):
            if expr.startswith('>>>', i):
                i += 3
                continue
            if expr.startswith('<<', i) or expr.startswith('>>', i):
                i += 2
                continue

        # Not inside a string: check for an operator
        for op in operators:
            if expr.startswith(op, i):
                left = expr[:i].strip()
                right = expr[i + len(op):].strip()
                return left, op, right

        i += 1

    return "", "", ""


def _calculate_triple_similarity(l1, op1, r1, l2, op2, r2) -> float:
    """Base similarity of two (left, op, right) triples; all three coefficients are 1/3: (left + operator + right) / 3."""
    sim_operand_left = _normalized_levenshtein_similarity(l1, l2) 
    sim_operand_right = _normalized_levenshtein_similarity(r1, r2)
    sim_op = _weighted_jaccard_similarity_for_operators(op1, op2)
    return (sim_operand_left + sim_op + sim_operand_right) / 3


def _normalize_operand(opnd: str) -> str:
    """Normalize numeric operands: drop Java type suffixes (f/d/l) and trailing zeros after the decimal point."""
    # Drop the Java numeric type suffix
    stripped = re.sub(r'(?i)([0-9.]+)[fdl]$', r'\1', opnd.strip())
    if stripped == opnd.strip():
        stripped = opnd.strip()
    # If it is a float, drop trailing zeros
    if '.' in stripped:
        stripped = stripped.rstrip('0')
        if stripped.endswith('.'):
            stripped = stripped[:-1]  # 1.0 → 1
        elif stripped == '' or stripped == '-':
            stripped = '0'  # defensive: "000" becomes empty after stripping
    return stripped


# Mirror operators — used to handle the equivalent flip a < b == b > a
mirror_operators = {
    "<": ">", ">": "<",
    "<=": ">=", ">=": "<=",
    "==": "==", "!=": "!=",
}

def _atomic_expression_similarity_relaxed(p1: str, p2: str) -> float:
    """
    Compute the semantic similarity of atomic expressions (Relaxed mode).
    For each side: generate the original + flipped forms -> for each form generate semantic-equivalent candidates -> take the maximum over all combinations.
    Input must already be a normalized binary atomic expression.
    """
    l1, op1, r1 = _split_to_left_op_right(p1)
    l2, op2, r2 = _split_to_left_op_right(p2)

    # Operand normalization: drop Java numeric type suffixes and trailing zeros
    l1, r1 = _normalize_operand(l1), _normalize_operand(r1)
    l2, r2 = _normalize_operand(l2), _normalize_operand(r2)

    if not op1 or not op2:
        # print(f"Warning: Unable to parse operators in expressions: "
        #       f"'{p1}' or '{p2}'. Falling back to Levenshtein similarity.")
        return _normalized_levenshtein_similarity(p1, p2)

    # Early exit: the triples are identical, or identical after flipping
    if (l1, op1, r1) == (l2, op2, r2):
        return 1.0
    if (l1, op1, r1) == (r2, mirror_operators[op2], l2):
        return 1.0

    # Generate semantic-equivalent candidates (assume canonical positions: .LEN on the left, bool on the right; flipping is handled by the caller)
    def _candidates(l, op, r):
        yield (l, op, r)
        # Equivalent variants of ==/!= true/false (bool on the right)
        if r in ('true', 'false') and op in ('==', '!='):
            yield (l, '!=' if op == '==' else '==', 'false' if r == 'true' else 'true')
        # Non-emptiness equivalences for .LEN: .LEN > 0 == .LEN != 0 == .LEN >= 1 (container size is non-negative)
        if l.strip().endswith('.LEN'):
            if op == '>' and r == '0':
                yield (l, '!=', '0')
                yield (l, '>=', '1')
            elif op == '!=' and r == '0':
                yield (l, '>', '0')
                yield (l, '>=', '1')
            elif op == '>=' and r == '1':
                yield (l, '>', '0')
                yield (l, '!=', '0')

    # Generate all forms of each expression: original + flipped (swap sides + mirror operator)
    def _forms(l, op, r):
        yield (l, op, r)
        yield (r, mirror_operators[op], l)

    best = 0.0
    for fl1, fop1, fr1 in _forms(l1, op1, r1):
        for fl2, fop2, fr2 in _forms(l2, op2, r2):
            for cl1, cop1, cr1 in _candidates(fl1, fop1, fr1):
                for cl2, cop2, cr2 in _candidates(fl2, fop2, fr2):
                    best = max(best, _calculate_triple_similarity(
                        cl1, cop1, cr1, cl2, cop2, cr2))

    return round(best, 4)


def _atomic_expression_similarity_strict(p1: str, p2: str) -> float:
    """
    Compute the strict similarity of atomic expressions.
    Return 1.0 only on a full semantic match (>= 0.9999), otherwise 0.0.
    """
    if p1 == p2:
        return 1.0
    if p1.replace(" ", "") == p2.replace(" ", ""):
        return 1.0

    sim = _atomic_expression_similarity_relaxed(p1, p2)
    if sim >= 0.9999:
        return 1.0
    return 0.0


def atomic_expression_similarity(p1: str, p2: str, strict: bool) -> float:
    """Main entry point for atomic-expression similarity.
    Input must already be a normalized binary atomic expression."""
    if strict:
        return _atomic_expression_similarity_strict(p1, p2)
    else:
        return _atomic_expression_similarity_relaxed(p1, p2)


# ============================================================
# Part 3: constraint-set hierarchical matching
# ============================================================

def _find_top_level_split_points(text: str) -> tuple:
    """
    Find the outermost split point of an expression, honoring && > || precedence.
    Protect || or && inside string literals from being split.
    """

    def get_split_indices(target_op: str):
        indices = []
        paren_level = 0
        in_quote = False
        quote_char = None
        i = 0
        n = len(text)
        while i < n:
            char = text[i]
            if char in ('"', "'"):
                if i == 0 or text[i - 1] != '\\':
                    if not in_quote:
                        in_quote = True
                        quote_char = char
                    elif char == quote_char:
                        in_quote = False
            if not in_quote:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                elif paren_level == 0:
                    if text[i:i + 2] == target_op:
                        indices.append(i)
            i += 1
        return indices

    # Look for || first (lowest precedence)
    or_indices = get_split_indices('||')
    if or_indices:
        operator = '||'
        last_split = 0
        components = []
        for point in or_indices:
            components.append(text[last_split:point].strip())
            last_split = point + 2
        components.append(text[last_split:].strip())
        return operator, components

    # Then look for &&
    and_indices = get_split_indices('&&')
    if and_indices:
        operator = '&&'
        last_split = 0
        components = []
        for point in and_indices:
            components.append(text[last_split:point].strip())
            last_split = point + 2
        components.append(text[last_split:].strip())
        return operator, components

    return None, [text]


# Logical negation operator map (for negation, not flipping)
_OP_INVERT = {
    '==': '!=',  '!=': '==',
    '>':  '<=',  '>=': '<',
    '<':  '>=',  '<=': '>',
}


def _negate_atom(inner: str) -> str:
    """Negate a normalized binary atomic expression: !(a > b) -> a <= b.
    If inner cannot be parsed as a binary comparison (e.g. a pure boolean symbol or malformed input),
    wrap it as !(inner).
    """
    if inner == "true":
        return "false"
    if inner == "false":
        return "true"
    l, op, r = _split_to_left_op_right(inner)
    if not op:
        return f"!{inner}"
    return f"{l} {_OP_INVERT[op]} {r}"


def normalize_unary_boolean(atom: str) -> str:
    """Turn unary expressions into binary form, ensuring every atom is a binary comparison."""
    """Both negation and conversion to binary can only happen after parsing the expression;
    the preprocessing step may use regex matching without changing semantics, but it cannot
    handle negation and binary conversion without parsing."""
    atom = atom.strip()
    if not atom:
        return atom

    # true / false literals -> unchanged
    if atom in ('true', 'false'):
        return atom

    # !expression -> recursively normalize the inside, then negate
    if atom.startswith('!'):
        inner = atom[1:].strip()
        if inner.startswith('('):
            inner = _strip_outer_parentheses(inner)
        inner = normalize_unary_boolean(inner)
        return _negate_atom(inner)

    # Already a binary comparison -> unchanged; bare atoms (boolean vars, method calls, etc.) stay unchanged,
    # and if preceded by !, _negate_atom wraps it as !(atom) and it goes through Levenshtein comparison.
    return atom


def parse_expression(text: str):
    """
    Recursively parse a raw constraint string into a structured nested-tuple format.
    Leaves are normalized from unary to binary form via normalize_unary_boolean.
    """
    text = text.strip()
    text = _strip_outer_parentheses(text)

    operator, components = _find_top_level_split_points(text)

    if operator is None:
        if components[0] == '':
            return ''
        return normalize_unary_boolean(components[0])
    else:
        parsed_components = [parse_expression(comp) for comp in components]

        # Flatten operators at the same level
        flattened = []
        for pc in parsed_components:
            if isinstance(pc, tuple) and len(pc) >= 1 and pc[0] == operator:
                flattened.extend(list(pc[1:]))
            else:
                flattened.append(pc)

        # Deduplicate (preserve first-occurrence order)
        seen = set()
        unique_components = []
        for item in flattened:
            try:
                key = item
                if key not in seen:
                    unique_components.append(item)
                    seen.add(key)
            except TypeError:
                key = repr(item)
                if key not in seen:
                    unique_components.append(item)
                    seen.add(key)

        if len(unique_components) == 1:
            return unique_components[0]

        return tuple([operator] + unique_components)


def _split_by_layer(parsed_expr, layer: int) -> list:
    """
    Split a parsed expression into component sets according to the operator at the current level.
    Odd levels handle AND; even levels handle OR.
    """
    layer_operator = '&&' if layer % 2 == 1 else '||'

    if parsed_expr == '':
        return []

    if not isinstance(parsed_expr, tuple) or parsed_expr[0] != layer_operator:
        return [parsed_expr]
    else:
        return list(parsed_expr[1:])


def _recursive_set_match(expr1, expr2, layer: int, strict: bool,
                         normalize_mode: str) -> float:
    """
    Core recursive function of the hierarchical set-matching algorithm.
    normalize_mode: "recall" or "precision"
    """
    if normalize_mode not in ("recall", "precision"):
        raise ValueError(
            f"Normalize_mode should be one of ['recall', 'precision'], "
            f"but got: {normalize_mode}"
        )

    # --- 1. Recursion base: both sides are atomic expressions ---
    if not isinstance(expr1, tuple) and not isinstance(expr2, tuple):
        if expr1 == '' or expr2 == '':
            pass  # fall through to the split logic below
        else:
            return atomic_expression_similarity(expr1, expr2, strict)

    # --- 2. Split the current level ---
    set1 = _split_by_layer(expr1, layer)
    set2 = _split_by_layer(expr2, layer)

    len_1 = len(set1)
    len_2 = len(set2)

    # --- 3. Handle edge cases ---
    if len_1 == 0 and len_2 == 0:
        return 1.0

    if len_1 == 0 and len_2 > 0:
        if normalize_mode == "recall":
            # return 1.0
            # Conventional handling
            return 0.0
        else:
            return 0.0

    if len_1 > 0 and len_2 == 0:
        if normalize_mode == "precision":
            # return 1.0
            # Conventional handling: when the reference set is non-empty and the predicted set is empty, precision is 0, avoiding inflated macro-averaged precision
            return 0.0
        else:
            return 0.0

    # --- 4. Build the cost matrix ---
    cost_matrix = np.ones((len_1, len_2))
    for i, e1 in enumerate(set1):
        for j, e2 in enumerate(set2):
            similarity = _recursive_set_match(e1, e2, layer + 1, strict, normalize_mode)
            cost_matrix[i, j] = 1.0 - similarity

    # --- 5. Optimal matching via the Hungarian algorithm ---
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    min_cost_sum = cost_matrix[row_ind, col_ind].sum()
    matched_count = len(row_ind)
    max_weight_sum = matched_count - min_cost_sum

    # --- 6. Normalization ---
    if normalize_mode == "recall":
        normalizer = len_1
    else:
        normalizer = len_2

    if normalizer == 0:
        return 0.0

    return max_weight_sum / normalizer


def calculate_constraint_similarity(constraint_str1: str, constraint_str2: str,
                                    strict: bool, normalize_mode: str) -> float:
    """
    Main entry function for computing the similarity of two constraint strings.
    Assumes the input expressions were already simplified by expr_process.simplify_java_bool_expr.
    Preprocessing: semantic conversion -> parse_expression
    normalize_mode: "recall" or "precision"
    """
    preConvert_expr1 = normalize_expression(constraint_str1)
    preConvert_expr2 = normalize_expression(constraint_str2)

    parsed_expr1 = parse_expression(preConvert_expr1)
    parsed_expr2 = parse_expression(preConvert_expr2)

    return _recursive_set_match(parsed_expr1, parsed_expr2, 1, strict, normalize_mode)


def count_sub_expressions(constraint_str: str) -> int:
    """Count the number of constraint components at the first level (the AND level)."""
    if not constraint_str:
        return 0

    preConvert_expr = normalize_expression(constraint_str)
    parsed_expr = parse_expression(preConvert_expr)
    set_layer1 = _split_by_layer(parsed_expr, layer=1)

    return len(set_layer1)


# ============================================================
# Test cases
# ============================================================
if __name__ == "__main__":

    # ---------- Part 1 tests: preprocessing ----------
    print("=" * 60)
    print("Part 1: expression preprocessing tests")
    print("=" * 60)

    empty_blank_tests = [
        ("myList.isEmpty()", "simple positive"),
        ("!myList.isEmpty()", "simple negation"),
        ("!(myList.isEmpty())", "negation with parentheses"),
        ("myList.isEmpty() == true", "equals true"),
        ("myList.isEmpty() == false", "equals false"),
        ("myList.isEmpty() != false", "double negation"),
        ("x > 0 && myVar.isEmpty()", "with context"),
        ("! obj.getVal().isEmpty()", "chained call"),
        ("myStr.isBlank()", "isBlank simple positive"),
        ("! (myStr.isBlank())", "isBlank negation"),
        ("myStr.isBlank() == true", "isBlank equals true"),
        ("x != null && !myStr.isBlank()", "isBlank with context"),
    ]
    print("\n--- isEmpty / isBlank tests ---")
    print(f"{'original':<45} | {'converted':<45}")
    print("-" * 90)
    for case, _desc in empty_blank_tests:
        print(f"{case:<45} | {expression_preConvert(case):<45}")

    equals_tests = [
        "a.equals(b)",
        "!a.equals(\"b\")",
        "!(\"a\".equals(b))",
        "a.equals(b) == false",
        "!a.equals(b) == false",
        "obj.equals(another.get(index.sub(1)))",
        "getUser().name.equals(\"tom\")",
    ]
    print("\n--- equals tests ---")
    print(f"{'original':<45} | {'converted':<45}")
    print("-" * 90)
    for case in equals_tests:
        print(f"{case:<45} | {expression_preConvert(case):<45}")

    # ---------- Part 2 tests: atomic similarity ----------
    print("\n" + "=" * 60)
    print("Part 2: atomic-expression similarity tests")
    print("=" * 60)

    test_cases_atomic = [
        ("x != null", "x < 0"),
        ("x != null", "x != null"),
        ("x != null", "y != null"),
        ("x != null", "null != x"),
        ("x >= 0", "0 <= x"),
        ("x > 0", "x < 0"),
        ("x > 0", "x <= 0"),
        ("x > 0", "x >= 0"),
        ("z != null", "x < 0"),
        ("gaps.LEN > 0", "gaps.LEN >= 0"),
    ]
    for p1, p2 in test_cases_atomic:
        relaxed = atomic_expression_similarity(p1, p2, strict=False)
        strict = atomic_expression_similarity(p1, p2, strict=True)
        print(f"{p1:>20} vs {p2:<20}  relaxed={relaxed:.4f}  strict={strict:.4f}")

    # ---------- Part 3 tests: constraint-set matching ----------
    print("\n" + "=" * 60)
    print("Part 3: constraint-set hierarchical matching tests")
    print("=" * 60)

    test_cases_matching = [
        ("A", "B"),
        ("A", ""),
        ("", "B"),
        ("A > 0", "B > 1"),
        ("A || B && C", "(A || B) && C"),
        ("A || B || C", "A || B"),
        ("A && (B || C)", "A && B"),
        ("A && (B || C)", "A && B && C"),
        ("b > 0", "b == Integer.MIN_VALUE || b > 0"),
        ("(a != null)", "(a != null) && (a.size() > 0)"),
        ("a != null && a.size() > 0 && b > 0", "(a != null) && (a.size() > 0)"),
        ("a != null", "(b != null) && a > 0"),
        ("a != null && !a.isEmpty()", "a != null && a.length() != 0"),
        ("a != null && !a.isEmpty()", "a != null && a.length() > 0"),
    ]

    print(f"{'Expr1':<45} | {'Expr2':<45} | {'Recall':>8} | {'Precision':>9}")
    print("-" * 115)
    for p1, p2 in test_cases_matching:
        recall_sim = calculate_constraint_similarity(
            p1, p2, strict=False, normalize_mode="recall")
        precision_sim = calculate_constraint_similarity(
            p1, p2, strict=False, normalize_mode="precision")
        f1_sim = (2 * recall_sim * precision_sim /
                  (recall_sim + precision_sim)
                  if (recall_sim + precision_sim) > 0 else 0.0)
        print(f"{p1:<45} | {p2:<45} | {recall_sim:8.4f} | {precision_sim:9.4f}")

    # ---------- Parser and dedup tests ----------
    print("\n" + "=" * 60)
    print("parser and dedup logic tests")
    print("=" * 60)

    parser_tests = [
        "A || B && C",
        "(A || B) && C",
        "A && B && A",
        "A && B && B",
        "(key != null) && (key != null && !key.isEmpty())",
        "(key != null) && (key != null || !key.isEmpty())",
        "(key != null) || (key != null || !key.isEmpty())",
        "a.equals(b) && c",
        "",
    ]
    for case in parser_tests:
        pre = expression_preConvert(case)
        parsed = parse_expression(pre)
        print(f"  '{case}' -> {parsed}")

    # ---------- First-level constraint-count tests ----------
    print("\n" + "=" * 60)
    print("first-level constraint-count tests")
    print("=" * 60)
    count_tests = [
        ("A || B || C", 3),
        ("A && (B || C)", 2),
        ("(gaps != null) && (batchSize >= 0) && (gaps.size() >= 0)", 3),
    ]
    for case, expected in count_tests:
        count = count_sub_expressions(case)
        status = "✓" if count == expected else f"✗ (expected {expected})"
        print(f"  {status} count('{case}') = {count}")
