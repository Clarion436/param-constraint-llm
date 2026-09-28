"""Utilities for parsing Java methods and building parameter-aware code slices.

The public entry point is :func:`slice_java_method`: given the source text of a
single Java method it parses the method with tree-sitter, detects formal and
multi-hop derived parameters, scans high-risk trigger statements, and renders one
combined slice that keeps only the statements relevant to those parameters
(masking the rest with ``// ...`` placeholders).  When the bundled astyle-based
Java formatter is available the slice is formatted before being returned.

Internal stages:
1. Parse the method into an AST via tree-sitter.
2. Extract formal parameters and multi-hop derived-parameter declarations.
3. Scan high-risk trigger candidates (NPE, bounds, math, assert, throw).
4. Filter redundant triggers and build the combined slice.
5. Compute statement/token retention metrics.
"""

from __future__ import annotations

import os as _os
import re
import sys as _sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, cast

# ── optional Java formatter ──────────────────────────────────────
_FORMATTER = None


def _get_formatter():
    """Lazy-load the Java formatter (``format_java_code_keep_comments``) from
    the sibling ``java_formatter`` package.  Returns ``None`` if unavailable."""
    global _FORMATTER
    if _FORMATTER is None:
        try:
            _fmt_dir = _os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                "java_formatter",
            )
            if _fmt_dir not in _sys.path:
                _sys.path.insert(0, _fmt_dir)
            from java_format import format_java_code_keep_comments as _fmt_fn
            _FORMATTER = _fmt_fn
        except Exception:
            _FORMATTER = False  # sentinel: tried, not available
    return _FORMATTER if _FORMATTER is not False else None

import tree_sitter_java as tsjava
from tree_sitter import Language, Node, Parser

# ============================================================
# Parser Setup
# ============================================================

def _load_java_language() -> Language:
    """Return the tree-sitter Language instance for Java."""
    language = getattr(tsjava, "language", None)
    if callable(language):
        # For tree-sitter-java >= 0.20.0
        result = language()
        return result if isinstance(result, Language) else Language(result)
    if language is not None:
        # For older versions
        return language if isinstance(language, Language) else Language(language)
    raise RuntimeError("Failed to load Java language from tree_sitter_java")


# --- Global Parser Setup ---
_JAVA_LANGUAGE = _load_java_language()
_PARSER = Parser()
_PARSER.language = _JAVA_LANGUAGE


def parse_java_method(source: str) -> Node:
    """Parse the given Java method source code and return the AST root node."""
    # Wrap in a dummy class for robust parsing
    wrapped_source = f"class DummyWrapper {{{source}}}"
    tree = _PARSER.parse(wrapped_source.encode("utf-8"))
    return tree.root_node


def _iter_children(node: Node) -> Iterator[Node]:
    """Iterate over named children of a node."""
    for i in range(node.named_child_count):
        child = node.named_child(i)
        if child:
            yield child


def _walk(node: Node) -> Iterator[Node]:
    """Yield node and all descendants in pre-order."""
    yield node
    for child in _iter_children(node):
        yield from _walk(child)


def find_first_method(root: Node) -> Optional[Node]:
    """Return the first method_declaration or constructor_declaration node."""
    for node in _walk(root):
        if node.type in {"method_declaration", "constructor_declaration"}:
            return node
    return None


def _node_text(node: Node) -> str:
    """Get the text of a node, decoding if necessary."""
    text = node.text
    return text.decode("utf-8") if isinstance(text, bytes) else str(text)


def extract_parameters(method_node: Node) -> List[str]:
    """Collect parameter names from a method or constructor declaration."""
    parameters_node = method_node.child_by_field_name("parameters")
    if not parameters_node:
        return []

    param_names: List[str] = []
    for param in _iter_children(parameters_node):
        # Handles both standard and varargs parameters
        if param.type == "formal_parameter":
            name_node = param.child_by_field_name("name")
            if name_node:
                param_names.append(_node_text(name_node))
        elif param.type == "spread_parameter":
            # For varargs, name is in variable_declarator
            for child in _iter_children(param):
                if child.type == "variable_declarator":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        param_names.append(_node_text(name_node))
                        break
    return param_names


# ============================================================
# Pass 1: Derived-Parameter Detection (Multi-Hop)
# ============================================================

def _identifier_in_subtree(node: Node, targets: Set[str]) -> bool:
    """Check if any identifier in the node's subtree is in the target set."""
    for descendant in _walk(node):
        if descendant.type == "identifier" and _node_text(descendant) in targets:
            return True
    return False


def _enclosing_stmt(node: Node) -> Optional[Node]:
    """Walk up from *node* to the nearest statement-level ancestor."""
    _STMT_TYPES = {
        "expression_statement", "local_variable_declaration",
        "return_statement", "if_statement", "switch_statement",
        "switch_expression", "for_statement", "enhanced_for_statement",
        "while_statement", "do_statement", "assert_statement",
        "throw_statement", "labeled_statement", "try_statement",
    }
    cur: Optional[Node] = node
    while cur is not None:
        if cur.type in _STMT_TYPES:
            return cur
        cur = cur.parent
    return node  # fallback


def collect_derived_param_declarations(
    method_node: Node,
    parameters: Iterable[str],
    max_hops: Optional[int] = None,
) -> Dict[str, List[Node]]:
    """Find local variables whose initializers trace back to parameters.

    If *max_hops* is ``None``, it defaults to 1.

    Scans both ``local_variable_declaration`` and ``assignment_expression``
    nodes inside the method body.  Uses iterative propagation up to
    ``max_hops`` hops; stops early when a full pass finds no new variables
    (fixed-point convergence).

    Hops:
    - Hop 1: variables initialized directly from formal parameters
      (e.g. ``int len = s.length()`` or ``x = param.trim()``).
    - Hop 2: variables initialized from hop-1 variables, etc.
    """
    if max_hops is None:
        max_hops = 1

    tracked_set: Set[str] = set(parameters)  # seed set: formal parameters
    derived_params: Dict[str, List[Node]] = defaultdict(list)

    if max_hops <= 0:
        return dict(derived_params)  # skip derived-parameter detection

    body_node = method_node.child_by_field_name("body")
    if not body_node:
        return dict(derived_params)

    for hop in range(max_hops):
        newly_discovered: Dict[str, List[Node]] = {}

        for node in _walk(body_node):
            # ── Case 1: local variable declaration ──────────────
            if node.type == "local_variable_declaration":
                for declarator in _iter_children(node):
                    if declarator.type != "variable_declarator":
                        continue
                    name_node = declarator.child_by_field_name("name")
                    value_node = declarator.child_by_field_name("value")
                    if not (name_node and value_node):
                        continue
                    local_name = _node_text(name_node)
                    if local_name in tracked_set:
                        continue
                    if _identifier_in_subtree(value_node, tracked_set):
                        if local_name not in newly_discovered:
                            newly_discovered[local_name] = []
                        newly_discovered[local_name].append(node)

            # ── Case 2: assignment expression ───────────────────
            # e.g.  x = param.doSomething();
            # We only track assignments whose LHS is a simple
            # identifier (not a field / array element).
            elif node.type == "assignment_expression":
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                if left is None or right is None:
                    continue
                if left.type != "identifier":
                    continue
                left_name = _node_text(left)
                if left_name in tracked_set:
                    continue
                if _identifier_in_subtree(right, tracked_set):
                    if left_name not in newly_discovered:
                        newly_discovered[left_name] = []
                    # Store the enclosing statement so the marker /
                    # renderer can match block-level siblings later.
                    stmt = _enclosing_stmt(node)
                    if stmt is not None:
                        newly_discovered[left_name].append(stmt)

        if not newly_discovered:
            break  # fixed-point: no more variables discovered

        for name, nodes in newly_discovered.items():
            derived_params[name].extend(nodes)
            tracked_set.add(name)

    return dict(derived_params)


# ============================================================
# Data classes for the slicing pipeline
# ============================================================

@dataclass
class SliceContext:
    """Holds reusable information gathered during Pass 1."""
    method_node: Node
    formal_params: List[str]
    formal_param_set: Set[str]
    derived_param_decls_by_name: Dict[str, List[Node]]


@dataclass
class TriggerRecord:
    """Represents a candidate trigger discovered during Pass 2."""
    node: Node
    related_formals_risks: Dict[str, List[str]] = field(default_factory=dict)
    related_derived_risks: Dict[str, List[str]] = field(default_factory=dict)
    is_fallback: bool = False


def build_slice_context(method_node: Node, max_hops: int = 1) -> SliceContext:
    """Pass 1: register formal parameters and derived-parameter declarations."""
    formal_params = extract_parameters(method_node)
    derived_decl_map = collect_derived_param_declarations(method_node, formal_params, max_hops=max_hops)
    return SliceContext(
        method_node=method_node,
        formal_params=formal_params,
        formal_param_set=set(formal_params),
        derived_param_decls_by_name=derived_decl_map,
    )


# ============================================================
# Helpers for trigger detection
# ============================================================

def _collect_identifier_names(node: Optional[Node]) -> Set[str]:
    if node is None:
        return set()
    names: Set[str] = set()
    for descendant in _walk(node):
        if descendant.type == "identifier":
            names.add(_node_text(descendant))
    return names


# ── Assertion / validation call detection ──────────────────────────
# Flat list of common validation method names (framework-agnostic).
# The regex uses \b to avoid matching variable names or prefixes
# (e.g. "testnotNull()" won't match, but "Objects.requireNonNull(" will).

_CHECK_METHOD_NAMES = [
    "requireNonNull",
    "checkNotNull",
    "checkArgument",
    "checkPositionIndex",
    "checkPositionIndexes",
    "checkElementIndex",
    "notNull",
    "hasText",
    "notEmpty",
    "isTrue",
    "isFalse",
    "hasLength",
    "notBlank",
    "matchesPattern",
    "orElseThrow",
    "checkState",
]

# Compile the regex once at module load:  \b(method1|method2|...)\s*\(
_METHOD_GROUP = "|".join(re.escape(m) for m in _CHECK_METHOD_NAMES)
_CHECK_METHOD_RE = re.compile(rf"\b({_METHOD_GROUP})\s*\(")


def _is_assert_call(method_name: str) -> bool:
    """Return True if ``method_name`` is an assertion-like or validation call.

    Matches two categories:
    1. Any method whose name starts with ``"assert"`` (case-insensitive).
       Examples: ``assertEquals``, ``assertThat``, ``assertThrows``.
    2. Any method listed in ``_CHECK_METHOD_NAMES``, using a word-boundary
       regex to avoid false positives on variable names / prefixes.
       Examples: ``Objects.requireNonNull``, ``Preconditions.checkArgument``,
       ``Validate.notBlank``, ``Optional.orElseThrow``.
    """
    if method_name.lower().startswith("assert"):
        return True
    # method_name is extracted cleanly from the AST, so we append "(" to
    # simulate a call and match against the compiled regex.
    return bool(_CHECK_METHOD_RE.search(method_name + "("))


# Control flow statement types that have a "controller + body" structure
CONTROL_FLOW_TYPES = {
    "if_statement",
    "while_statement",
    "for_statement",
    "enhanced_for_statement",
    "do_statement",
    "switch_statement",
    "switch_expression",       # tree-sitter-java ≥ 0.21 parses all switch as this
    "synchronized_statement",
    "try_statement",
    "try_with_resources_statement"
}


def _get_control_field(node_type: str) -> Optional[str]:
    """Return the field name for the control part of the control flow statement."""
    control_fields = {
        "if_statement": "condition",
        "while_statement": "condition",
        "for_statement": "condition",
        "enhanced_for_statement": None,
        "do_statement": "condition",
        "switch_statement": "condition",
        "switch_expression": "condition",
        "synchronized_statement": None,
        "try_statement": None,
        "try_with_resources_statement": None,
    }
    return control_fields.get(node_type)


# Expression types that we consider as candidate "core" expressions
CORE_EXPR_TYPES = {
    "binary_expression",
    "method_invocation",
    "field_access",
    "array_access",
    "subscript_expression",
    "unary_expression",
    "update_expression",
    "parenthesized_expression",
    "assignment_expression",
}


def _find_core_expression(node: Node) -> Node:
    """Return the most relevant expression node that represents the core risky usage.

    This climbs from the given node up through parents while the parent is an
    expression type we consider "core" (e.g. method call within a binary
    comparison). For `v.length() >= 1` this will return the `binary_expression`.
    If no such expression parent exists, return the original node.
    """
    current = node
    while current.parent is not None:
        if current.parent.type in CORE_EXPR_TYPES:
            current = current.parent
        else:
            break

    # If the core expression is within the control part of a control flow
    # statement, return the control flow statement
    is_in_control, control_type = _is_in_control_part(current)
    if is_in_control:
        ancestor = current
        while ancestor.parent and ancestor.type != control_type:
            ancestor = ancestor.parent
        if ancestor.type == control_type:
            return ancestor

    # Now, climb up to the enclosing statement if it's a statement type
    statement_types = {
        "expression_statement",
        "local_variable_declaration",
        "return_statement",
        "if_statement",
        "switch_statement",
        "switch_expression",
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        "throw_statement",
        "assert_statement",
        "labeled_statement",
        "try_statement",
    }
    # For certain statement types, do not climb to enclosing statement
    if current.type in {"throw_statement", "assert_statement"}:
        return current
    ancestor = current
    while ancestor.parent and ancestor.parent.type not in statement_types:
        ancestor = ancestor.parent
    if ancestor.parent and ancestor.parent.type in statement_types:
        return ancestor.parent
    return current


def _is_in_control_part(node: Node) -> tuple[bool, str]:
    """Check if the node is within the control part of a control flow statement."""
    current = node
    while current:
        if current.type in CONTROL_FLOW_TYPES:
            control_field = _get_control_field(current.type)
            if control_field:
                control_node = current.child_by_field_name(control_field)
                if control_node and _contains_node(control_node, node):
                    return True, current.type
        current = current.parent
    return False, ""


def _contains_node(container: Node, target: Node) -> bool:
    """Check if container contains target node."""
    return target.start_byte >= container.start_byte and target.end_byte <= container.end_byte


def _collect_names_with_scope(
    node: Optional[Node], usage_anchor: Node, context: SliceContext
) -> Tuple[Set[str], Set[str]]:
    """Return formal-param names and derived-param names referenced in `node`."""
    params: Set[str] = set()
    derived_decls: Set[str] = set()
    if node is None:
        return params, derived_decls

    for ident in _walk(node):
        if ident.type != "identifier":
            continue
        name = _node_text(ident)
        if name in context.formal_param_set:
            params.add(name)
            continue
        if name in context.derived_param_decls_by_name:
            derived_decls.add(name)

    return params, derived_decls


# ============================================================
# Pass 2: Trigger Detection
# ============================================================

def _detect_trigger(
    node: Node, context: SliceContext
) -> Optional[Tuple[str, Set[str], Set[str]]]:
    """Return (risk_kind, related_formals, related_deriveds) or None."""
    # Check for throw_statement first
    if node.type == "throw_statement":
        # A throw statement signals a parameter constraint through its
        # ENCLOSING control-flow condition, not through the thrown
        # expression.  e.g. in  if (x < 0) throw new RTE();
        # the constraint is "x < 0" — the throw argument is irrelevant.
        #
        # Walk up to the nearest enclosing if / while / for / switch and
        # extract parameter references from its condition.
        control_node, targets = _nearest_control_with_params(node, context)
        if control_node is not None and targets:
            formal_refs = targets & context.formal_param_set
            derived_refs = targets - context.formal_param_set
            return ("throw", formal_refs, derived_refs)
        return None  # unconditional throw — no constraint to extract

    # Check for assert_statement or expression_statement with assert
    if node.type == "assert_statement" or (
        node.type == "expression_statement"
        and _node_text(node).strip().startswith("assert")
    ):
        if node.type == "assert_statement":
            # Find the expression child (condition)
            condition = None
            for child in node.children:
                if child.type not in ["assert", ";"]:
                    condition = child
                    break
        else:
            expr = node.child_by_field_name("expression")
            condition = expr
        formal_refs, derived_refs = _collect_names_with_scope(condition, node, context)
        if formal_refs or derived_refs:
            return ("assert", formal_refs, derived_refs)
        return None

    # Check for method_invocation
    if node.type == "method_invocation":
        name_node = node.child_by_field_name("name")
        method_name = _node_text(name_node) if name_node is not None else ""
        object_node = node.child_by_field_name("object")
        arguments_node = node.child_by_field_name("arguments")
        arg_params, arg_deriveds = _collect_names_with_scope(arguments_node, node, context)
        obj_params, obj_deriveds = _collect_names_with_scope(object_node, node, context)

        if _is_assert_call(method_name):
            related_formals = arg_params | obj_params
            related_deriveds = arg_deriveds | obj_deriveds
            if related_formals or related_deriveds:
                return ("assert", related_formals, related_deriveds)
        else:
            # Treat as potential NPE when the callee object is parameter-derived
            if obj_params or obj_deriveds:
                return ("npe", obj_params, obj_deriveds)
        return None

    # Check for field_access
    if node.type == "field_access":
        obj = node.child_by_field_name("object")
        obj_params, obj_deriveds = _collect_names_with_scope(obj, node, context)
        if obj_params or obj_deriveds:
            return ("npe", obj_params, obj_deriveds)
        return None

    # Check for array_access or subscript_expression
    if node.type in {"array_access", "subscript_expression"}:
        array_node = node.child_by_field_name("array")
        index_node = node.child_by_field_name("index")
        index_params, index_deriveds = _collect_names_with_scope(index_node, node, context)
        array_params, array_deriveds = _collect_names_with_scope(array_node, node, context)
        if index_params or index_deriveds:
            related_formals = index_params | array_params
            related_deriveds = index_deriveds | array_deriveds
            return ("bounds", related_formals, related_deriveds)
        if array_params or array_deriveds:
            return ("npe", array_params, array_deriveds)
        return None

    # Check for binary_expression (divide by zero)
    if node.type == "binary_expression":
        operator_node = node.child_by_field_name("operator")
        if operator_node is not None:
            op_text = _node_text(operator_node)
            if op_text in {"/", "%"}:
                right = node.child_by_field_name("right")
                right_params, right_deriveds = _collect_names_with_scope(right, node, context)
                if right_params or right_deriveds:
                    return ("math", right_params, right_deriveds)
        return None

    # Check for enhanced_for_statement
    if node.type == "enhanced_for_statement":
        value = node.child_by_field_name("value")
        value_params, value_deriveds = _collect_names_with_scope(value, node, context)
        if value_params or value_deriveds:
            return ("npe", value_params, value_deriveds)
        return None

    return None


def _generate_fallback_triggers(context: SliceContext) -> List[TriggerRecord]:
    """Return TriggerRecords based on first usages of formal parameters."""
    body = context.method_node.child_by_field_name("body")
    if body is None:
        return []
    formal_set = context.formal_param_set
    if not formal_set:
        return []
    visited: Set[str] = set()
    triggers: List[TriggerRecord] = []
    for node in _walk(body):
        if node.type != "identifier":
            continue
        name = _node_text(node)
        if name not in formal_set or name in visited:
            continue
        core_node = _find_core_expression(node)
        core_formals, _ = _collect_names_with_scope(core_node, core_node, context)
        formal_refs = [param for param in core_formals if param in formal_set]
        if not formal_refs:
            continue
        record = TriggerRecord(node=core_node, is_fallback=True)
        for param in formal_refs:
            record.related_formals_risks[param] = cast(List[str], None)
        triggers.append(record)
        for param in formal_refs:
            visited.add(param)
        if visited == formal_set:
            break
    return triggers


def scan_trigger_candidates(
    method_node: Node, context: SliceContext
) -> List[TriggerRecord]:
    """Pass 2: traverse the AST and collect all high-risk trigger candidates."""
    body = method_node.child_by_field_name("body")
    if body is None:
        return []
    trigger_map: Dict[Node, TriggerRecord] = {}
    for node in _walk(body):
        detection = _detect_trigger(node, context)
        if detection is None:
            continue
        risk_kind, related_formals, related_deriveds = detection
        core_node = _find_core_expression(node)
        if core_node not in trigger_map:
            trigger_map[core_node] = TriggerRecord(node=core_node)
        record = trigger_map[core_node]
        for formal in related_formals:
            if formal not in record.related_formals_risks:
                record.related_formals_risks[formal] = []
            record.related_formals_risks[formal].append(risk_kind)
        for derived in related_deriveds:
            if derived not in record.related_derived_risks:
                record.related_derived_risks[derived] = []
            record.related_derived_risks[derived].append(risk_kind)
    triggers = list(trigger_map.values())
    return triggers


# ============================================================
# Trigger Deduplication
# ============================================================

def _filter_strict(triggers: List[TriggerRecord]) -> List[TriggerRecord]:
    """Original dedup: per (variable, risk) tuple; throw/assert always kept."""
    covered_risks: Set[Tuple[str, str]] = set()
    filtered: List[TriggerRecord] = []

    for trigger in triggers:
        # Extract all (variable, risk) pairs from this trigger
        current_risks: Set[Tuple[str, str]] = set()
        for var, risks in trigger.related_formals_risks.items():
            for risk in (risks or []):
                current_risks.add((var, risk))
        for var, risks in trigger.related_derived_risks.items():
            for risk in (risks or []):
                current_risks.add((var, risk))

        # Check if this is a whitelisted statement (throw or assert)
        is_whitelisted = False
        if trigger.node.type == "throw_statement":
            is_whitelisted = True
        else:
            for risks in trigger.related_formals_risks.values():
                if "assert" in (risks or []) or "throw" in (risks or []):
                    is_whitelisted = True
                    break
            if not is_whitelisted:
                for risks in trigger.related_derived_risks.values():
                    if "assert" in (risks or []) or "throw" in (risks or []):
                        is_whitelisted = True
                        break

        if trigger.is_fallback:
            filtered.append(trigger)
            continue

        if is_whitelisted:
            filtered.append(trigger)
            covered_risks.update(current_risks)
        else:
            new_risks = current_risks - covered_risks
            if new_risks:
                filtered.append(trigger)
                covered_risks.update(current_risks)

    return filtered


def _filter_relaxed(triggers: List[TriggerRecord]) -> List[TriggerRecord]:
    """Dedup only by identical statement text."""
    seen_texts: Set[str] = set()
    filtered: List[TriggerRecord] = []
    for trigger in triggers:
        text = _node_text(trigger.node).strip()
        if text not in seen_texts:
            seen_texts.add(text)
            filtered.append(trigger)
    return filtered


def _filter_none(triggers: List[TriggerRecord]) -> List[TriggerRecord]:
    """No deduplication — return all triggers as-is."""
    return list(triggers)


# Map strategy names to implementations
_DEDUP_IMPL = {
    "strict": _filter_strict,
    "relaxed": _filter_relaxed,
    "none": _filter_none,
}


def filter_redundant_triggers(
    triggers: List[TriggerRecord],
    strategy: str = "strict",
) -> List[TriggerRecord]:
    """Filter triggers according to the configured dedup strategy.

    Args:
        triggers: Candidate trigger records from Pass 2.
        strategy: ``\"strict\"`` (default) | ``\"relaxed\"`` | ``\"none\"``.

    Returns:
        Filtered (or unfiltered) trigger list per the active strategy.

    Strategy reference:
        - ``"none"``    — keep everything (default for this module).
        - ``"relaxed"`` — drop only exact-statement-text duplicates.
        - ``"strict"``  — original behaviour: dedup by (var, risk) tuple;
          throw/assert triggers are always whitelisted.
    """
    impl = _DEDUP_IMPL.get(strategy)
    if impl is None:
        raise ValueError(
            f"Unknown dedup strategy: {strategy!r}. "
            f"Valid options: {sorted(_DEDUP_IMPL.keys())}"
        )
    return impl(triggers)


# ============================================================
# Slice construction helpers
# ============================================================

@dataclass
class SlicePlan:
    """Holds derived information needed to render a slice for a trigger."""
    trigger: TriggerRecord
    target_vars: Set[str]
    derived_decl_signatures: Set[Tuple[int, int]]
    is_controller_core: bool
    is_throw: bool


class SliceBuilder:
    """Utility to accumulate slice lines with placeholder coalescing."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self._placeholder_active = False

    def emit(self, text: str) -> None:
        if text.strip() == "// ...":
            if self._placeholder_active:
                return
            self._placeholder_active = True
        else:
            self._placeholder_active = False
        self.lines.append(text)

    def emit_placeholder(self) -> None:
        self.emit("// ...")

    def open_block(self, header: str) -> None:
        self.emit(f"{header} {{")
        self._placeholder_active = False

    def close_block(self) -> None:
        self.lines.append("}")
        self._placeholder_active = False

    def render(self) -> str:
        return "\n".join(self.lines)


def _node_signature(node: Node) -> Tuple[int, int]:
    return (node.start_byte, node.end_byte)


def _gather_derived_decl_signatures(
    context: SliceContext, target_vars: Set[str]
) -> Set[Tuple[int, int]]:
    signatures: Set[Tuple[int, int]] = set()
    for name in target_vars:
        for decl in context.derived_param_decls_by_name.get(name, []):
            signatures.add(_node_signature(decl))
    return signatures


def _statement_defines_derived(
    node: Node, derived_decl_signatures: Set[Tuple[int, int]]
) -> bool:
    return _node_signature(node) in derived_decl_signatures


def _statement_assignment_targets(node: Node) -> Set[str]:
    targets: Set[str] = set()
    for descendant in _walk(node):
        if descendant.type == "assignment_expression":
            left = descendant.child_by_field_name("left")
            if left is not None:
                targets.update(_collect_identifier_names(left))
        elif descendant.type == "update_expression":
            targets.update(_collect_identifier_names(descendant))
    return targets


def _statement_updates_targets(node: Node, target_vars: Set[str]) -> bool:
    if not target_vars:
        return False
    return bool(_statement_assignment_targets(node) & target_vars)


def _find_child_containing(parent: Node, target: Node) -> Optional[Node]:
    for child in _iter_children(parent):
        if _contains_node(child, target):
            return child
    return None


def _find_first_child_of_type(parent: Node, child_type: str) -> Optional[Node]:
    """Return the first named child of ``parent`` with the given type."""
    for child in _iter_children(parent):
        if child.type == child_type:
            return child
    return None


def _switch_group_is_empty(group: Node) -> bool:
    """Return True if *group* (a ``switch_block_statement_group``) contains
    no statement children — only ``switch_label`` nodes (fall-through)."""
    for child in _iter_children(group):
        if child.type != "switch_label":
            return False
    return True


def _nearest_control_with_params(
    node: Node, context: SliceContext
) -> Tuple[Optional[Node], Set[str]]:
    current = node.parent
    while current is not None and current != context.method_node:
        if current.type in CONTROL_FLOW_TYPES:
            control_field = _get_control_field(current.type)
            control_expr = (
                current.child_by_field_name(control_field) if control_field else None
            )
            formals, deriveds = _collect_names_with_scope(control_expr, current, context)
            targets = set(formals) | set(deriveds)
            return current, targets
        current = current.parent
    return None, set()


def _prepare_slice_plan(
    context: SliceContext, trigger: TriggerRecord
) -> Optional[SlicePlan]:
    node = trigger.node
    if node.type == "throw_statement":
        control_node, targets = _nearest_control_with_params(node, context)
        if control_node is None or not targets:
            return None
        target_vars = set(targets)
        derived_sigs = _gather_derived_decl_signatures(context, target_vars)
        return SlicePlan(
            trigger=trigger,
            target_vars=target_vars,
            derived_decl_signatures=derived_sigs,
            is_controller_core=False,
            is_throw=True,
        )

    target_vars = set(trigger.related_formals_risks.keys()) | set(
        trigger.related_derived_risks.keys()
    )
    if not target_vars:
        return None
    in_control_part, _ = _is_in_control_part(node)
    is_controller_core = in_control_part or node.type in CONTROL_FLOW_TYPES
    derived_sigs = _gather_derived_decl_signatures(context, target_vars)
    return SlicePlan(
        trigger=trigger,
        target_vars=target_vars,
        derived_decl_signatures=derived_sigs,
        is_controller_core=is_controller_core,
        is_throw=False,
    )


def _control_header_text(node: Node) -> str:
    """Return the header part of a control-flow node, excluding the body.

    For nodes whose body is wrapped in ``{ }``, this is everything before
    the opening brace.  For single-statement bodies (no braces), the
    header is the text up to (but not including) the body node.
    """
    text = _node_text(node).strip()
    brace_index = text.find("{")
    if brace_index != -1:
        return text[:brace_index].strip()
    # No braces — body is a single statement; extract header before it.
    body = node.child_by_field_name("body")
    if body is not None:
        body_start = body.start_byte - node.start_byte
        return text[:body_start].strip()
    return text or node.type.replace("_", " ")


# ============================================================
# Pass 3: Combined slice (single output for all triggers)
# ============================================================

def _extract_method_header(method_node: Node) -> str:
    """Return the method signature including the opening brace."""
    body = method_node.child_by_field_name("body")
    if body is None:
        text = _node_text(method_node)
        return text.decode("utf-8") if isinstance(text, bytes) else str(text)
    full_text = _node_text(method_node)
    if isinstance(full_text, bytes):
        full_text = full_text.decode("utf-8")
    offset = body.start_byte - method_node.start_byte
    return full_text[:offset].rstrip() + " {"


def _mark_nodes_for_trigger(
    current: Node, target_node: Node, plan: SlicePlan,
    keep_set: Set[Tuple[int, int]],
) -> None:
    """Walk from *current* down to *target_node*, adding kept nodes to *keep_set*.

    Accumulates node signatures instead of emitting text.  Also marks
    derived-parameter declarations and target-variable update statements
    that appear in blocks before the path child.
    """
    keep_set.add(_node_signature(current))
    if current == target_node:
        return

    if current.type == "block":
        child = _find_child_containing(current, target_node)
        if child is None:
            return
        for sibling in _iter_children(current):
            if sibling == child:
                break
            if sibling.type not in {
                "local_variable_declaration", "expression_statement",
                "return_statement", "if_statement", "switch_statement",
                "switch_expression", "for_statement", "enhanced_for_statement",
                "while_statement", "do_statement", "assert_statement",
                "throw_statement", "try_statement",
            }:
                continue
            if _statement_defines_derived(sibling, plan.derived_decl_signatures):
                keep_set.add(_node_signature(sibling))
            elif _statement_updates_targets(sibling, plan.target_vars):
                keep_set.add(_node_signature(sibling))
        _mark_nodes_for_trigger(child, target_node, plan, keep_set)

    elif current.type in CONTROL_FLOW_TYPES:
        child = _find_child_containing(current, target_node)
        if child is None:
            return
        _mark_cf_children(current, child, target_node, plan, keep_set)

    else:
        child = _find_child_containing(current, target_node)
        if child is None:
            return
        _mark_nodes_for_trigger(child, target_node, plan, keep_set)


def _mark_cf_children(
    cf_node: Node, child_on_path: Node, target_node: Node,
    plan: SlicePlan, keep_set: Set[Tuple[int, int]],
) -> None:
    """Mark children of a control-flow node on the path to *target_node*."""
    if cf_node.type == "if_statement":
        consequence = cf_node.child_by_field_name("consequence")
        alternative = cf_node.child_by_field_name("alternative")
        if consequence is not None and _contains_node(consequence, child_on_path):
            _mark_nodes_for_trigger(consequence, target_node, plan, keep_set)
        if alternative is not None and _contains_node(alternative, child_on_path):
            _mark_nodes_for_trigger(alternative, target_node, plan, keep_set)
    elif cf_node.type in {"while_statement", "for_statement", "enhanced_for_statement"}:
        body = cf_node.child_by_field_name("body")
        if body is not None and _contains_node(body, child_on_path):
            _mark_nodes_for_trigger(body, target_node, plan, keep_set)
    elif cf_node.type == "do_statement":
        body = cf_node.child_by_field_name("body")
        if body is not None and _contains_node(body, child_on_path):
            _mark_nodes_for_trigger(body, target_node, plan, keep_set)
    elif cf_node.type in {"switch_statement", "switch_expression"}:
        for group in _iter_children(child_on_path):
            if group.type == "switch_block_statement_group" and _contains_node(group, target_node):
                _mark_nodes_for_trigger(group, target_node, plan, keep_set)
                break
    elif cf_node.type == "try_statement":
        body = cf_node.child_by_field_name("body")
        if body is not None and _contains_node(body, child_on_path):
            _mark_nodes_for_trigger(body, target_node, plan, keep_set)
        for child in _iter_children(cf_node):
            if child.type == "catch_clause" and _contains_node(child, child_on_path):
                catch_body = child.child_by_field_name("body")
                if catch_body is not None:
                    _mark_nodes_for_trigger(catch_body, target_node, plan, keep_set)
            elif child.type == "finally_clause" and _contains_node(child, child_on_path):
                fin_body = child.named_child(0) if child.named_child_count > 0 else None
                if fin_body is not None:
                    _mark_nodes_for_trigger(fin_body, target_node, plan, keep_set)
    else:
        _mark_nodes_for_trigger(child_on_path, target_node, plan, keep_set)


# ── Masked rendering (Phase B) ───────────────────────────────────

def _render_masked_block(
    block_node: Node, keep_set: Set[Tuple[int, int]], builder: SliceBuilder,
) -> None:
    """Render a block: kept children in full, others as ``// ...``."""
    for child in _iter_children(block_node):
        sig = _node_signature(child)
        if sig in keep_set:
            _render_kept_child(child, keep_set, builder)
        else:
            builder.emit_placeholder()


def _render_kept_child(
    node: Node, keep_set: Set[Tuple[int, int]], builder: SliceBuilder,
) -> None:
    """Render a single kept child, recursing into control-flow bodies."""
    if node.type in CONTROL_FLOW_TYPES:
        _render_masked_cf(node, keep_set, builder)
    elif node.type == "block":
        _render_masked_block(node, keep_set, builder)
    elif node.type == "switch_block_statement_group":
        label_node = _find_first_child_of_type(node, "switch_label")
        if label_node:
            builder.emit(_node_text(label_node).strip() + ":")
        _render_masked_block(node, keep_set, builder)
    else:
        builder.emit(_node_text(node).strip())


def _render_masked_cf(
    node: Node, keep_set: Set[Tuple[int, int]], builder: SliceBuilder,
) -> None:
    """Render a kept control-flow node, masking irrelevant bodies."""
    if node.type == "if_statement":
        condition = node.child_by_field_name("condition")
        cond_text = (
            _node_text(condition).strip() if condition is not None else "<condition>"
        )
        builder.open_block(f"if {cond_text}")
        consequence = node.child_by_field_name("consequence")
        if consequence is not None:
            _render_masked_block_or_stmt(consequence, keep_set, builder)
        builder.close_block()
        alternative = node.child_by_field_name("alternative")
        _render_masked_alternative(alternative, keep_set, builder)

    elif node.type in {"while_statement", "for_statement", "enhanced_for_statement"}:
        header = _control_header_text(node)
        builder.open_block(header)
        body = node.child_by_field_name("body")
        if body is not None:
            _render_masked_block_or_stmt(body, keep_set, builder)
        builder.close_block()

    elif node.type == "do_statement":
        condition = node.child_by_field_name("condition")
        cond_text = (
            _node_text(condition).strip() if condition is not None else "<condition>"
        )
        builder.open_block("do")
        body = node.child_by_field_name("body")
        if body is not None:
            _render_masked_block_or_stmt(body, keep_set, builder)
        builder.close_block()
        builder.emit(f"while {cond_text};")

    elif node.type in {"switch_statement", "switch_expression"}:
        condition = node.child_by_field_name("condition")
        cond_text = (
            _node_text(condition).strip() if condition is not None else "<expr>"
        )
        builder.open_block(f"switch {cond_text}")
        for child in _iter_children(node):
            if child.type == "switch_block":
                for group in _iter_children(child):
                    if group.type != "switch_block_statement_group":
                        continue
                    sig = _node_signature(group)
                    if sig in keep_set:
                        _render_kept_child(group, keep_set, builder)
                    elif not _switch_group_is_empty(group):
                        label_node = _find_first_child_of_type(group, "switch_label")
                        if label_node:
                            builder.emit(f"{_node_text(label_node).strip()}: // ...")
                        else:
                            builder.emit_placeholder()
                break
        builder.close_block()

    elif node.type == "try_statement":
        builder.open_block("try")
        body = node.child_by_field_name("body")
        if body is not None:
            _render_masked_block_or_stmt(body, keep_set, builder)
        builder.close_block()
        # catch / finally clauses
        for child in _iter_children(node):
            if child.type == "catch_clause":
                catch_body = child.child_by_field_name("body")
                param_node = _find_first_child_of_type(child, "catch_formal_parameter")
                header = (
                    f"catch ({_node_text(param_node).strip()})"
                    if param_node else "catch"
                )
                builder.open_block(header)
                if catch_body is not None:
                    _render_masked_block_or_stmt(catch_body, keep_set, builder)
                builder.close_block()
            elif child.type == "finally_clause":
                fin_body = (
                    child.named_child(0)
                    if child.named_child_count > 0 else None
                )
                builder.open_block("finally")
                if fin_body is not None:
                    _render_masked_block_or_stmt(fin_body, keep_set, builder)
                builder.close_block()

    else:
        builder.open_block(node.type.replace("_", " "))
        builder.emit_placeholder()
        builder.close_block()


def _render_masked_alternative(
    alternative: Optional[Node],
    keep_set: Set[Tuple[int, int]],
    builder: SliceBuilder,
) -> None:
    """Render the else / else-if chain of an if-statement."""
    if alternative is None:
        return
    if alternative.type == "if_statement":
        # "else if" — render without wrapping in "else { }"
        condition = alternative.child_by_field_name("condition")
        cond_text = (
            _node_text(condition).strip() if condition is not None else "<condition>"
        )
        builder.lines.append(f"else if {cond_text} {{")
        builder._placeholder_active = False
        consequence = alternative.child_by_field_name("consequence")
        if consequence is not None:
            _render_masked_block_or_stmt(consequence, keep_set, builder)
        builder.lines.append("}")
        builder._placeholder_active = False
        # recurse for further else-if / else
        _render_masked_alternative(
            alternative.child_by_field_name("alternative"), keep_set, builder
        )
    else:
        builder.open_block("else")
        _render_masked_block_or_stmt(alternative, keep_set, builder)
        builder.close_block()


def _render_masked_block_or_stmt(
    node: Node, keep_set: Set[Tuple[int, int]], builder: SliceBuilder,
) -> None:
    """Render a node that is either a block or a single statement body."""
    if node.type == "block":
        _render_masked_block(node, keep_set, builder)
    else:
        sig = _node_signature(node)
        if sig in keep_set:
            _render_kept_child(node, keep_set, builder)
        else:
            builder.emit_placeholder()


def build_combined_slice(
    context: SliceContext, triggers: List[TriggerRecord],
) -> Tuple[Optional[str], Set[Tuple[int, int]]]:
    """Build a single combined slice from all *triggers*.

    Phase A — Mark: for each trigger compute a slice plan, then walk
    from the method body down to the trigger node, recording kept nodes.
    Phase B — Render: walk the method body once; kept nodes render in
    full, others become ``// ...`` placeholders.  The function signature
    and enclosing braces are always included.

    Returns:
        ``(slice_text, keep_set)`` — *slice_text* may be ``None`` if no
        nodes were kept; *keep_set* contains the signatures of all nodes
        marked during Phase A (empty when *slice_text* is ``None``).
    """
    body = context.method_node.child_by_field_name("body")
    if body is None:
        return None, set()

    keep_set: Set[Tuple[int, int]] = set()

    # Phase A: union of all trigger paths
    for trigger in triggers:
        plan = _prepare_slice_plan(context, trigger)
        if plan is None:
            continue
        _mark_nodes_for_trigger(body, plan.trigger.node, plan, keep_set)

    if not keep_set:
        return None, keep_set

    # Phase B: single masked rendering pass
    builder = SliceBuilder()
    builder.emit(_extract_method_header(context.method_node))
    _render_masked_block(body, keep_set, builder)
    builder.emit("}")
    return builder.render() or None, keep_set


# ── Retention metrics ─────────────────────────────────────────────

# Statement types that count as one "statement node" in a block.
_STMT_NODE_TYPES: Set[str] = {
    "local_variable_declaration", "expression_statement",
    "return_statement", "assert_statement", "throw_statement",
    "labeled_statement",
}


def _is_stmt_node(node: Node) -> bool:
    """Return True if *node* is a statement-level AST node."""
    return node.type in _STMT_NODE_TYPES or node.type in CONTROL_FLOW_TYPES


def _count_stmts_in_block(
    block_node: Node, keep_set: Set[Tuple[int, int]],
) -> Tuple[int, int]:
    """Count statement nodes inside *block_node*, recursing into bodies.

    Returns ``(kept, total)``.
    """
    kept, total = 0, 0
    for child in _iter_children(block_node):
        if child.type in CONTROL_FLOW_TYPES:
            total += 1
            if _node_signature(child) in keep_set:
                kept += 1
            sk, st = _count_cf_body_stmts(child, keep_set)
            kept += sk
            total += st
        elif child.type == "switch_block":
            for group in _iter_children(child):
                if group.type == "switch_block_statement_group":
                    total += 1
                    if _node_signature(group) in keep_set:
                        kept += 1
                    sk, st = _count_stmts_in_block(group, keep_set)
                    kept += sk
                    total += st
        elif _is_stmt_node(child):
            total += 1
            if _node_signature(child) in keep_set:
                kept += 1
    return kept, total


def _count_cf_body_stmts(
    cf_node: Node, keep_set: Set[Tuple[int, int]],
) -> Tuple[int, int]:
    """Count statements inside the body/bodies of a control-flow node."""
    kept, total = 0, 0
    if cf_node.type == "if_statement":
        for field in ("consequence", "alternative"):
            body = cf_node.child_by_field_name(field)
            if body is not None:
                if body.type == "block":
                    sk, st = _count_stmts_in_block(body, keep_set)
                else:
                    sk = 1 if _node_signature(body) in keep_set else 0
                    st = 1
                kept += sk
                total += st
    elif cf_node.type in {
        "while_statement", "for_statement", "enhanced_for_statement",
        "do_statement",
    }:
        body = cf_node.child_by_field_name("body")
        if body is not None:
            if body.type == "block":
                sk, st = _count_stmts_in_block(body, keep_set)
            else:
                sk = 1 if _node_signature(body) in keep_set else 0
                st = 1
            kept += sk
            total += st
    elif cf_node.type == "try_statement":
        body = cf_node.child_by_field_name("body")
        if body is not None:
            if body.type == "block":
                sk, st = _count_stmts_in_block(body, keep_set)
            else:
                sk = 1 if _node_signature(body) in keep_set else 0
                st = 1
            kept += sk; total += st
        for child in _iter_children(cf_node):
            if child.type == "catch_clause":
                catch_body = child.child_by_field_name("body")
                if catch_body is not None:
                    sk, st = _count_stmts_in_block(catch_body, keep_set)
                    kept += sk; total += st
            elif child.type == "finally_clause":
                fin_body = child.named_child(0) if child.named_child_count > 0 else None
                if fin_body is not None:
                    sk, st = _count_stmts_in_block(fin_body, keep_set)
                    kept += sk; total += st
    # switch_expression bodies are handled in _count_stmts_in_block
    # via the switch_block branch.
    return kept, total


def _compute_stmt_retention(
    body_node: Node, keep_set: Set[Tuple[int, int]],
) -> Tuple[int, int, float]:
    """Return ``(kept, total, ratio)`` for statement-level retention."""
    kept, total = _count_stmts_in_block(body_node, keep_set)
    ratio = kept / total if total > 0 else 0.0
    return kept, total, ratio


# ── Token counting (OpenAI o200k_base / GPT-4o) ───────────────────

try:
    import tiktoken as _tiktoken
except ImportError:
    _tiktoken = None  # type: ignore[assignment]


_O200K_ENCODER = None


def _get_o200k_encoder():
    """Return (or initialise) a cached o200k_base tokenizer instance."""
    global _O200K_ENCODER
    if _O200K_ENCODER is None:
        if _tiktoken is None:
            raise RuntimeError(
                "tiktoken is not installed; cannot compute token count.  "
                "Install it with:  pip install tiktoken"
            )
        _O200K_ENCODER = _tiktoken.get_encoding("o200k_base")
    return _O200K_ENCODER


def _count_tokens_o200k(text: str) -> int:
    """Return the number of GPT-4o tokens in *text*."""
    enc = _get_o200k_encoder()
    return len(enc.encode(text))


# ============================================================
# Public API
# ============================================================

def slice_java_method(
    java_code: str,
    max_hops: int = 1,
    dedup_strategy: str = "strict",
) -> dict:
    """Slice a Java method string and return a dict with slices and metadata.

    Args:
        java_code: Source code of a single Java method.
        max_hops: Maximum derived-parameter hop distance.
            - 0: skip derived-parameter detection.
            - 1 (default): only derived-parameters directly initialised from formal params.
            - N: propagate up to N hops.
        dedup_strategy: Controls how redundant triggers are filtered before
            slice construction.  Three strategies are available:

            - ``\"strict\"`` (default): deduplicate by ``(parameter, risk_kind)``
              tuple.  The first trigger covering each unique pair is kept;
              throw and assert triggers are always whitelisted.  Most
              aggressive reduction — keeps only the first occurrence of
              each distinct risk.
            - ``\"relaxed\"``: deduplicate only triggers whose statement text
              is *identical*.  Different statements that cover the same
              ``(parameter, risk_kind)`` pair are all kept, providing
              richer signal to downstream consumers (e.g. an LLM) while
              still removing verbatim duplicates.
            - ``\"none\"``: no deduplication — every candidate trigger
              contributes to the slice.  Maximises signal at the cost of
              potential redundancy.

    Returns:
        dict with keys:
            - ``"is_fallback"`` (bool): whether fallback logic was used.
            - ``"is_effective"`` (bool): ``True`` when at least one
              statement was masked (stmt_retention < 1.0), i.e. the
              slice actually improves signal-to-noise ratio.
            - ``"slice"`` (str | None): rendered slice code, or ``None``.
            - ``"stmt_retention"`` (float | None): ``kept / total``,
              rounded to 4 decimal places (0.0 – 1.0), or ``None``.
            - ``"token_retention"`` (float | None): slice / source
              token count (GPT-4o o200k_base), rounded to 4 places, or ``None``.
    """
    try:
        root_node = parse_java_method(java_code)
        method_node = find_first_method(root_node)
        if method_node is None:
            return {
                "is_effective": False, "is_fallback": False,
                "slice": None, "stmt_retention": None, "token_retention": None,
            }
        body = method_node.child_by_field_name("body")
        context = build_slice_context(method_node, max_hops=max_hops)

        # PASS 2: Scan high-risk triggers
        triggers = scan_trigger_candidates(method_node, context)
        # PASS 3: Filter and build combined slice
        filtered_triggers = filter_redundant_triggers(triggers, strategy=dedup_strategy)
        combined, keep_set = build_combined_slice(context, filtered_triggers)
        if combined is None:
            # PASS 4: Fallback if no slice produced
            fallback_triggers = _generate_fallback_triggers(context)
            filtered_fallback = filter_redundant_triggers(fallback_triggers, strategy=dedup_strategy)
            combined, keep_set = build_combined_slice(context, filtered_fallback)
            is_fallback = True
        else:
            is_fallback = False

        # ── Formatting (optional) ────────────────────────────────
        fmt = _get_formatter()
        if combined is not None and fmt is not None:
            try:
                combined = fmt(combined)
            except Exception:
                pass  # keep unformatted slice on formatting error

        # ── Retention metrics ──────────────────────────────────
        stmt_retention: Optional[float] = None
        token_retention: Optional[float] = None

        if combined is not None and body is not None:
            # Statement retention (based on keep_set — formatting
            # does not change which statements were kept).
            _, _, stmt_retention = _compute_stmt_retention(body, keep_set)

            # Token retention (o200k_base) — numerator uses the
            # *formatted* slice, denominator uses the raw input.
            src_tokens = _count_tokens_o200k(java_code.strip())
            slice_tokens = _count_tokens_o200k(combined)
            token_retention = (
                slice_tokens / src_tokens if src_tokens > 0 else 0.0
            )

        return {
            "is_effective": (
                combined is not None
                and stmt_retention is not None
                and stmt_retention < 1.0
            ),
            "is_fallback": is_fallback,
            "slice": combined,
            "stmt_retention": round(stmt_retention, 4) if stmt_retention is not None else None,
            "token_retention": round(token_retention, 4) if token_retention is not None else None,
        }
    except Exception:
        return {
            "is_effective": False, "is_fallback": False,
            "slice": None, "stmt_retention": None, "token_retention": None,
        }


if __name__ == "__main__":
    _demo = """
    public void demo(int[] arr, String s, int x) {
        if (x < 0) {
            throw new IllegalArgumentException("x must be non-negative");
        }
        int len = s.length();
        int elem = arr[x];
        int ratio = 100 / len;
        System.out.println(ratio);
    }
    """
    result = slice_java_method(_demo, max_hops=1, dedup_strategy="strict")
    print(f"is_fallback={result['is_fallback']}  "
          f"is_effective={result['is_effective']}  "
          f"stmt_retention={result['stmt_retention']}  "
          f"token_retention={result['token_retention']}")
    print()
    print(result["slice"] or "(no slice)")


