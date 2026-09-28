from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import sympy as sp
from sympy.core.relational import Eq, Ne, StrictGreaterThan, StrictLessThan, LessThan, GreaterThan
from sympy.logic.boolalg import And, Or, Not, BooleanTrue, BooleanFalse


@dataclass
class _SymbolTable:
    bool_map: Dict[str, sp.Symbol]
    operand_map: Dict[str, sp.Symbol]
    reverse_map: Dict[sp.Symbol, str]
    bool_count: int = 0
    operand_count: int = 0

    def get_bool_symbol(self, name: str) -> sp.Symbol:
        if name in self.bool_map:
            return self.bool_map[name]
        sym = sp.Symbol(f"__b{self.bool_count}")
        self.bool_count += 1
        self.bool_map[name] = sym
        self.reverse_map[sym] = name
        return sym

    def get_operand_symbol(self, name: str) -> sp.Symbol:
        if name in self.operand_map:
            return self.operand_map[name]
        sym = sp.Symbol(f"__v{self.operand_count}")
        self.operand_count += 1
        self.operand_map[name] = sym
        self.reverse_map[sym] = name
        return sym

    def resolve(self, sym: sp.Symbol) -> Optional[str]:
        return self.reverse_map.get(sym)


class _Parser:
    def __init__(self, text: str, symbols: _SymbolTable) -> None:
        self.text = text
        self.symbols = symbols
        self.i = 0
        self.n = len(text)

    def parse(self) -> sp.Expr:
        expr = self._parse_or()
        self._skip_ws()
        if self.i != self.n:
            raise ValueError(f"Unexpected trailing input at position {self.i}: {self.text[self.i:]}")
        return expr

    def _skip_ws(self) -> None:
        while self.i < self.n and self.text[self.i].isspace():
            self.i += 1

    def _peek(self, s: str) -> bool:
        return self.text.startswith(s, self.i)

    def _consume(self, s: str) -> bool:
        if self._peek(s):
            self.i += len(s)
            return True
        return False

    def _parse_or(self) -> sp.Expr:
        left = self._parse_and()
        while True:
            self._skip_ws()
            if self._consume("||"):
                right = self._parse_and()
                left = Or(left, right)
            else:
                break
        return left

    def _parse_and(self) -> sp.Expr:
        left = self._parse_not()
        while True:
            self._skip_ws()
            if self._consume("&&"):
                right = self._parse_not()
                left = And(left, right)
            else:
                break
        return left

    def _parse_not(self) -> sp.Expr:
        self._skip_ws()
        if self._consume("!"):
            return Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> sp.Expr:
        self._skip_ws()
        atom = self._read_atom_text()
        if not atom:
            raise ValueError(f"Expected expression at position {self.i}")

        # Treat the atom as a boolean group (and parse it recursively) only when it
        # is fully wrapped in one pair of parentheses. This avoids mistaking
        # arithmetic sub-expressions such as "(a % 4) + 1" for boolean groups.
        if self._is_fully_parenthesized(atom):
            inner = atom[1:-1].strip()
            if not inner:
                raise ValueError("Empty parenthesized expression")
            try:
                nested = _Parser(inner, self.symbols)
                return nested.parse()
            except ValueError:
                # Not a parsable boolean group: fall back to plain atom handling.
                pass

        return self._parse_comparison_or_symbol(atom)

    @staticmethod
    def _is_fully_parenthesized(text: str) -> bool:
        if len(text) < 2 or text[0] != "(" or text[-1] != ")":
            return False
        depth = 0
        in_single = False
        in_double = False
        escape = False

        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if in_single:
                if ch == "'":
                    in_single = False
                continue
            if in_double:
                if ch == '"':
                    in_double = False
                continue
            if ch == "'":
                in_single = True
                continue
            if ch == '"':
                in_double = True
                continue

            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(text) - 1:
                    return False
                if depth < 0:
                    return False

        return depth == 0

    def _read_atom_text(self) -> str:
        start = self.i
        depth_paren = 0
        depth_brack = 0
        depth_brace = 0
        in_single = False
        in_double = False
        escape = False

        while self.i < self.n:
            ch = self.text[self.i]

            if escape:
                escape = False
                self.i += 1
                continue

            if ch == "\\":
                escape = True
                self.i += 1
                continue

            if in_single:
                if ch == "'":
                    in_single = False
                self.i += 1
                continue

            if in_double:
                if ch == '"':
                    in_double = False
                self.i += 1
                continue

            if ch == "'":
                in_single = True
                self.i += 1
                continue

            if ch == '"':
                in_double = True
                self.i += 1
                continue

            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                if depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
                    break
                depth_paren -= 1
            elif ch == "[":
                depth_brack += 1
            elif ch == "]":
                depth_brack -= 1
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace -= 1

            if depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
                if self.text.startswith("&&", self.i) or self.text.startswith("||", self.i):
                    break

            self.i += 1

        return self.text[start:self.i].strip()

    def _parse_comparison_or_symbol(self, atom: str) -> sp.Expr:
        comparison = self._split_comparison(atom)
        if comparison is None:
            if atom == "true":
                return BooleanTrue()
            if atom == "false":
                return BooleanFalse()
            return self.symbols.get_bool_symbol(atom)

        left, op, right = comparison
        lhs = self.symbols.get_operand_symbol(left)
        rhs = self.symbols.get_operand_symbol(right)
        if op == "==":
            return Eq(lhs, rhs)
        if op == "!=":
            return Ne(lhs, rhs)
        if op == "<":
            return StrictLessThan(lhs, rhs)
        if op == "<=":
            return LessThan(lhs, rhs)
        if op == ">":
            return StrictGreaterThan(lhs, rhs)
        if op == ">=":
            return GreaterThan(lhs, rhs)
        raise ValueError(f"Unsupported comparison operator: {op}")

    def _split_comparison(self, atom: str) -> Optional[tuple[str, str, str]]:
        depth_paren = 0
        depth_brack = 0
        depth_brace = 0
        in_single = False
        in_double = False
        escape = False

        i = 0
        while i < len(atom):
            ch = atom[i]

            if escape:
                escape = False
                i += 1
                continue

            if ch == "\\":
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

            if ch == "(":
                depth_paren += 1
                i += 1
                continue
            if ch == ")":
                depth_paren -= 1
                i += 1
                continue
            if ch == "[":
                depth_brack += 1
                i += 1
                continue
            if ch == "]":
                depth_brack -= 1
                i += 1
                continue
            if ch == "{":
                depth_brace += 1
                i += 1
                continue
            if ch == "}":
                depth_brace -= 1
                i += 1
                continue

            if depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
                two = atom[i : i + 2]
                if two in ("==", "!=", "<=", ">="):
                    left = atom[:i].strip()
                    right = atom[i + 2 :].strip()
                    return left, two, right
                # Skip shift operators << >> >>> so they are not mis-matched by < >
                if ch in ("<", ">"):
                    if atom.startswith(">>>", i):
                        i += 3
                        continue
                    if atom.startswith("<<", i) or atom.startswith(">>", i):
                        i += 2
                        continue
                    left = atom[:i].strip()
                    right = atom[i + 1 :].strip()
                    return left, ch, right

            i += 1

        return None


def _render_java(expr: sp.Expr, symbols: _SymbolTable, parent_prec: int = 0) -> str:
    if isinstance(expr, BooleanTrue):
        return "true"
    if isinstance(expr, BooleanFalse):
        return "false"
    if isinstance(expr, sp.Symbol):
        resolved = symbols.resolve(expr)
        return resolved if resolved is not None else expr.name

    if isinstance(expr, Not):
        inner = expr.args[0]
        rendered = _render_java(inner, symbols, 3)
        if _precedence(inner) < 3:
            rendered = f"({rendered})"
        return f"!{rendered}"

    if isinstance(expr, And):
        parts = [_render_java(arg, symbols, 2) for arg in expr.args]
        rendered = " && ".join(parts)
        if parent_prec > 2:
            return f"({rendered})"
        return rendered

    if isinstance(expr, Or):
        parts = [_render_java(arg, symbols, 1) for arg in expr.args]
        rendered = " || ".join(parts)
        if parent_prec > 1:
            return f"({rendered})"
        return rendered

    if isinstance(expr, (Eq, Ne, StrictGreaterThan, StrictLessThan, LessThan, GreaterThan)):
        op = _rel_op(expr)
        left = _render_operand(expr.lhs, symbols)
        right = _render_operand(expr.rhs, symbols)
        return f"{left} {op} {right}"

    rendered = _render_operand(expr, symbols)
    if rendered is None:
        return str(expr)
    return rendered


def _render_operand(node: sp.Expr, symbols: _SymbolTable) -> str:
    if isinstance(node, sp.Symbol):
        resolved = symbols.resolve(node)
        return resolved if resolved is not None else node.name
    if isinstance(node, BooleanTrue):
        return "true"
    if isinstance(node, BooleanFalse):
        return "false"
    return str(node)


def _precedence(expr: sp.Expr) -> int:
    if isinstance(expr, Or):
        return 1
    if isinstance(expr, And):
        return 2
    if isinstance(expr, Not):
        return 3
    return 4


def _rel_op(expr: sp.Expr) -> str:
    if isinstance(expr, Eq):
        return "=="
    if isinstance(expr, Ne):
        return "!="
    if isinstance(expr, StrictLessThan):
        return "<"
    if isinstance(expr, LessThan):
        return "<="
    if isinstance(expr, StrictGreaterThan):
        return ">"
    if isinstance(expr, GreaterThan):
        return ">="
    raise ValueError(f"Unknown relational operator: {expr}")


def negate_java_bool_expr(java_expr: str) -> str:
    """
    Given a Java boolean expression, return its negated and simplified form as a
    Java expression string.
    Supports: &&, ||, !, (), and the comparison operators == != < <= > >=.
    Other Java atomic expressions are treated as boolean atoms and returned as-is.
    """
    symbols = _SymbolTable(bool_map={}, operand_map={}, reverse_map={})
    parser = _Parser(java_expr, symbols)
    parsed = parser.parse()
    negated = Not(parsed)
    try:
        simplified = sp.simplify(negated)
    except Exception:
        simplified = negated
    return _render_java(simplified, symbols)


def simplify_java_bool_expr(java_expr: str) -> str:
    """
    Given a Java boolean expression, return its simplified Java boolean expression
    string.
    Supports: &&, ||, !, (), and the comparison operators == != < <= > >=.
    Other Java atomic expressions are treated as boolean atoms and returned as-is.
    """
    symbols = _SymbolTable(bool_map={}, operand_map={}, reverse_map={})
    parser = _Parser(java_expr, symbols)
    parsed = parser.parse()
    try:
        simplified = sp.simplify(parsed)
    except Exception:
        simplified = parsed
    return _render_java(simplified, symbols)



def _parse_with_symbols(java_expr: str, symbols: _SymbolTable) -> sp.Expr:
    parser = _Parser(java_expr, symbols)
    return parser.parse()


def _assert_negation_equivalence(java_expr: str) -> None:
    symbols = _SymbolTable(bool_map={}, operand_map={}, reverse_map={})
    original = _parse_with_symbols(java_expr, symbols)
    expected = sp.simplify(Not(original))
    output = negate_java_bool_expr(java_expr)
    got = _parse_with_symbols(output, symbols)
    got_simplified = sp.simplify(got)
    expected_simplified = sp.simplify(expected)
    if got_simplified != expected_simplified:
        raise AssertionError(
            "Negation mismatch\n"
            f"  input:   {java_expr}\n"
            f"  output:  {output}\n"
            f"  expected:{expected_simplified}\n"
            f"  got:     {got_simplified}"
        )


def _assert_simplify_equivalence(java_expr: str) -> None:
    symbols = _SymbolTable(bool_map={}, operand_map={}, reverse_map={})
    original = _parse_with_symbols(java_expr, symbols)
    expected = sp.simplify(original)
    output = simplify_java_bool_expr(java_expr)
    got = _parse_with_symbols(output, symbols)
    got_simplified = sp.simplify(got)
    expected_simplified = sp.simplify(expected)
    if got_simplified != expected_simplified:
        raise AssertionError(
            "Simplify mismatch\n"
            f"  input:   {java_expr}\n"
            f"  output:  {output}\n"
            f"  expected:{expected_simplified}\n"
            f"  got:     {got_simplified}"
        )


def _run_tests() -> None:
    cases = [
        "a",
        "!a",
        "a && b",
        "a || b",
        "a && (b || c)",
        "(a || b) && (c || d)",
        "true",
        "false",
        "a && true",
        "a || false",
        "x < y",
        "x <= y",
        "x > y",
        "x >= y",
        "x == y",
        "x != y",
        "(x < y) && (y >= z)",
        "foo(bar) && baz",
        "!(foo(bar) || baz)",
        "((user != null) && user.isAdmin())",
        "null == s || !Pattern.matches(\"\\\\d{1,5}\", s)",
        "Pattern.matches(\"\\\\d{1,5}\", s) && Integer.parseInt(s) > 65535 && null != s",
        "(null == s || !Pattern.matches(\"\\\\d{1,5}\", s)) || (Pattern.matches(\"\\\\d{1,5}\", s) && Integer.parseInt(s) > 65535 && null != s)",
        "((dimension % 4) + 4) % 4 != 1",
        "(dimension % 4 != 1)",
    ]

    for expr in cases:
        _assert_negation_equivalence(expr)
        print("Testing:", expr)
        print("Negation:", negate_java_bool_expr(expr))
        print()
    
    print(f"All tests passed: {len(cases)} cases")
    print()

    simplify_cases = [
        "a && true",
        "a || false",
        "a && false",
        "a || true",
        "!!a",
        "!(a || b) && c",
        "(x < y) || (x >= y)",
        "(x < y) && (x >= y)",
        "(a > 0) && (a != null) && (a > 0)",
        "((dimension % 4) + 4) % 4 != 1",
        "(dimension % 4 != 1)",
    ]

    for expr in simplify_cases:
        _assert_simplify_equivalence(expr)
        print("Simplify input:", expr)
        print("Simplified:", simplify_java_bool_expr(expr))
        print()

    print(f"All tests passed: {len(simplify_cases)} cases")


if __name__ == "__main__":
    _run_tests()
