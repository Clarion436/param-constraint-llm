"""
Java code formatter using astyle_py.

Usage:
    from java_format import format_java_code, format_java_code_keep_comments

    # Remove comments, then format
    formatted = format_java_code(java_code_string)

    # Format only (keep comments)
    formatted = format_java_code_keep_comments(java_code_string)
"""

import astyle_py
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

# ---------- tree-sitter init ----------

_JAVA_LANG = Language(tsjava.language())
_PARSER = Parser(_JAVA_LANG)


# ---------- comment removal ----------

def remove_comments(java_code: str) -> str:
    """
    Remove all comments (line_comment, block_comment) from Java source code
    using tree-sitter for precise parsing.

    String literals and character literals are NOT affected — only actual
    comments are removed.

    Args:
        java_code: The Java source code string.

    Returns:
        The code with all comments removed.
    """
    if not java_code or not java_code.strip():
        return java_code

    src_bytes = java_code.encode('utf-8')
    tree = _PARSER.parse(src_bytes)
    root = tree.root_node

    # Collect all comment node byte ranges
    comment_ranges = []

    def collect_comments(node):
        if node.type in ('line_comment', 'block_comment'):
            comment_ranges.append((node.start_byte, node.end_byte))
        for child in node.children:
            collect_comments(child)

    collect_comments(root)

    # Remove comments in reverse order (avoids byte-offset drift)
    result_bytes = bytearray(src_bytes)
    for start, end in sorted(comment_ranges, reverse=True):
        result_bytes[start:end] = b''

    return result_bytes.decode('utf-8')


# ---------- formatting ----------

def format_java_code(java_code: str, version: str = "3.1") -> str:
    """
    Format a Java code string: remove comments, then format with astyle
    --style=java.

    Args:
        java_code: The Java source code string to format.
        version: Astyle version to use (default: "3.1").
                 Supported versions: "3.1", "3.4.7"

    Returns:
        The formatted Java code string (comments removed).

    Raises:
        astyle_py.AstyleError: If formatting fails.
    """
    # Step 1: remove comments
    code_no_comments = remove_comments(java_code)

    # Step 2: format with astyle
    astyle = astyle_py.Astyle(version=version)
    astyle.set_options("--style=java --delete-empty-lines --indent-col1-comments")
    return astyle.format(code_no_comments).replace('\r\n', '\n')


def format_java_code_keep_comments(java_code: str, version: str = "3.1") -> str:
    """
    Format a Java code string with astyle --style=java, preserving comments.

    Unlike format_java_code(), this function does NOT strip comments before
    formatting.

    Args:
        java_code: The Java source code string to format.
        version: Astyle version to use (default: "3.1").
                 Supported versions: "3.1", "3.4.7"

    Returns:
        The formatted Java code string (comments preserved).

    Raises:
        astyle_py.AstyleError: If formatting fails.
    """
    astyle = astyle_py.Astyle(version=version)
    astyle.set_options("--style=java --delete-empty-lines --indent-col1-comments")
    return astyle.format(java_code).replace('\r\n', '\n')


# ---------- helpers ----------

def _run_test(title: str, code: str):
    """Run a single test case and print input/output."""
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(">>> input:")
    print(code.rstrip())
    print(">>> output:")
    try:
        result = format_java_code(code)
        print(result.rstrip())
    except astyle_py.AstyleError as e:
        print(f"error: {e}")
    print()


# ---------- test cases (single Java functions only) ----------

def test_basic():
    """Basic formatting: indentation, brace style, comments removed."""
    _run_test("Test 1: basic formatting (comments removed + --style=java)", """\
public static boolean isPalindrome(int number) {
// check negative
if (number < 0) {
return false;
}
int original = number;
int reversed = 0;
/* reverse digits */
while (number != 0) {
reversed = reversed * 10 + number % 10;
number /= 10;
}
return original == reversed; // return result
}""")


def test_delete_empty_lines():
    """--delete-empty-lines: blank lines inside the function are removed, comments removed."""
    _run_test("Test 2: --delete-empty-lines removes extra blank lines inside the function", """\
public static boolean isPalindrome(int number) {

// negative numbers are not palindromes
if (number < 0) {

return false;

}

int original = number;


int reversed = 0;

while (number != 0) {

reversed = reversed * 10 + number % 10;

number /= 10;
}


return original == reversed;

}""")


def test_chinese_string_literal():
    """Non-ASCII string literals are preserved; comment markers inside strings are not removed."""
    _run_test("Test 3: non-ASCII string literals (with pseudo comment markers)", """\
public static String getGreeting(String name) {
String greeting = "hello, world!";
String url = "https://example.com"; // real comment
String fake = "this // is not /* a comment */"; // real comment
String template = "invalid input for user %s: %d";
return greeting + "\\n" + url + "\\n" + fake;
}""")


def test_chinese_comments():
    """Non-ASCII comments (single-line + block) are all removed."""
    _run_test("Test 4: non-ASCII comments removed", """\
public static boolean isPalindrome(int number) {
// negative numbers are not palindromes
if (number < 0) {

return false;

}

int original = number;
int reversed = 0;

// reverse the number digit by digit, from least to most significant
while (number != 0) {

reversed = reversed * 10 + number % 10;
number /= 10;

}

/* Compare the original and reversed numbers:
* if equal, it is a palindrome.
* A block comment is used here to exercise multi-line comment handling.
*/
return original == reversed;
}""")


def test_mixed():
    """Combined: non-ASCII comments + non-ASCII strings + blank lines."""
    _run_test("Test 5: combined (non-ASCII comments removed + non-ASCII strings kept + blank lines removed)", """\
public static boolean isPalindrome(int number) {
// negative numbers are not palindromes
if (number < 0) {

return false;

}

int original = number;

int reversed = 0;

// reverse digit by digit
while (number != 0) {

reversed = reversed * 10 + number % 10;

number /= 10;

}

/* Final comparison:
* original == reversed -> palindrome
*/
System.out.println("result: original=" + original + ", reversed=" + reversed);
return original == reversed;
}""")


def test_comment_only():
    """A function with only comments and no code."""
    _run_test("Test 6: after removing comments only an empty body remains", """\
public static int getValue() {
// TODO: implement this method
/* This method should return
the configured value */
}""")


# ---------- main ----------

if __name__ == "__main__":
    test_basic()
    test_delete_empty_lines()
    test_chinese_string_literal()
    test_chinese_comments()
    test_mixed()
    test_comment_only()
    print("=" * 60)
    print("all tests done")
