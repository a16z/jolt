#!/usr/bin/env python3
"""Check the machine-checkable style invariants from CLAUDE.md (Style Invariants).

Repo-wide rules (always checked, on every tracked `.rs` file):

  folded-cfg-attr    Adjacent `#[cfg_attr(P, A)]` `#[cfg_attr(P, B)]` with the
                     same predicate must be folded into `#[cfg_attr(P, A, B)]`.
  native-visit       `#[allocative(visit = ...)]` must not decorate a container
                     of primitives; the native impls report strictly more.

Diff-scoped rules (checked on lines added relative to `--base`; without
`--base`, or when the merge base equals HEAD, they are skipped):

  nominal-imports    Types, traits, enums, and constants use imported short
                     names. Enum variants remain qualified by the enum type;
                     lowercase namespace function calls may stay qualified.
                     An imported path is never also spelled qualified.
  todo-issue-link    `TODO`/`FIXME` comments must carry an issue reference
                     (`#123` or a URL).

Diff-scoping exists because the rules are debt-tolerant by convention: style
passes fix what a PR introduces and leave predating sites alone.

CLAUDE.md rules that are NOT checked here, and why:

  - derive-over-hand-rolled impls, single-caller helper inlining — need call
    graphs and knowledge of what a derive can express.
  - speculative lifecycle guards (`RwLock<Option<..>>`) — the same type shape
    is legitimate for lazy-init globals and error slots; the violation is a
    lifecycle transition that never happens, which is semantic.
  - docs-honesty (enforced vs honest-encoder properties) and
    names-track-semantics — prose meaning, not syntax.

Exit status: 0 when clean, 1 when violations were found, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Pre-existing split cfg_attr pairs deliberately left alone by the style pass
# that introduced this rule ("predate this PR and are left alone", #1732).
# Remove entries as the sites get folded.
FOLDED_CFG_ATTR_EXEMPT = {
    "crates/jolt-field/src/solinas/ext.rs",
}

PRIMITIVE = r"(?:u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|i128|isize|bool|char|f32|f64)"
PRIMITIVE_CONTAINER = re.compile(
    r":\s*(?:Vec|Box)\s*<\s*(?:\[\s*)?(?:(?:Vec|Option)\s*<\s*)?" + PRIMITIVE + r"\b"
)
CFG_ATTR_START = re.compile(r"^\s*#\[cfg_attr\(")
TODO = re.compile(r"\b(?:TODO|FIXME)\b")
ISSUE_REF = re.compile(r"#\d+|https?://\S+")
USE_LINE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?use\s+(.+?);\s*(?://.*)?$", re.DOTALL)
MACRO_BODY_OPEN = re.compile(r"\b(?:macro_rules!\s*\w+|quote!|quote_spanned!)\s*[({\[]")
CFG_GATE = re.compile(r"^\s*#\[cfg(?:_attr)?\(")
QUALIFIED_PATH = re.compile(
    r"(?<![\w:$])(?P<path>(?:::)?(?:[A-Za-z_]\w*::)+[A-Za-z_]\w*)"
)
NOMINAL_SEGMENT = re.compile(r"^[A-Z][A-Za-z0-9]*$")
CONSTANT_SEGMENT = re.compile(r"^[A-Z][A-Z0-9_]*$")
ENUM_VARIANT_IMPORT = re.compile(
    r"(?P<enum>(?:[A-Za-z_]\w*::)*[A-Z][A-Za-z0-9]*)::"
    r"(?:[A-Z][A-Za-z0-9_]*|\{|\*)"
)


def run_git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def strip_strings_and_line_comments(line: str) -> str:
    """Blank out string literal contents and trailing // comments.

    Good enough for invariant matching: it does not need to survive every
    escape corner, only to stop paths inside strings/comments from matching.
    """
    out = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == '"':
            out.append('"')
            i += 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == '"':
                    break
                i += 1
            out.append('"')
            i += 1
        elif c == "'" and i + 2 < n and line[i + 1] == "\\":
            out.append("''")
            i += 3
            while i < n and line[i - 1] != "'":
                i += 1
        elif c == "/" and i + 1 < n and line[i + 1] == "/":
            break
        else:
            out.append(c)
            i += 1
    return "".join(out)


def line_masks(lines: list[str]) -> tuple[list[bool], list[bool]]:
    """Per-line masks: inside a multi-line raw string, inside a macro body.

    Both are conservative overapproximations driven by brace/quote counting;
    they exist to suppress false positives, not to parse Rust.
    """
    in_raw = [False] * len(lines)
    raw_close: str | None = None
    for i, ln in enumerate(lines):
        if raw_close is not None:
            in_raw[i] = True
            if raw_close in ln:
                raw_close = None
            continue
        m = re.search(r'r(#*)"', ln)
        if m:
            close = '"' + m.group(1)
            if close not in ln[m.end():]:
                raw_close = close

    in_macro = [False] * len(lines)
    depth = 0
    active = False
    for i, ln in enumerate(lines):
        if in_raw[i]:
            in_macro[i] = active
            continue
        stripped = strip_strings_and_line_comments(ln)
        if not active and MACRO_BODY_OPEN.search(stripped):
            active = True
            depth = 0
        if active:
            in_macro[i] = True
            depth += sum(stripped.count(c) for c in "{([")
            depth -= sum(stripped.count(c) for c in "})]")
            if depth <= 0:
                active = False
    return in_raw, in_macro


def inline_mod_mask(lines: list[str], in_raw: list[bool]) -> list[bool]:
    """Lines inside inline `mod x { ... }` blocks (typically `mod tests`).

    Inline modules are separate scopes: their imports do not apply to the
    rest of the file and vice versa, so the qualified-duplicate rule treats
    them as out of bounds rather than tracking per-scope imports.
    """
    mask = [False] * len(lines)
    depth = 0
    active = False
    for i, ln in enumerate(lines):
        if in_raw[i]:
            mask[i] = active
            continue
        stripped = strip_strings_and_line_comments(ln)
        if not active and re.match(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{", stripped):
            active = True
            depth = 0
        if active:
            mask[i] = True
            depth += stripped.count("{") - stripped.count("}")
            if depth <= 0:
                active = False
    return mask


def cfg_attr_predicate(attr_text: str) -> str | None:
    """Extract the predicate of a cfg_attr attribute: text up to the first
    top-level comma inside the parens. Whitespace-normalized."""
    start = attr_text.find("cfg_attr(")
    if start == -1:
        return None
    i = start + len("cfg_attr(")
    depth = 0
    pred = []
    while i < len(attr_text):
        c = attr_text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return None  # no top-level comma: attribute list is empty
            depth -= 1
        elif c == "," and depth == 0:
            return " ".join("".join(pred).split())
        pred.append(c)
        i += 1
    return None


def whole_line_attrs(lines: list[str]) -> list[tuple[int, int, str]]:
    """All attributes that occupy whole lines: (start, end, text).

    Attributes sharing a line with code (e.g. on function parameters) are
    excluded — folding only applies to stacked item attributes.
    """
    attrs = []
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if not stripped.startswith("#["):
            i += 1
            continue
        depth = 0
        raw = []
        start = i
        while i < len(lines):
            clean = strip_strings_and_line_comments(lines[i])
            raw.append(lines[i])
            depth += clean.count("[") - clean.count("]")
            if depth <= 0:
                break
            i += 1
        end_line = lines[min(i, len(lines) - 1)]
        after = strip_strings_and_line_comments(end_line)
        # whole-line iff nothing follows the closing bracket
        if after.rstrip().endswith("]"):
            attrs.append((start, i, " ".join(raw)))
        i += 1
    return attrs


def check_folded_cfg_attr(rel: str, lines: list[str]) -> list[tuple[int, str]]:
    if rel in FOLDED_CFG_ATTR_EXEMPT:
        return []
    findings = []
    attrs = whole_line_attrs(lines)
    for (s1, e1, t1), (s2, _e2, t2) in zip(attrs, attrs[1:]):
        if s2 != e1 + 1:
            continue  # not adjacent
        p1, p2 = cfg_attr_predicate(t1), cfg_attr_predicate(t2)
        if p1 is not None and p1 == p2:
            findings.append(
                (
                    s2 + 1,
                    f"[folded-cfg-attr] adjacent cfg_attr with identical predicate "
                    f"`{p1}`; fold into one `#[cfg_attr({p1}, ..., ...)]`",
                )
            )
    return findings


def check_native_visit(rel: str, lines: list[str]) -> list[tuple[int, str]]:
    findings = []
    for i, ln in enumerate(lines):
        if "allocative(visit" not in ln:
            continue
        for j in range(i + 1, min(i + 8, len(lines))):
            nxt = lines[j].strip()
            if not nxt or nxt.startswith(("#", "//", "/*", "*")):
                continue
            if PRIMITIVE_CONTAINER.search(nxt):
                findings.append(
                    (
                        i + 1,
                        "[native-visit] `allocative(visit = ...)` on a primitive-element "
                        "container; drop the attribute and let the native impl render "
                        "(visit helpers are for foreign-scalar element types only)",
                    )
                )
            break
    return findings


def expand_use_tree(prefix: str, tree: str, out: set[str]) -> None:
    tree = tree.strip()
    if tree.startswith("{") and tree.endswith("}"):
        depth = 0
        item = []
        for c in tree[1:-1]:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if c == "," and depth == 0:
                expand_use_tree(prefix, "".join(item), out)
                item = []
            else:
                item.append(c)
        if item:
            expand_use_tree(prefix, "".join(item), out)
        return
    if " as " in tree:
        return  # aliased: qualifying the original path elsewhere may be deliberate
    if tree.endswith("*") or tree == "self":
        if tree == "self" and prefix:
            out.add(prefix.rstrip(":"))
        return
    if "{" in tree:
        head, rest = tree.split("{", 1)
        expand_use_tree(prefix + head, "{" + rest, out)
        return
    full = (prefix + tree).strip()
    if "::" in full:
        out.add(full)


def collect_imports(
    lines: list[str], in_raw: list[bool], in_macro: list[bool], in_mod: list[bool]
) -> set[str]:
    """Exact paths imported by top-level plain (non-cfg-gated, non-aliased)
    use items. Imports inside inline modules are separate scopes and skipped."""
    imports: set[str] = set()
    i = 0
    while i < len(lines):
        if in_raw[i] or in_macro[i] or in_mod[i]:
            i += 1
            continue
        # skip cfg-gated use items: the import may not be live in every cfg arm
        prev = next(
            (lines[k].strip() for k in range(i - 1, -1, -1) if lines[k].strip()), ""
        )
        stripped = lines[i].strip()
        if not re.match(r"^(?:pub(?:\([^)]*\))?\s+)?use\b", stripped):
            i += 1
            continue
        stmt_lines = []
        j = i
        while j < len(lines):
            stmt_lines.append(strip_strings_and_line_comments(lines[j]))
            if ";" in lines[j]:
                break
            j += 1
        stmt = " ".join(stmt_lines)
        m = USE_LINE.match(stmt.strip())
        if m and not CFG_GATE.match(prev):
            expand_use_tree("", m.group(1).strip(), imports)
        i = j + 1
    return imports


def check_qualified_dup(
    rel: str, lines: list[str], in_raw: list[bool], in_macro: list[bool]
) -> list[tuple[int, str]]:
    in_mod = inline_mod_mask(lines, in_raw)
    imports = collect_imports(lines, in_raw, in_macro, in_mod)
    if not imports:
        return []
    in_attr = [False] * len(lines)
    for start, end, _text in whole_line_attrs(lines):
        for k in range(start, end + 1):
            in_attr[k] = True
    patterns = {
        path: re.compile(r"(?<![\w:$])" + re.escape(path) + r"(?![\w!])")
        for path in imports
    }
    findings = []
    for i, ln in enumerate(lines):
        if in_raw[i] or in_macro[i] or in_attr[i] or in_mod[i]:
            continue
        s = ln.strip()
        if s.startswith(("//", "///", "//!", "*")) or re.search(r"\buse\b", s):
            continue
        clean = strip_strings_and_line_comments(ln)
        clean = re.sub(r"#!?\[[^\]]*\]", "", clean)  # attribute args are exempt
        for path, pat in patterns.items():
            if pat.search(clean):
                findings.append(
                    (
                        i + 1,
                        f"[nominal-imports] `{path}` is already imported in this "
                        f"file; use the imported name",
                    )
                )
                break
    return findings


def check_nominal_paths(
    rel: str, lines: list[str], in_raw: list[bool], in_macro: list[bool]
) -> list[tuple[int, str]]:
    in_mod = inline_mod_mask(lines, in_raw)
    in_attr = [False] * len(lines)
    for start, end, _text in whole_line_attrs(lines):
        for k in range(start, end + 1):
            in_attr[k] = True

    findings = []
    for i, ln in enumerate(lines):
        if in_raw[i] or in_macro[i] or in_attr[i] or in_mod[i]:
            continue
        stripped = ln.strip()
        if stripped.startswith(("//", "///", "//!", "*")) or re.search(
            r"\buse\b", stripped
        ):
            continue
        clean = strip_strings_and_line_comments(ln)
        clean = re.sub(r"#!?\[[^\]]*\]", "", clean)
        clean = re.sub(r"<[^<>]*\bas\s+[^<>]*>", "", clean)
        for match in QUALIFIED_PATH.finditer(clean):
            path = match.group("path").removeprefix("::")
            parts = path.split("::")
            suffix = clean[match.end():].lstrip()
            nominal = next(
                (j for j, part in enumerate(parts) if NOMINAL_SEGMENT.match(part)),
                None,
            )
            if (
                nominal == len(parts) - 1
                and len(parts) == 2
                and re.fullmatch(PRIMITIVE, parts[0])
            ):
                continue
            if nominal is not None and nominal > 0:
                import_path = "::".join(parts[: nominal + 1])
                short_path = "::".join(parts[nominal:])
                findings.append(
                    (
                        i + 1,
                        f"[nominal-imports] import `{import_path}` and use "
                        f"`{short_path}`",
                    )
                )
                break
            if nominal is None and CONSTANT_SEGMENT.match(parts[-1]):
                findings.append(
                    (
                        i + 1,
                        f"[nominal-imports] import `{path}` and use "
                        f"`{parts[-1]}`",
                    )
                )
                break
            if nominal is None and not suffix.startswith(("(", "::<", "!")):
                findings.append(
                    (
                        i + 1,
                        f"[nominal-imports] import `{path}` and use "
                        f"`{parts[-1]}`",
                    )
                )
                break
    return findings


def check_enum_variant_imports(
    rel: str, lines: list[str], in_raw: list[bool], in_macro: list[bool]
) -> list[tuple[int, str]]:
    in_mod = inline_mod_mask(lines, in_raw)
    findings = []
    for i, ln in enumerate(lines):
        if in_raw[i] or in_macro[i] or in_mod[i] or not re.search(r"\buse\b", ln):
            continue
        clean = strip_strings_and_line_comments(ln)
        if match := ENUM_VARIANT_IMPORT.search(clean):
            enum_path = match.group("enum")
            enum_name = enum_path.rsplit("::", 1)[-1]
            findings.append(
                (
                    i + 1,
                    f"[nominal-imports] import enum `{enum_path}` and keep its "
                    f"variant qualified as `{enum_name}::VARIANT`",
                )
            )
    return findings


def check_todo_issue(rel: str, lines: list[str]) -> list[tuple[int, str]]:
    findings = []
    for i, ln in enumerate(lines):
        if not TODO.search(ln):
            continue
        comment = None
        if "//" in ln:
            comment = ln[ln.index("//"):]
        elif "/*" in ln:
            comment = ln[ln.index("/*"):]
        if comment and TODO.search(comment) and not ISSUE_REF.search(comment):
            findings.append(
                (i + 1, "[todo-issue-link] TODO/FIXME without an issue link; "
                        "use `TODO(#123):` or a full issue URL")
            )
    return findings


def added_lines_by_file(base: str) -> dict[str, set[int]] | None:
    try:
        merge_base = run_git(["merge-base", base, "HEAD"]).strip()
    except subprocess.CalledProcessError as e:
        print(f"error: cannot resolve merge base with {base}: {e.stderr.strip()}",
              file=sys.stderr)
        raise SystemExit(2) from e
    diff = run_git(["diff", "-U0", "--no-color", merge_base, "--", "*.rs"])
    added: dict[str, set[int]] = {}
    current = None
    for ln in diff.splitlines():
        if ln.startswith("+++ b/"):
            current = ln[6:]
        elif ln.startswith("@@") and current:
            m = re.search(r"\+(\d+)(?:,(\d+))?", ln)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) is not None else 1
                added.setdefault(current, set()).update(range(start, start + count))
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        help="git ref to diff against (e.g. origin/main); enables the "
             "diff-scoped rules on lines added since the merge base",
    )
    args = parser.parse_args()

    files = [
        f for f in run_git(["ls-files", "*.rs"]).splitlines()
        if Path(f).is_file()
    ]

    added = added_lines_by_file(args.base) if args.base else None

    findings: list[tuple[str, int, str]] = []
    for rel in files:
        text = Path(rel).read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        for line_no, msg in check_folded_cfg_attr(rel, lines):
            findings.append((rel, line_no, msg))
        for line_no, msg in check_native_visit(rel, lines):
            findings.append((rel, line_no, msg))

        if added is not None and rel in added:
            in_raw, in_macro = line_masks(lines)
            scoped = (
                check_qualified_dup(rel, lines, in_raw, in_macro)
                + check_nominal_paths(rel, lines, in_raw, in_macro)
                + check_enum_variant_imports(rel, lines, in_raw, in_macro)
                + check_todo_issue(rel, lines)
            )
            for line_no, msg in scoped:
                if line_no in added[rel]:
                    findings.append((rel, line_no, msg))

    for rel, line_no, msg in sorted(findings):
        print(f"{rel}:{line_no}: {msg}")
    if findings:
        print(f"\n{len(findings)} style invariant violation(s); "
              f"see CLAUDE.md (Style Invariants)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
