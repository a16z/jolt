import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "check_style_invariants.py"
SPEC = importlib.util.spec_from_file_location("check_style_invariants", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {MODULE_PATH}")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def check(lines: list[str]) -> list[tuple[int, str]]:
    in_raw, in_macro = CHECKER.line_masks(lines)
    return CHECKER.check_nominal_paths("test.rs", lines, in_raw, in_macro)


def check_variant_imports(lines: list[str]) -> list[tuple[int, str]]:
    in_raw, in_macro = CHECKER.line_masks(lines)
    return CHECKER.check_enum_variant_imports("test.rs", lines, in_raw, in_macro)


class CheckStyleInvariantsTests(unittest.TestCase):
    def test_requires_import_for_nominal_item(self) -> None:
        findings = check(["let value = std::sync::Arc::new(1);"])
        self.assertEqual(len(findings), 1)
        self.assertIn("import `std::sync::Arc`", findings[0][1])

    def test_requires_enum_import_but_keeps_variant_qualified(self) -> None:
        findings = check(["let kind = crate::instruction::Kind::ADD;"])
        self.assertEqual(len(findings), 1)
        self.assertIn("use `Kind::ADD`", findings[0][1])
        self.assertEqual(check(["let kind = Kind::ADD;"]), [])

    def test_rejects_imported_enum_variant(self) -> None:
        findings = check_variant_imports(["use crate::instruction::Kind::ADD;"])
        self.assertEqual(len(findings), 1)
        self.assertIn("import enum `crate::instruction::Kind`", findings[0][1])
        self.assertEqual(
            check_variant_imports(["use crate::instruction::Kind;"]), []
        )

    def test_allows_namespace_function_calls(self) -> None:
        self.assertEqual(
            check(
                [
                    "std::mem::take(&mut value);",
                    "std::alloc::alloc(layout);",
                    "ram::reconstruct_full_eval();",
                ]
            ),
            [],
        )
        self.assertEqual(len(check(["let take = std::mem::take;"])), 1)

    def test_allows_associated_items_on_short_type_names(self) -> None:
        self.assertEqual(check(["let max = u64::MAX;", "let kind = Kind::ADD;"]), [])

    def test_allows_ufcs_for_disambiguation(self) -> None:
        self.assertEqual(
            check(["<D::Error as serde::de::Error>::custom(message);"]), []
        )


if __name__ == "__main__":
    unittest.main()
