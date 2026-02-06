#!/usr/bin/env python3
"""
Update test-count references in documentation using pytest collection output.

Usage:
  uv run python scripts/update_test_counts.py           # update files in-place
  uv run python scripts/update_test_counts.py --check   # fail if updates are needed
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]


def run_pytest_collect() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "pytest collection failed. stderr:\n" + (proc.stderr.strip() or "(empty)")
        )
    return proc.stdout


def parse_counts(collect_output: str) -> Dict[str, int]:
    module_re = re.compile(r"^\s*<Module ([^>]+)>")
    func_re = re.compile(r"^\s*<Function ")
    counts: Dict[str, int] = {}
    current: str | None = None
    for line in collect_output.splitlines():
        mod_match = module_re.match(line)
        if mod_match:
            module_path = mod_match.group(1).strip()
            module_name = Path(module_path).name
            if module_name.startswith("test_"):
                current = module_name
                counts.setdefault(current, 0)
            else:
                current = None
            continue
        if current and func_re.match(line):
            counts[current] += 1
    return counts


def update_readme(text: str, total: int) -> str:
    return re.sub(
        r"Test suite \(.*?tests\)",
        f"Test suite ({total:,}+ tests)",
        text,
    )


def update_tests_readme(text: str, counts: Dict[str, int], total: int) -> str:
    lines = text.splitlines()
    updated: List[str] = []

    for line in lines:
        if line.strip().startswith("**Total:") and "comprehensive tests" in line:
            line = f"**Total: {total:,} comprehensive tests**"

        if line.strip().startswith("| **Total** |"):
            line = f"| **Total** | **{total:,}** | **Collected via pytest --collect-only** |"

        # Update table rows for test files
        if line.strip().startswith("| `test_") and "|" in line:
            match = re.search(r"`(test_[^`]+)`", line)
            if match:
                test_file = match.group(1)
                if test_file in counts:
                    line = re.sub(
                        r"\| `test_[^`]+` \| \d+ \|",
                        f"| `{test_file}` | {counts[test_file]} |",
                        line,
                    )

        # Update completed enhancements counts
        for test_file, count in counts.items():
            pattern = rf"({re.escape(test_file)}\) - )\d+ tests"
            if re.search(pattern, line):
                line = re.sub(pattern, rf"\g<1>{count} tests", line)

        # Update overall total mentions
        if "Comprehensive test coverage" in line:
            line = re.sub(
                r"\(\d[\d,]* tests total\)",
                f"({total:,} tests total)",
                line,
            )

        updated.append(line)

    return "\n".join(updated) + "\n"


def update_diff_comparison_docs(text: str, count: int) -> str:
    return re.sub(
        r"\*\*Total: \d+ tests\*\*",
        f"**Total: {count} tests**",
        text,
    )


def update_output_formats_docs(text: str, count: int) -> str:
    text = re.sub(
        r"includes \d+ comprehensive tests covering",
        f"includes {count} comprehensive tests covering",
        text,
    )
    text = re.sub(
        r"\*\*Fully Tested:\*\* \d+ comprehensive tests",
        f"**Fully Tested:** {count} comprehensive tests",
        text,
    )
    return text


def apply_updates(files: List[Tuple[Path, str]]) -> int:
    changed = 0
    for path, new_content in files:
        old_content = path.read_text()
        if old_content != new_content:
            path.write_text(new_content)
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if updates are needed.")
    args = parser.parse_args()

    collect_output = run_pytest_collect()
    counts = parse_counts(collect_output)
    if not counts:
        raise RuntimeError("No tests detected from pytest collection output.")

    total = sum(counts.values())

    files: List[Tuple[Path, str]] = []

    readme_path = ROOT / "README.md"
    files.append((readme_path, update_readme(readme_path.read_text(), total)))

    tests_readme_path = ROOT / "tests" / "README.md"
    files.append(
        (tests_readme_path, update_tests_readme(tests_readme_path.read_text(), counts, total))
    )

    diff_docs_path = ROOT / "docs" / "DIFF_COMPARISON.md"
    if diff_docs_path.exists() and "test_diff_comparison.py" in counts:
        files.append(
            (diff_docs_path, update_diff_comparison_docs(diff_docs_path.read_text(), counts["test_diff_comparison.py"]))
        )

    output_docs_path = ROOT / "docs" / "OUTPUT_FORMATS.md"
    if output_docs_path.exists() and "test_output_formats.py" in counts:
        files.append(
            (output_docs_path, update_output_formats_docs(output_docs_path.read_text(), counts["test_output_formats.py"]))
        )

    if args.check:
        for path, new_content in files:
            if path.read_text() != new_content:
                print(f"Out of date: {path.relative_to(ROOT)}")
                return 1
        print("Test counts are up to date.")
        return 0

    changed = apply_updates(files)
    print(f"Updated {changed} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
