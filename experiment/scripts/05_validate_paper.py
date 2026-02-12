#!/usr/bin/env python3
"""
Paper Quality Validation Script
Checks grammar, citations, references, and LaTeX issues before PDF compilation
"""

import re
import subprocess
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PAPER_DIR = BASE_DIR / "paper"


def check_latex_compilation():
    """Check if LaTeX compiles without errors"""
    print("=" * 60)
    print("1. LATEX COMPILATION CHECK")
    print("=" * 60)

    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
        cwd=PAPER_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    warnings = []
    errors = []

    for line in result.stdout.split("\n"):
        if "Warning" in line:
            warnings.append(line.strip())
        if "Error" in line or "!" in line:
            errors.append(line.strip())

    print(f"✓ Compilation {'succeeded' if result.returncode == 0 else 'FAILED'}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Errors: {len(errors)}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings[:10]:
            print(f"  - {warning}")

    return len(errors) == 0


def check_citations():
    r"""Verify all \cite{} have matching BibTeX entries"""
    print("\n" + "=" * 60)
    print("2. CITATION INTEGRITY CHECK")
    print("=" * 60)

    paper_tex = (PAPER_DIR / "paper.tex").read_text()
    cited_keys = set(re.findall(r"\\cite\{([^}]+)\}", paper_tex))

    all_cited = set()
    for keys in cited_keys:
        all_cited.update(key.strip() for key in keys.split(","))

    bib_file = PAPER_DIR / "references.bib"
    if not bib_file.exists():
        print("✗ references.bib not found!")
        return False

    bib_text = bib_file.read_text()
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))

    missing = all_cited - bib_keys
    unused = bib_keys - all_cited

    print(f"✓ Citations in paper: {len(all_cited)}")
    print(f"✓ BibTeX entries: {len(bib_keys)}")

    if missing:
        print(f"\n✗ MISSING BibTeX entries ({len(missing)}):")
        for key in sorted(missing):
            print(f"  - {key}")

    if unused:
        print(f"\n⚠ Unused BibTeX entries ({len(unused)}):")
        for key in sorted(unused)[:5]:
            print(f"  - {key}")

    return len(missing) == 0


def check_references():
    r"""Verify all \ref{} have matching \label{}"""
    print("\n" + "=" * 60)
    print("3. REFERENCE INTEGRITY CHECK")
    print("=" * 60)

    paper_tex = (PAPER_DIR / "paper.tex").read_text()

    refs = set(re.findall(r"\\ref\{([^}]+)\}", paper_tex))
    labels = set(re.findall(r"\\label\{([^}]+)\}", paper_tex))

    table_files = list((PAPER_DIR / "tables").glob("*.tex"))
    for table_file in table_files:
        table_text = table_file.read_text()
        labels.update(re.findall(r"\\label\{([^}]+)\}", table_text))

    missing_labels = refs - labels
    unused_labels = labels - refs

    print(f"✓ References (\\ref): {len(refs)}")
    print(f"✓ Labels (\\label): {len(labels)}")

    if missing_labels:
        print(f"\n✗ MISSING labels ({len(missing_labels)}):")
        for label in sorted(missing_labels):
            print(f"  - {label}")

    if unused_labels:
        print(f"\n⚠ Unused labels ({len(unused_labels)}):")
        for label in sorted(unused_labels):
            print(f"  - {label}")

    return len(missing_labels) == 0


def check_grammar():
    """Check grammar and spelling using language_tool_python"""
    print("\n" + "=" * 60)
    print("4. GRAMMAR & SPELLING CHECK")
    print("=" * 60)

    try:
        import language_tool_python
    except ImportError:
        print("⚠ language_tool_python not installed")
        print("  Install: pip install language-tool-python")
        return True

    tool = language_tool_python.LanguageTool("en-US")

    paper_tex = (PAPER_DIR / "paper.tex").read_text()

    text_lines = []
    for line in paper_tex.split("\n"):
        if not line.strip().startswith("\\") and line.strip():
            clean = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", line)
            clean = re.sub(r"\\[a-zA-Z]+", "", clean)
            if clean.strip():
                text_lines.append(clean)

    text = " ".join(text_lines)

    matches = tool.check(text)

    real_issues = []
    for match in matches:
        if "WHITESPACE_RULE" in match.ruleId:
            continue
        if match.context.count("\\") > 0:
            continue
        real_issues.append(match)

    print(f"✓ Grammar/spelling issues found: {len(real_issues)}")

    if real_issues:
        print("\nTop issues:")
        for issue in real_issues[:10]:
            print(f"  - Line ~{issue.offset}: {issue.message}")
            print(f"    Context: ...{issue.context}...")

    return len(real_issues) < 20


def check_common_issues():
    """Check for common LaTeX issues"""
    print("\n" + "=" * 60)
    print("5. COMMON ISSUES CHECK")
    print("=" * 60)

    paper_tex = (PAPER_DIR / "paper.tex").read_text()

    issues = []
    issue_counts = defaultdict(int)

    if "TODO" in paper_tex or "FIXME" in paper_tex:
        issue_counts["todo_fixme"] += 1
        issues.append("Contains TODO/FIXME markers")

    if "{{" in paper_tex or "}}" in paper_tex:
        issue_counts["placeholder"] += 1
        issues.append("Contains placeholder {{}} markers")

    if "  " in paper_tex:
        count = paper_tex.count("  ")
        issue_counts["double_space"] += 1
        issues.append(f"Contains {count} double spaces")

    lines = paper_tex.split("\n")
    for line in lines:
        if line.strip() and not line.strip().startswith("%"):
            if line.strip().endswith("\\\\") and not line.strip().endswith(".\\\\"):
                pass

    total_issues = sum(issue_counts.values())
    print(f"✓ Common issues found: {total_issues}")
    for issue in issues:
        print(f"  - {issue}")

    return total_issues == 0


def main():
    print("\n" + "=" * 60)
    print("PAPER QUALITY VALIDATION")
    print("=" * 60)
    print(f"Paper: {PAPER_DIR / 'paper.tex'}")
    print()

    results = {
        "LaTeX Compilation": check_latex_compilation(),
        "Citation Integrity": check_citations(),
        "Reference Integrity": check_references(),
        "Grammar & Spelling": check_grammar(),
        "Common Issues": check_common_issues(),
    }

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {check}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready for PDF compilation")
    else:
        print("✗ SOME CHECKS FAILED - Fix issues before PDF compilation")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
