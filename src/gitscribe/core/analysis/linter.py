"""
core/analysis/linter.py
Stage 3: wraps ruff (already a project dependency) — not reinventing a
linter, per the explicit non-goal. Normalizes ruff's JSON output into
symbol-scoped findings so risk_classifier and CLI output share one shape.
"""

from __future__ import annotations

import json
import subprocess

from pydantic import BaseModel


class LintFinding(BaseModel):
    file: str
    lineno: int
    code: str  # ruff rule code, e.g. "F401"
    message: str
    severity: str  # "error" | "warning" -- derived from ruff code prefix


_ERROR_PREFIXES = ("F", "E9", "S")  # pyflakes/syntax/security -> treated as errors


def _severity_for(code: str) -> str:
    return "error" if code.startswith(_ERROR_PREFIXES) else "warning"


def run_ruff(repo_root: str = ".") -> list[LintFinding]:
    result = subprocess.run(
        ["ruff", "check", repo_root, "--output-format=json"],
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        return []

    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    findings = []
    for item in raw:
        code = item.get("code") or ""
        findings.append(
            LintFinding(
                file=item["filename"],
                lineno=item["location"]["row"],
                code=code,
                message=item.get("message", ""),
                severity=_severity_for(code),
            )
        )
    return findings


def findings_by_file(findings: list[LintFinding]) -> dict[str, list[LintFinding]]:
    grouped: dict[str, list[LintFinding]] = {}
    for f in findings:
        grouped.setdefault(f.file, []).append(f)
    return grouped


def severity_score(findings: list[LintFinding]) -> float:
    """0-1 normalized signal for risk_classifier: proportion of findings
    that are errors vs warnings. Simple, explainable, no magic weighting.
    """
    if not findings:
        return 0.0
    errors = sum(1 for f in findings if f.severity == "error")
    return errors / len(findings)
