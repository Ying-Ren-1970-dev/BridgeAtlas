"""Run local preflight checks: regression + benchmark suites."""

from __future__ import annotations

import subprocess
import sys
import time
from typing import List, Tuple


CHECKS: List[Tuple[str, List[str]]] = [
    ("Identifier regression", [sys.executable, "test_identifier_query_regression.py"]),
    ("Default benchmark", [sys.executable, "benchmark_defaults.py"]),
    ("Mar Vista benchmark", [sys.executable, "benchmark_marvista.py"]),
]


def run_check(name: str, command: List[str]) -> int:
    print("\n" + "=" * 90)
    print(f"RUNNING: {name}")
    print(f"COMMAND: {' '.join(command)}")
    print("=" * 90)

    started = time.time()
    completed = subprocess.run(command)
    elapsed = time.time() - started

    if completed.returncode == 0:
        print(f"\nPASS: {name} ({elapsed:.1f}s)")
    else:
        print(f"\nFAIL: {name} ({elapsed:.1f}s, exit={completed.returncode})")

    return completed.returncode


def main() -> int:
    print("Starting preflight checks...")
    print("Note: Ensure local API is running at http://localhost:8000")

    failures = 0
    for name, command in CHECKS:
        rc = run_check(name, command)
        if rc != 0:
            failures += 1

    print("\n" + "=" * 90)
    if failures == 0:
        print("ALL PRECHECKS PASSED")
        return 0

    print(f"PRECHECKS FAILED: {failures} check(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
