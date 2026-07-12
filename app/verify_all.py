"""Unified verification runner — the QA gate for the physics kernels.

Runs every closed-form verification suite and prints a single pass/fail table, the
way a commercial CAE vendor maintains a verification manual: each release must show
its physics still reproduces the analytic references.

    python -m app.verify_all            # fast suites (CLT, liner, netting)
    python -m app.verify_all --full     # + general-geometry FEA validation (meshes + solves)

Exit code 0 only if every suite passes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all physics verification suites.")
    ap.add_argument("--full", action="store_true",
                    help="Include the general-geometry FEA validation (builds meshes and solves; minutes)")
    args = ap.parse_args()

    from app import verify_clt, verify_liner, verify_netting
    suites = [
        ("Classical laminate theory (CLT)", verify_clt.main),
        ("Type 3/4 liner (thin-wall closed form)", verify_liner.main),
        ("Netting analysis (closed form)", verify_netting.main),
    ]
    if args.full:
        from app import validate_general
        suites.append(("General-geometry FEA vs analytic COPV", validate_general.main))

    results = []
    for name, fn in suites:
        print("\n" + "=" * 66 + f"\n SUITE: {name}\n" + "=" * 66)
        t0 = time.time()
        try:
            rc = int(fn())
        except Exception as exc:  # a crashed suite is a failure, not an abort
            print(f"  suite crashed: {exc}")
            rc = 1
        results.append((name, rc, time.time() - t0))

    print("\n" + "=" * 66 + "\n VERIFICATION SUMMARY\n" + "=" * 66)
    for name, rc, dt in results:
        print(f"  [{'PASS' if rc == 0 else 'FAIL'}] {name:<44} {dt:6.1f}s")
    overall = all(rc == 0 for _, rc, _ in results)
    print(f"\n OVERALL: {'ALL SUITES PASS' if overall else 'FAILURES PRESENT'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
