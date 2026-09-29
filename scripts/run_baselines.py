"""Run: python scripts/run_baselines.py --output results/phase1.json"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_unlearning.experiments import CASES, run_case


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "phase1.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    source_paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]
    report = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                  git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")),
                  python=sys.version, platform=platform.platform(), device="CPU",
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source_paths},
                  timing_note="Instrumented single runs on tiny MDPs; correctness pilot, not a speedup claim. Request time includes required solver verification; exact oracle is offline evaluation overhead.",
                  runs=[run_case(case, seed) for seed in args.seeds for case in CASES])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    failures = [r for run in report["runs"] for r in run["records"]
                if r["method"] != "noop" and (r["policy_disagreement"] or not r["converged"])]
    print(f"Saved {len(report['runs'])} cases / {sum(len(r['records']) for r in report['runs'])} method runs to {args.output}")
    print(f"Non-noop policy/convergence failures: {len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
