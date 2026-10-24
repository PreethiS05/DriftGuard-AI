"""Regenerate the bundled SYNTHETIC sample CSVs in data/samples/.

Usage:  python scripts/generate_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driftguard.synthetic import generate_scenario  # noqa: E402

OUT = ROOT / "data" / "samples"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stable = generate_scenario("stable", 5000, 3000, seed=7)
    stable.reference.to_csv(OUT / "reference_customers.csv", index=False)
    stable.current.to_csv(OUT / "current_customers_stable.csv", index=False)
    generate_scenario("combined", 5000, 3000, seed=7).current.to_csv(
        OUT / "current_customers_drifted.csv", index=False
    )
    generate_scenario("schema_change", 5000, 3000, seed=7).current.to_csv(
        OUT / "current_customers_schema_change.csv", index=False
    )
    generate_scenario("stable", 5000, 50, seed=8).current.head(25).to_csv(
        OUT / "current_customers_tiny.csv", index=False
    )
    for path in sorted(OUT.glob("*.csv")):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
