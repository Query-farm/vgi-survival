"""Generate invented device follow-up records without fitting an estimator."""

import csv
import hashlib
import json
from pathlib import Path


def main():
    """Write deterministic follow-up observations and explicit provenance."""
    folder = Path(__file__).resolve().parent / "fixtures"
    failures = [2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 12, 14, 16, 18, 22]
    censored = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    with (folder / "followup.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["device_id", "recorded_days", "recorded_event"])
        for i, (days, event) in enumerate([(t, 1) for t in failures] + [(t, 0) for t in censored], start=1):
            writer.writerow([f"synthetic-{i:02d}", days, event])
    provenance = {
        "provenance": "All 24 device identifiers and follow-up times are invented; not measured devices or patients.",
        "rows": 24,
        "time_unit": "days since a shared synthetic time zero; no calendar dates",
        "event_coding": "1 = observed failure, 0 = right-censored; follow-up end is not failure",
        "recorded_failure_days": failures,
        "recorded_censor_days": censored,
        "sha256": hashlib.sha256((folder / "followup.csv").read_bytes()).hexdigest(),
        "design": (
            "Deliberate ties and censoring illustrate product-limit arithmetic; "
            "no random sampling or population inference."
        ),
    }
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
