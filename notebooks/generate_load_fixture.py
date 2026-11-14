"""Create synthetic device follow-up without computing any fitted answers."""

import csv
import hashlib
import json
import math
import random
from pathlib import Path

folder = Path(__file__).resolve().parent / "fixtures"
rng = random.Random(20261114)
with (folder / "load_followup.csv").open("w", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["device_id", "load_pct", "duration_days", "event"])
    for i in range(80):
        load = 20 + (i % 13) * 5
        rate = (1 / 35) * math.exp(math.log(1.3) * (load - 50) / 10)
        event_time = -math.log1p(-rng.random()) / rate
        stop = 12 + 48 * rng.random()
        writer.writerow([f"sim-{i + 1:03}", load, f"{min(event_time, stop):.8f}", int(event_time <= stop)])
manifest = {
    "provenance": "80 simulated devices, no real measured or clinical records",
    "seed": 20261114,
    "rows": 80,
    "algorithm": (
        "Python Random; load cycles20..80 in5-point steps; exponential event time "
        "rate=(1/35)*exp(log(1.3)*(load-50)/10); independently drawn stop=12+48*U days. "
        "duration=min(event,stop), event=1 iff event<=stop."
    ),
    "time_unit": "days",
    "load_unit": "percentage points",
    "sha256": hashlib.sha256((folder / "load_followup.csv").read_bytes()).hexdigest(),
    "scope": (
        "Illustrates estimator interfaces and exact covariate-unit rescaling; "
        "no population inference or model adequacy claim."
    ),
}
(folder / "load_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
