# Turns the Locust CSV files in results/ into a markdown table for the README.
# Expected file names: results/containers_1_stats.csv, results/containers_2_stats.csv, ...
import glob
import re

import pandas as pd

rows = []
for path in sorted(glob.glob("results/containers_*_stats.csv")):
    n = int(re.search(r"containers_(\d+)_stats", path).group(1))
    df = pd.read_csv(path)
    agg = df[df["Name"] == "Aggregated"].iloc[0]
    rows.append({
        "Containers": n,
        "Requests": int(agg["Request Count"]),
        "Failures": int(agg["Failure Count"]),
        "Requests/s": round(float(agg["Requests/s"]), 2),
        "Avg (ms)": round(float(agg["Average Response Time"]), 1),
        "Median (ms)": round(float(agg["Median Response Time"]), 1),
        "95th pct (ms)": round(float(agg["95%"]), 1),
        "99th pct (ms)": round(float(agg["99%"]), 1),
    })

table = pd.DataFrame(rows).sort_values("Containers")
header = "| " + " | ".join(table.columns) + " |"
divider = "| " + " | ".join("---" for _ in table.columns) + " |"
lines = ["| " + " | ".join(str(v) for v in r) + " |" for r in table.values.tolist()]
print("\n".join([header, divider] + lines))
