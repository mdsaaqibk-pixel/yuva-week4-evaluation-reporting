"""
YUVA Week 4 — E-Governance Reporting Analysis
IMPORTANT: The included dataset is synthetic/illustrative and is not official government data.
Run from this folder: python week4_reporting_analysis.py
Dependencies: pandas, matplotlib
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "simulated_egovernance_data_with_kpis.csv"
OUT = HERE / "analysis_outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
required = ["state", "applications_received", "applications_disposed",
            "pending_applications", "rejected_applications",
            "avg_processing_days", "median_processing_days",
            "citizen_satisfaction_score"]
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing columns: {missing}")

count_cols = ["applications_received", "applications_disposed",
              "pending_applications", "rejected_applications"]
if df[count_cols].isna().any().any():
    print("WARNING: missing application counts; inspect before analysis.")
if (df[count_cols] < 0).any().any():
    raise ValueError("Application counts cannot be negative.")
if not df["citizen_satisfaction_score"].between(0, 1).all():
    raise ValueError("Satisfaction score must be in the 0–1 range.")

df["disposal_rate"] = df["applications_disposed"] / df["applications_received"].replace(0, float("nan"))
df["pending_share_of_workload"] = df["pending_applications"] / (
    df["applications_disposed"] + df["pending_applications"]
).replace(0, float("nan"))
df["rejection_rate"] = df["rejected_applications"] / df["applications_received"].replace(0, float("nan"))
df["received_minus_disposed"] = df["applications_received"] - df["applications_disposed"]

df.to_csv(OUT / "state_kpis.csv", index=False)
df.select_dtypes(include="number").describe().T.to_csv(OUT / "summary_statistics.csv")

plt.figure(figsize=(10, 6))
p = df.sort_values("pending_applications", ascending=True)
plt.barh(p["state"], p["pending_applications"])
plt.xlabel("Pending applications (simulated)")
plt.title("Pending Applications by State — Illustrative Dataset")
plt.tight_layout()
plt.savefig(OUT / "pending_applications_by_state.png", dpi=180)
plt.close()

plt.figure(figsize=(9, 6))
plt.scatter(df["avg_processing_days"], df["citizen_satisfaction_score"])
for _, r in df.iterrows():
    plt.annotate(r["state"], (r["avg_processing_days"], r["citizen_satisfaction_score"]),
                 xytext=(4, 3), textcoords="offset points", fontsize=8)
plt.xlabel("Average processing time (days)")
plt.ylabel("Citizen satisfaction score (0–1)")
plt.title("Processing Time vs Satisfaction — Illustrative Dataset")
plt.tight_layout()
plt.savefig(OUT / "processing_time_vs_satisfaction.png", dpi=180)
plt.close()

corr = df["avg_processing_days"].corr(df["citizen_satisfaction_score"])
(OUT / "interpretation.txt").write_text(
    f"Pearson correlation (processing time vs satisfaction): {corr:.3f}\\n"
    "This is an association in a small synthetic dataset, not proof of causation.\\n",
    encoding="utf-8"
)
print("Analysis complete. Outputs saved in:", OUT)
