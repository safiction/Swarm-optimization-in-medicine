import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pso import run_pso_pipeline

results = run_pso_pipeline(
    data_dir=PROJECT_ROOT / "data" / "processed",
    n_particles=10,
    n_iterations=20,
    alpha=0.99,
    scoring="f1_weighted",
    sample_size=20000,
    random_state=42,
    verbose=True,
)

print("\nSelected features:")
for feature in results["selected_features"]:
    print(f"- {feature}")

print("\nTest results:")
for key, value in results["test_results"].items():
    print(f"{key}: {value}")


results_dir = PROJECT_ROOT / "results" / "metrics"
results_dir.mkdir(exist_ok=True)

pd.DataFrame({
    "selected_feature": results["selected_features"]
}).to_csv(results_dir / "selected_features.csv", index=False)

results["history"].to_csv(results_dir / "pso_history.csv", index=False)

pd.DataFrame([results["test_results"]]).to_csv(results_dir / "test_results.csv", index=False)