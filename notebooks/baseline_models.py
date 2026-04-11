import sys
import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils import resample

sys.path.append(os.path.abspath("../"))

from src.pso_algorithm import PSOFeatureSelector
from src.models import get_models
from src.evaluation import evaluate

# path to preprocessed train/test splits
base_path = "../data/processed/"

# load data
X_train = pd.read_csv(base_path + "X_train.csv")
X_test = pd.read_csv(base_path + "X_test.csv")
y_train = pd.read_csv(base_path + "y_train.csv")
y_test = pd.read_csv(base_path + "y_test.csv")

# convert targets to 1D arrays (required by sklearn)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# initialize models
models = get_models()
results = {}

# BASELINE
for name, model in models.items():
    print(f"\nTraining: {name}")
    
    model.fit(X_train, y_train)
    results[name] = evaluate(model, X_test, y_test)

    # detailed per-class metrics
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# collect results into table
df_results = pd.DataFrame(results).T
df_results = df_results[["accuracy", "f1"]]

df_results.sort_values(by="f1", ascending=False)

# PSO
print("\nRunning PSO...")

# subsampling to speed up PSO (full dataset is large)
X_sample, y_sample = resample(
    X_train,
    y_train,
    n_samples=20000,
    random_state=42,
    stratify=y_train
)

# PSO configuration (trade-off between speed and quality)
pso = PSOFeatureSelector(
    n_particles=15,
    n_iterations=15,
    random_state=42
)

# running feature selection
pso.fit(X_sample, y_sample)

# extract selected feature names
selected_features = X_train.columns[pso.best_features_]

print("\nSelected features:")
print(selected_features.tolist())

# save selected features for reproducibility
output_dir = "../results"
os.makedirs(output_dir, exist_ok=True)

pd.Series(selected_features).to_csv(
    "../results/selected_features.csv",
    index=False,
    header=False
)

# reduce dataset to selected features only
X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]

# MODELS AFTER PSO
print("\n=== AFTER PSO ===")

results_pso = {}

for name, model in models.items():
    print(f"\nTraining (PSO): {name}")
    
    model.fit(X_train_reduced, y_train)
    results_pso[name] = evaluate(model, X_test_reduced, y_test)

    # classification report after feature selection
    y_pred = model.predict(X_test_reduced)
    print(classification_report(y_test, y_pred))

# COMPARISON
df_before = pd.DataFrame(results).T
df_after = pd.DataFrame(results_pso).T

# compare performance before and after PSO
comparison = df_before[["accuracy", "f1"]].copy()
comparison.columns = ["acc_before", "f1_before"]

comparison["acc_after"] = df_after["accuracy"]
comparison["f1_after"] = df_after["f1"]

print(comparison)