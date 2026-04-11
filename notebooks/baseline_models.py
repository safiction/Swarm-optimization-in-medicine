import sys
import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils import resample

sys.path.append(os.path.abspath("../"))

from src.pso_algorithm import PSOFeatureSelector
from src.models import get_models
from src.evaluation import evaluate
from src.pso_algorithm import PSOFeatureSelector

# Accessing train and test data
base_path = "../data/processed/"

X_train = pd.read_csv(base_path + "X_train.csv")
X_test = pd.read_csv(base_path + "X_test.csv")
y_train = pd.read_csv(base_path + "y_train.csv")
y_test = pd.read_csv(base_path + "y_test.csv")

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

models = get_models()

results = {}

# Training + evaluation
for name, model in models.items():
    print(f"\nTraining: {name}")
    
    model.fit(X_train, y_train)
    results[name] = evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

df_results = pd.DataFrame(results).T
df_results = df_results[["accuracy", "f1"]]

df_results.sort_values(by="f1", ascending=False)

print("\nRunning PSO...")

X_sample, y_sample = resample(
    X_train,
    y_train,
    n_samples=20000,
    random_state=42,
    stratify=y_train
)

pso = PSOFeatureSelector(
    n_particles=15,
    n_iterations=15,
    random_state=42
)
    

pso.fit(X_sample, y_sample)

selected_features = X_train.columns[pso.best_features_]

print("\nSelected features:")
print(selected_features.tolist())
output_dir = "../results"
os.makedirs(output_dir, exist_ok=True)
pd.Series(selected_features).to_csv(
    "../results/selected_features.csv",
    index=False,
    header=False
)

X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]


print("\n=== AFTER PSO ===")

results_pso = {}

for name, model in models.items():
    print(f"\nTraining (PSO): {name}")
    
    model.fit(X_train_reduced, y_train)
    results_pso[name] = evaluate(model, X_test_reduced, y_test)
    y_pred = model.predict(X_test_reduced)
    print(classification_report(y_test, y_pred))

df_before = pd.DataFrame(results).T
df_after = pd.DataFrame(results_pso).T

comparison = df_before[["accuracy", "f1"]].copy()
comparison.columns = ["acc_before", "f1_before"]

comparison["acc_after"] = df_after["accuracy"]
comparison["f1_after"] = df_after["f1"]

print(comparison)
