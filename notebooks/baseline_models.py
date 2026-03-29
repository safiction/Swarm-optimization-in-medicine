import sys
import os
import pandas as pd

sys.path.append(os.path.abspath("../"))

from src.models import get_models
from src.evaluation import evaluate

# Accessing train and test data
base_path = "../data/processed/"

X_train = pd.read_csv(base_path + "X_train.csv")
X_test = pd.read_csv(base_path + "X_test.csv")
y_train = pd.read_csv(base_path + "y_train.csv")
y_test = pd.read_csv(base_path + "y_test.csv")

# Vectorizing data
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

models = get_models()

results = {}

# training + evaluation
for name, model in models.items():
    print(f"\nTraining: {name}")
    
    model.fit(X_train, y_train)
    results[name] = evaluate(model, X_test, y_test)

# Printing results
for name, res in results.items():
    print(f"\n{name}")
    print("Accuracy:", res["accuracy"])
    print("F1 (macro):", res["f1_macro"])

df_results = pd.DataFrame(results).T
df_results = df_results[["accuracy", "f1_macro"]]

df_results.sort_values(by="f1_macro", ascending=False)