from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_models():
    # define a set of baseline models for comparison
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),

        # tree-based model (handles non-linearity and interactions)
        "random_forest": RandomForestClassifier(
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        # gradient boosting model
        "catboost": CatBoostClassifier(
            verbose=0,
            auto_class_weights="Balanced",
            random_state=42
        )
    }
    return models