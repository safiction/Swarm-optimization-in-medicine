from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_models():
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "random_forest": RandomForestClassifier(random_state=42),

        "catboost": CatBoostClassifier(
            verbose=0,
            random_state=42
        )
    }

    return models

# "svm": Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", SVC(kernel="rbf", probability=True))
#         ]),