from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier()
    }