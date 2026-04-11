from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

def evaluate(model, X_test, y_test):
    # generate predictions on test set
    y_pred = model.predict(X_test)

    # compute main classification metrics (binary setting)
    return {
        "accuracy": accuracy_score(y_test, y_pred),

        # F1 focuses on balance between precision and recall
        "f1": f1_score(y_test, y_pred, pos_label=1),

        # precision = how many predicted positives are correct
        "precision": precision_score(y_test, y_pred, pos_label=1),

        # recall = how many actual positives are captured
        "recall": recall_score(y_test, y_pred, pos_label=1),

        # per-class report (used mainly for printing)
        "report": classification_report(y_test, y_pred)
    }