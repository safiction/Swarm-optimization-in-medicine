from sklearn.metrics import accuracy_score, classification_report, f1_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred)
    }