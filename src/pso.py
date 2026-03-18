# draft, preparation for actual algorithm implementation
import numpy as np

def fitness_function(selected_features, X, y, model):
    if np.sum(selected_features) == 0:
        return 1.0
    
    X_selected = X[:, selected_features == 1]
    
    model.fit(X_selected, y)
    score = model.score(X_selected, y)
    
    penalty = np.sum(selected_features) / len(selected_features)
    
    return 1 - score + 0.1 * penalty