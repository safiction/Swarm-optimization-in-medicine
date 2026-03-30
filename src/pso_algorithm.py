import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class PSOFeatureSelector:
    def __init__(self, n_particles=20, n_iterations=20, alpha=0.9, beta=0.1):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta

    def _fitness(self, particle, X, y):
        if np.sum(particle) == 0:
            return 1.0

        selected = particle.astype(bool)
        X_selected = X[:, selected]

        model = RandomForestClassifier(random_state=42)

        scores = cross_val_score(model, X_selected, y, cv=3, scoring="f1_macro")

        score = scores.mean()

        penalty = np.sum(selected) / X.shape[1]

        return 1 - (self.alpha * score - self.beta * penalty)

    def fit(self, X, y):
        X = X.values if hasattr(X, "values") else X

        n_features = X.shape[1]

        particles = np.random.randint(0, 2, (self.n_particles, n_features))
        velocities = np.random.rand(self.n_particles, n_features)

        personal_best = particles.copy()
        personal_best_scores = np.array(
            [self._fitness(p, X, y) for p in particles]
        )

        global_best = personal_best[np.argmin(personal_best_scores)]

        for _ in range(self.n_iterations):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()

                velocities[i] = (
                    0.5 * velocities[i]
                    + r1 * (personal_best[i] - particles[i])
                    + r2 * (global_best - particles[i])
                )

                sigmoid = 1 / (1 + np.exp(-velocities[i]))
                particles[i] = (np.random.rand(n_features) < sigmoid).astype(int)

                score = self._fitness(particles[i], X, y)

                if score < personal_best_scores[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_scores[i] = score

            global_best = personal_best[np.argmin(personal_best_scores)]

        self.best_features_ = global_best.astype(bool)
        return self

    def transform(self, X):
        return X.loc[:, self.best_features_]
        
selected_features = X_train.columns[pso.best_features_]

selected_features.to_series().to_csv(
    "../results/selected_features.csv",
    index=False
)
