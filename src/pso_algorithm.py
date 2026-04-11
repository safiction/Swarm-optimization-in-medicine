import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class PSOFeatureSelector:
    def __init__(self, n_particles=5, n_iterations=5, alpha=0.9, beta=0.05, random_state=None):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state

    def _fitness(self, particle, X, y):
        if np.sum(particle) == 0:
            return 1.0

        selected = particle.astype(bool)
        X_selected = X[:, selected]

        model = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        scores = cross_val_score(model, X_selected, y, cv=3, scoring="f1", n_jobs=-1)

        score = scores.mean()

        penalty = np.sum(selected) / X.shape[1]

        return 1 - (self.alpha * score - self.beta * penalty)

    def fit(self, X, y):
        X = X.values if hasattr(X, "values") else X
        n_features = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        particles = rng.integers(0, 2, size=(self.n_particles, n_features))
        velocities = rng.random((self.n_particles, n_features))

        personal_best = particles.copy()
        personal_best_scores = np.array(
            [self._fitness(p, X, y) for p in particles]
        )

        global_best = personal_best[np.argmin(personal_best_scores)]

        for _ in range(self.n_iterations):
            for i in range(self.n_particles):
                r1, r2 = rng.random(), rng.random()

                velocities[i] = (
                    0.5 * velocities[i]
                    + r1 * (personal_best[i] - particles[i])
                    + r2 * (global_best - particles[i])
                )

                sigmoid = 1 / (1 + np.exp(-velocities[i]))
                particles[i] = (rng.random(n_features) < sigmoid).astype(int)

                score = self._fitness(particles[i], X, y)

                if score < personal_best_scores[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_scores[i] = score

            global_best = personal_best[np.argmin(personal_best_scores)]

        self.best_features_ = global_best.astype(bool)
        return self

    def transform(self, X):
        return X.loc[:, self.best_features_]
    
