from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]


@dataclass
class ParticleEvaluation:
    """Result of evaluating one particle."""
    fitness: float
    cv_score: float
    n_selected: int


class BinaryPSOFeatureSelector:
    """
    Binary PSO for feature selection.

    Each particle is a binary vector:
        1 -> keep feature
        0 -> remove feature

    Fitness:
        fitness = alpha * cv_score + (1 - alpha) * sparsity_score

    where:
        cv_score = average cross-validation quality
        sparsity_score = reward for using fewer features
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_particles: int = 10,
        n_iterations: int = 10,
        inertia_weight: float = 0.729,
        cognitive_weight: float = 1.49445,
        social_weight: float = 1.49445,
        max_velocity: float = 6.0,
        alpha: float = 0.99,
        cv_splits: int = 3,
        scoring: str = "f1_weighted",
        sample_size: Optional[int] = 20000,
        random_state: Optional[int] = 42,
        verbose: bool = True,
    ) -> None:
        self.estimator = estimator if estimator is not None else Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                solver="saga",
                random_state=random_state,
            )),
        ])

        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.max_velocity = max_velocity
        self.alpha = alpha
        self.cv_splits = cv_splits
        self.scoring = scoring
        self.sample_size = sample_size
        self.random_state = random_state
        self.verbose = verbose

        self.feature_names_: Optional[np.ndarray] = None
        self.support_mask_: Optional[np.ndarray] = None
        self.selected_features_: Optional[List[str]] = None
        self.best_fitness_: Optional[float] = None
        self.best_cv_score_: Optional[float] = None
        self.best_subset_size_: Optional[int] = None
        self.history_: List[Dict[str, float]] = []
        self.best_estimator_: Optional[BaseEstimator] = None

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        """Convert velocities into probabilities."""
        values = np.clip(values, -50, 50)
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _to_numpy(
        X: ArrayLike,
        y: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Convert pandas objects to numpy arrays."""
        feature_names = X.columns.to_numpy() if isinstance(X, pd.DataFrame) else None
        X_array = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
        y_array = y.to_numpy().ravel() if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y).ravel()
        return X_array, y_array, feature_names

    def _build_cv(self) -> StratifiedKFold:
        """Create stratified CV splitter."""
        return StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )

    def _ensure_non_empty_particle(
        self,
        particle: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Ensure particle selects at least one feature."""
        if particle.sum() == 0:
            random_index = rng.integers(0, len(particle))
            particle[random_index] = 1
        return particle

    def _sample_for_evaluation(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use a smaller subset for particle evaluation if dataset is very large.
        This makes PSO much faster.
        """
        if self.sample_size is None or len(X) <= self.sample_size:
            return X, y

        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), size=self.sample_size, replace=False)
        return X[indices], y[indices]

    def _evaluate_particle(
        self,
        particle: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ParticleEvaluation:
        """Evaluate one particle using CV score + sparsity reward."""
        selected_idx = particle.astype(bool)
        X_selected = X[:, selected_idx]

        X_eval, y_eval = self._sample_for_evaluation(X_selected, y)

        model = clone(self.estimator)
        cv_scores = cross_val_score(
            model,
            X_eval,
            y_eval,
            cv=self._build_cv(),
            scoring=self.scoring,
            n_jobs=-1,
        )

        cv_score = float(np.mean(cv_scores))
        n_selected = int(selected_idx.sum())
        sparsity_score = 1.0 - (n_selected / X.shape[1])
        fitness = self.alpha * cv_score + (1.0 - self.alpha) * sparsity_score

        return ParticleEvaluation(
            fitness=fitness,
            cv_score=cv_score,
            n_selected=n_selected,
        )

    def _initialize_swarm(self, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize particles and velocities."""
        rng = np.random.default_rng(self.random_state)

        positions = rng.integers(0, 2, size=(self.n_particles, n_features))
        velocities = rng.uniform(-1.0, 1.0, size=(self.n_particles, n_features))

        for i in range(self.n_particles):
            positions[i] = self._ensure_non_empty_particle(positions[i], rng)

        return positions.astype(int), velocities.astype(float)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "BinaryPSOFeatureSelector":
        """Run Binary PSO and store best feature subset."""
        X_array, y_array, feature_names = self._to_numpy(X, y)
        n_samples, n_features = X_array.shape

        if n_features < 1:
            raise ValueError("X must contain at least one feature.")
        if self.n_particles < 1:
            raise ValueError("n_particles must be at least 1.")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be at least 1.")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1.")

        self.feature_names_ = (
            feature_names
            if feature_names is not None
            else np.array([f"feature_{i}" for i in range(n_features)])
        )

        positions, velocities = self._initialize_swarm(n_features)

        personal_best_positions = positions.copy()
        personal_best_fitness = np.full(self.n_particles, -np.inf, dtype=float)
        personal_best_scores = np.zeros(self.n_particles, dtype=float)
        personal_best_sizes = np.zeros(self.n_particles, dtype=int)

        global_best_position = None
        global_best_fitness = -np.inf
        global_best_score = 0.0
        global_best_size = 0

        for i in range(self.n_particles):
            evaluation = self._evaluate_particle(positions[i], X_array, y_array)
            personal_best_fitness[i] = evaluation.fitness
            personal_best_scores[i] = evaluation.cv_score
            personal_best_sizes[i] = evaluation.n_selected

            if evaluation.fitness > global_best_fitness:
                global_best_fitness = evaluation.fitness
                global_best_score = evaluation.cv_score
                global_best_size = evaluation.n_selected
                global_best_position = positions[i].copy()

        if global_best_position is None:
            raise RuntimeError("PSO failed to initialize global best particle.")

        rng = np.random.default_rng(self.random_state)
        self.history_ = []

        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                r1 = rng.random(n_features)
                r2 = rng.random(n_features)

                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_weight * r1 * (personal_best_positions[i] - positions[i])
                    + self.social_weight * r2 * (global_best_position - positions[i])
                )

                velocities[i] = np.clip(
                    velocities[i],
                    -self.max_velocity,
                    self.max_velocity,
                )

                probabilities = self._sigmoid(velocities[i])
                positions[i] = (rng.random(n_features) < probabilities).astype(int)
                positions[i] = self._ensure_non_empty_particle(positions[i], rng)

                evaluation = self._evaluate_particle(positions[i], X_array, y_array)

                if evaluation.fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = evaluation.fitness
                    personal_best_scores[i] = evaluation.cv_score
                    personal_best_sizes[i] = evaluation.n_selected
                    personal_best_positions[i] = positions[i].copy()

                if evaluation.fitness > global_best_fitness:
                    global_best_fitness = evaluation.fitness
                    global_best_score = evaluation.cv_score
                    global_best_size = evaluation.n_selected
                    global_best_position = positions[i].copy()

            self.history_.append(
                {
                    "iteration": float(iteration + 1),
                    "best_fitness": float(global_best_fitness),
                    "best_cv_score": float(global_best_score),
                    "best_subset_size": float(global_best_size),
                    "selected_ratio": float(global_best_size / n_features),
                    "n_samples": float(n_samples),
                }
            )

            if self.verbose:
                print(
                    f"Iteration {iteration + 1:03d}/{self.n_iterations} | "
                    f"fitness={global_best_fitness:.6f} | "
                    f"cv_score={global_best_score:.6f} | "
                    f"selected={global_best_size}/{n_features}"
                )

        self.support_mask_ = global_best_position.astype(bool)
        self.selected_features_ = self.feature_names_[self.support_mask_].tolist()
        self.best_fitness_ = float(global_best_fitness)
        self.best_cv_score_ = float(global_best_score)
        self.best_subset_size_ = int(global_best_size)

        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.fit(X_array[:, self.support_mask_], y_array)

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Keep only selected features."""
        if self.support_mask_ is None:
            raise RuntimeError("Call fit before transform.")

        X_array = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
        return X_array[:, self.support_mask_]

    def fit_transform(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Fit selector and transform X."""
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return mask or indices of selected features."""
        if self.support_mask_ is None:
            raise RuntimeError("Call fit before get_support.")
        return np.where(self.support_mask_)[0] if indices else self.support_mask_

    def get_selected_features(self) -> List[str]:
        """Return selected feature names."""
        if self.selected_features_ is None:
            raise RuntimeError("Call fit before get_selected_features.")
        return self.selected_features_

    def get_history_dataframe(self) -> pd.DataFrame:
        """Return optimization history."""
        return pd.DataFrame(self.history_)

    def evaluate_on_test(self, X_test: ArrayLike, y_test: ArrayLike) -> Dict[str, Any]:
        """Evaluate final model on test set."""
        if self.best_estimator_ is None or self.support_mask_ is None:
            raise RuntimeError("Call fit before evaluate_on_test.")

        X_test_array = X_test.to_numpy() if isinstance(X_test, (pd.DataFrame, pd.Series)) else np.asarray(X_test)
        y_test_array = y_test.to_numpy().ravel() if isinstance(y_test, (pd.DataFrame, pd.Series)) else np.asarray(y_test).ravel()

        scorer = get_scorer(self.scoring)
        test_score = float(scorer(self.best_estimator_, X_test_array[:, self.support_mask_], y_test_array))

        return {
            "test_score": test_score,
            "test_metric": self.scoring,
            "selected_features": self.get_selected_features(),
            "n_selected": int(self.support_mask_.sum()),
            "best_cv_score": self.best_cv_score_,
            "best_fitness": self.best_fitness_,
        }


def load_processed_split(
    data_dir: Union[str, Path] = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load train/test split from CSV files.

    Expected files:
        X_train.csv
        X_test.csv
        y_train.csv
        y_test.csv
    """
    data_dir = Path(data_dir)

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")

    return X_train, X_test, y_train, y_test


def run_pso_pipeline(
    estimator: Optional[ClassifierMixin] = None,
    data_dir: Union[str, Path] = "data/processed",
    n_particles: int = 10,
    n_iterations: int = 10,
    alpha: float = 0.99,
    scoring: str = "f1_weighted",
    sample_size: Optional[int] = 20000,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    1. Load processed train/test data.
    2. Run Binary PSO on train set.
    3. Evaluate selected subset on test set.
    """
    X_train, X_test, y_train, y_test = load_processed_split(data_dir)

    selector = BinaryPSOFeatureSelector(
        estimator=estimator,
        n_particles=n_particles,
        n_iterations=n_iterations,
        alpha=alpha,
        scoring=scoring,
        sample_size=sample_size,
        random_state=random_state,
        verbose=verbose,
    )

    selector.fit(X_train, y_train)
    test_results = selector.evaluate_on_test(X_test, y_test)

    return {
        "selector": selector,
        "selected_features": selector.get_selected_features(),
        "history": selector.get_history_dataframe(),
        "test_results": test_results,
    }


if __name__ == "__main__":
    results = run_pso_pipeline(verbose=True)

    print("\nSelected features:")
    for feature in results["selected_features"]:
        print(f"- {feature}")

    print("\nTest results:")
    for key, value in results["test_results"].items():
        print(f"{key}: {value}")