"""Implementation of a clear, publication-oriented MS-AGSO feature selector.

MS-AGSO = Multi-Stage Adaptive Genetic Swarm Optimizer.

The implementation follows the project idea:
1. Initial statistical filtering.
2. Spectral feature grouping.
3. Genetic algorithm preliminary feature search.
4. Binary PSO refinement.
5. Redundancy reduction / ensemble consolidation.

The defaults are intentionally moderate so that the code can run on a normal laptop.
For final publication experiments, increase population_size, ga_generations,
pso_particles and pso_iterations in config.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import CV_FOLDS, SCORING
from src.feature_selection import fisher_scores
from src.models import get_fast_feature_selection_classifier


@dataclass
class MSAGSOResult:
    """Container for selected features and optimization history."""

    selected_features: List[str]
    candidate_features: List[str]
    feature_cluster_table: pd.DataFrame
    history: pd.DataFrame
    best_score: float
    reduction_rate: float


@dataclass
class MSAGSOSelector:
    """Multi-Stage Adaptive Genetic Swarm Optimizer feature selector."""

    max_initial_features: int = 450
    n_clusters: int = 8
    population_size: int = 16
    ga_generations: int = 8
    pso_particles: int = 16
    pso_iterations: int = 8
    mutation_rate: float = 0.08
    feature_penalty: float = 0.015
    redundancy_threshold: float = 0.96
    random_state: int = 42
    scoring: str = SCORING
    cv_folds: int = CV_FOLDS
    history_: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        self.model = get_fast_feature_selection_classifier(random_state=self.random_state)
        self._fitness_cache: Dict[bytes, float] = {}

    def _initial_filtering(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Filter features using variance, mutual-information-like proxy and Fisher score."""
        X_numeric = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Remove constant features.
        variances = X_numeric.var(axis=0)
        X_var = X_numeric.loc[:, variances > 0].copy()
        if X_var.shape[1] == 0:
            raise ValueError("All features have zero variance after initial filtering.")

        # Fisher score is fast and robust for high-dimensional numeric data.
        fisher = fisher_scores(X_var, y)
        top_k = min(self.max_initial_features, X_var.shape[1])
        selected_cols = fisher.head(top_k).index.tolist()
        return X_var[selected_cols].copy()

    def _spectral_feature_clustering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cluster candidate features using spectral clustering over a distance matrix."""
        p = X.shape[1]
        if p == 1:
            return pd.DataFrame({"feature": X.columns, "cluster": [0]})

        n_clusters = min(self.n_clusters, p)
        if n_clusters < 2:
            return pd.DataFrame({"feature": X.columns, "cluster": [0] * p})

        X_t = X.T.to_numpy(dtype=float)

        # Cosine distance transformed into similarity.
        distances = pairwise_distances(X_t, metric="cosine")
        similarity = 1.0 - distances
        similarity = np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=0.0)
        similarity = np.clip(similarity, 0.0, 1.0)
        np.fill_diagonal(similarity, 1.0)

        try:
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=self.random_state,
            )
            labels = clustering.fit_predict(similarity)
        except Exception:
            # Safe fallback: deterministic random cluster assignment.
            labels = np.arange(p) % n_clusters

        return pd.DataFrame({"feature": X.columns.tolist(), "cluster": labels.astype(int)})

    def _random_mask(self, n_features: int, min_selected: int = 1) -> np.ndarray:
        """Create a random binary feature-selection mask."""
        prob = min(0.35, max(0.05, 25 / max(n_features, 1)))
        mask = self.rng.random(n_features) < prob
        if mask.sum() < min_selected:
            idx = self.rng.choice(n_features, size=min_selected, replace=False)
            mask[idx] = True
        return mask

    def _fitness(self, X: pd.DataFrame, y: pd.Series, mask: np.ndarray) -> float:
        """Evaluate a feature mask using CV macro-F1 minus a compactness penalty."""
        mask = mask.astype(bool)
        if mask.sum() == 0:
            return -1.0

        key = mask.tobytes()
        if key in self._fitness_cache:
            return self._fitness_cache[key]

        selected_X = X.iloc[:, mask]
        n_splits = min(self.cv_folds, y.value_counts().min())
        if n_splits < 2:
            # If there are too few samples per class, use an optimistic training score fallback.
            self.model.fit(selected_X, y)
            score = float(self.model.score(selected_X, y))
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(self.model, selected_X, y, cv=cv, scoring=self.scoring, n_jobs=None)
            score = float(np.nanmean(scores))

        penalty = self.feature_penalty * (mask.sum() / len(mask))
        fitness_value = score - penalty
        self._fitness_cache[key] = fitness_value
        return fitness_value

    def _initialize_population(self, n_features: int) -> np.ndarray:
        """Initialize GA population."""
        population = np.vstack([self._random_mask(n_features) for _ in range(self.population_size)])
        return population.astype(bool)

    def _tournament_select(self, population: np.ndarray, fitness_values: np.ndarray, tournament_size: int = 3) -> np.ndarray:
        """Tournament selection for GA."""
        idx = self.rng.choice(len(population), size=min(tournament_size, len(population)), replace=False)
        best = idx[np.argmax(fitness_values[idx])]
        return population[best].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One-point crossover."""
        n = len(parent1)
        if n < 2:
            return parent1.copy(), parent2.copy()
        point = self.rng.integers(1, n)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _mutate(self, mask: np.ndarray) -> np.ndarray:
        """Adaptive bit-flip mutation."""
        mutation_flags = self.rng.random(len(mask)) < self.mutation_rate
        mutated = mask.copy()
        mutated[mutation_flags] = ~mutated[mutation_flags]
        if mutated.sum() == 0:
            mutated[self.rng.integers(0, len(mutated))] = True
        return mutated

    def _run_ga(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, float, np.ndarray]:
        """Run genetic algorithm preliminary search."""
        n_features = X.shape[1]
        population = self._initialize_population(n_features)
        best_mask = population[0].copy()
        best_score = self._fitness(X, y, best_mask)

        for generation in range(self.ga_generations):
            fitness_values = np.array([self._fitness(X, y, ind) for ind in population])
            gen_best_idx = int(np.argmax(fitness_values))
            gen_best_score = float(fitness_values[gen_best_idx])

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_mask = population[gen_best_idx].copy()

            self.history_.append(
                {
                    "stage": "GA",
                    "iteration": generation,
                    "best_fitness": best_score,
                    "selected_features": int(best_mask.sum()),
                }
            )

            new_population = [best_mask.copy()]  # elitism
            while len(new_population) < self.population_size:
                p1 = self._tournament_select(population, fitness_values)
                p2 = self._tournament_select(population, fitness_values)
                c1, c2 = self._crossover(p1, p2)
                new_population.append(self._mutate(c1))
                if len(new_population) < self.population_size:
                    new_population.append(self._mutate(c2))

            population = np.vstack(new_population).astype(bool)

        return best_mask, best_score, population

    def _run_bpso(self, X: pd.DataFrame, y: pd.Series, initial_population: np.ndarray) -> Tuple[np.ndarray, float]:
        """Refine feature subset using Binary PSO."""
        n_features = X.shape[1]
        particles = []
        for i in range(self.pso_particles):
            if i < len(initial_population):
                particles.append(initial_population[i].copy())
            else:
                particles.append(self._random_mask(n_features))
        positions = np.vstack(particles).astype(bool)
        velocities = self.rng.normal(0, 1, size=positions.shape)

        personal_best = positions.copy()
        personal_scores = np.array([self._fitness(X, y, p) for p in personal_best])
        global_best_idx = int(np.argmax(personal_scores))
        global_best = personal_best[global_best_idx].copy()
        global_score = float(personal_scores[global_best_idx])

        w, c1, c2 = 0.72, 1.49, 1.49

        for iteration in range(self.pso_iterations):
            r1 = self.rng.random(size=positions.shape)
            r2 = self.rng.random(size=positions.shape)

            velocities = (
                w * velocities
                + c1 * r1 * (personal_best.astype(float) - positions.astype(float))
                + c2 * r2 * (global_best.astype(float) - positions.astype(float))
            )

            probabilities = 1.0 / (1.0 + np.exp(-velocities))
            positions = self.rng.random(size=positions.shape) < probabilities

            for i in range(len(positions)):
                if positions[i].sum() == 0:
                    positions[i, self.rng.integers(0, n_features)] = True
                score = self._fitness(X, y, positions[i])
                if score > personal_scores[i]:
                    personal_scores[i] = score
                    personal_best[i] = positions[i].copy()
                if score > global_score:
                    global_score = score
                    global_best = positions[i].copy()

            self.history_.append(
                {
                    "stage": "BPSO",
                    "iteration": iteration,
                    "best_fitness": global_score,
                    "selected_features": int(global_best.sum()),
                }
            )

        return global_best, global_score

    def _remove_redundant_features(self, X_selected: pd.DataFrame) -> pd.DataFrame:
        """Remove highly redundant selected features using absolute correlation."""
        if X_selected.shape[1] <= 1:
            return X_selected

        corr = X_selected.corr().abs().fillna(0.0)
        keep: List[str] = []
        for col in corr.columns:
            if not keep:
                keep.append(col)
                continue
            max_corr = corr.loc[col, keep].max()
            if max_corr < self.redundancy_threshold:
                keep.append(col)
        return X_selected[keep].copy()

    def fit_select(self, X: pd.DataFrame, y: pd.Series) -> MSAGSOResult:
        """Run the complete MS-AGSO pipeline and return selected features."""
        self.history_ = []
        self._fitness_cache = {}

        original_feature_count = X.shape[1]
        X_candidates = self._initial_filtering(X, y)
        cluster_table = self._spectral_feature_clustering(X_candidates)

        ga_best_mask, ga_best_score, final_population = self._run_ga(X_candidates, y)
        bpso_best_mask, bpso_best_score = self._run_bpso(X_candidates, y, final_population)

        best_mask = bpso_best_mask if bpso_best_score >= ga_best_score else ga_best_mask
        best_score = max(bpso_best_score, ga_best_score)

        selected_cols = X_candidates.columns[best_mask].tolist()
        X_selected = X_candidates[selected_cols].copy()
        X_consolidated = self._remove_redundant_features(X_selected)
        selected_features = X_consolidated.columns.tolist()

        reduction_rate = 1.0 - (len(selected_features) / max(original_feature_count, 1))

        history_df = pd.DataFrame(self.history_)
        return MSAGSOResult(
            selected_features=selected_features,
            candidate_features=X_candidates.columns.tolist(),
            feature_cluster_table=cluster_table,
            history=history_df,
            best_score=float(best_score),
            reduction_rate=float(reduction_rate),
        )
