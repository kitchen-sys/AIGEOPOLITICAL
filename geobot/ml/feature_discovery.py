"""
Feature discovery using Random Forests and other methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA


class FeatureDiscovery:
    """
    Discover important features and relationships in geopolitical data.
    """

    def __init__(self):
        """Initialize feature discovery."""
        self.feature_scores = {}

    def discover_important_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_top: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Discover most important features using Random Forest.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Target variable
        n_top : int
            Number of top features to return

        Returns
        -------
        list
            List of (feature_name, importance_score) tuples
        """
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importance = model.feature_importances_
        self.feature_scores = dict(zip(X.columns, importance))

        sorted_features = sorted(
            self.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_features[:n_top]

    def discover_latent_factors(
        self,
        X: pd.DataFrame,
        n_components: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discover latent factors using PCA.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        n_components : int
            Number of components

        Returns
        -------
        tuple
            (transformed_data, explained_variance)
        """
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(X)

        return transformed, pca.explained_variance_ratio_
