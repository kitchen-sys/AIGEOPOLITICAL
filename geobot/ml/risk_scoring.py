"""
Risk scoring using gradient boosting and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score


class RiskScorer:
    """
    Nonlinear risk scoring using ensemble methods.

    Uses Gradient Boosting and Random Forests to discover nonlinear
    patterns in geopolitical risk factors.
    """

    def __init__(self, method: str = 'gradient_boosting'):
        """
        Initialize risk scorer.

        Parameters
        ----------
        method : str
            Method to use ('gradient_boosting', 'random_forest', 'ensemble')
        """
        self.method = method
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 5
    ) -> Dict[str, Any]:
        """
        Train risk scoring model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Risk labels (0 = low risk, 1 = high risk)
        n_estimators : int
            Number of estimators
        max_depth : int
            Maximum tree depth

        Returns
        -------
        dict
            Training results
        """
        self.feature_names = X.columns.tolist()

        if self.method == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        elif self.method == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Train
        self.model.fit(X, y)
        self.is_trained = True

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5)

        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': self.get_feature_importance()
        }

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores.

        Parameters
        ----------
        X : pd.DataFrame
            Features

        Returns
        -------
        np.ndarray
            Risk probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns
        -------
        dict
            Feature importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def explain_prediction(self, X: pd.DataFrame, index: int) -> Dict[str, Any]:
        """
        Explain a specific prediction.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        index : int
            Index of sample to explain

        Returns
        -------
        dict
            Explanation
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        sample = X.iloc[index:index+1]
        risk_score = self.predict_risk(sample)[0]

        # Feature contributions (simplified)
        feature_values = sample.iloc[0].to_dict()
        feature_importance = self.get_feature_importance()

        contributions = {
            feat: feature_values[feat] * feature_importance[feat]
            for feat in feature_values.keys()
        }

        return {
            'risk_score': risk_score,
            'top_risk_factors': sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:5],
            'feature_values': feature_values
        }
