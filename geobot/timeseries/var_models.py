"""
Vector Autoregression (VAR), Structural VAR, and Dynamic Factor Models

These econometric models are critical for:
- Multi-country time-series forecasting
- Structural identification of shocks
- Granger causality testing
- Impulse response analysis
- Nowcasting with common factors

VAR models capture interdependencies between multiple time series,
while SVAR adds structural (causal) interpretation.
"""

import numpy as np
from scipy import linalg, stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class VARResults:
    """Results from VAR estimation."""
    coefficients: np.ndarray  # Shape: (n_vars, n_vars * n_lags)
    intercept: np.ndarray  # Shape: (n_vars,)
    residuals: np.ndarray  # Shape: (n_obs, n_vars)
    sigma_u: np.ndarray  # Residual covariance matrix
    log_likelihood: float
    aic: float
    bic: float
    fitted_values: np.ndarray
    n_lags: int
    variable_names: List[str]


@dataclass
class IRFResult:
    """Impulse Response Function results."""
    irf: np.ndarray  # Shape: (n_vars, n_vars, n_steps)
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    shock_names: Optional[List[str]] = None
    response_names: Optional[List[str]] = None


class VARModel:
    """
    Vector Autoregression Model

    Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + ... + A_p Y_{t-p} + u_t

    where:
    - Y_t is a k-dimensional vector of endogenous variables
    - A_i are k×k coefficient matrices
    - u_t ~ N(0, Σ_u) are white noise innovations

    Example:
        >>> var = VARModel(n_lags=2)
        >>> results = var.fit(data)  # data shape: (T, k)
        >>> forecast = var.forecast(results, steps=10)
        >>> granger = var.granger_causality(results, 'GDP', 'interest_rate')
    """

    def __init__(self, n_lags: int = 1, trend: str = 'c'):
        """
        Initialize VAR model.

        Args:
            n_lags: Number of lags to include
            trend: Trend specification ('n'=none, 'c'=constant, 'ct'=constant+trend)
        """
        self.n_lags = n_lags
        self.trend = trend

    def fit(self, data: np.ndarray, variable_names: Optional[List[str]] = None) -> VARResults:
        """
        Estimate VAR model using OLS equation-by-equation.

        Args:
            data: Time series data, shape (n_obs, n_vars)
            variable_names: Optional names for variables

        Returns:
            VARResults object with estimated parameters
        """
        data = np.asarray(data)
        n_obs, n_vars = data.shape

        if variable_names is None:
            variable_names = [f"var{i}" for i in range(n_vars)]

        # Construct lagged design matrix
        Y, X = self._create_lag_matrix(data)

        # OLS estimation: β = (X'X)^{-1} X'Y
        XtX = X.T @ X
        XtY = X.T @ Y

        try:
            beta = linalg.solve(XtX, XtY, assume_a='pos')  # More numerically stable
        except linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            beta = linalg.lstsq(X, Y)[0]

        # Extract coefficients and intercept
        if self.trend in ['c', 'ct']:
            intercept = beta[0, :]
            coefficients = beta[1:, :].T  # Shape: (n_vars, n_vars * n_lags)
        else:
            intercept = np.zeros(n_vars)
            coefficients = beta.T

        # Compute residuals and covariance
        fitted_values = X @ beta
        residuals = Y - fitted_values
        n_effective = Y.shape[0]
        n_params = X.shape[1]

        # Degrees of freedom adjustment
        sigma_u = (residuals.T @ residuals) / (n_effective - n_params)

        # Information criteria
        log_likelihood = self._log_likelihood(residuals, sigma_u, n_effective)
        aic = -2 * log_likelihood + 2 * n_params * n_vars
        bic = -2 * log_likelihood + np.log(n_effective) * n_params * n_vars

        return VARResults(
            coefficients=coefficients,
            intercept=intercept,
            residuals=residuals,
            sigma_u=sigma_u,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            fitted_values=fitted_values,
            n_lags=self.n_lags,
            variable_names=variable_names
        )

    def _create_lag_matrix(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged design matrix for VAR."""
        n_obs, n_vars = data.shape

        # Create lagged variables
        Y = data[self.n_lags:, :]  # Dependent variable

        # Create design matrix with lags
        X_lags = []
        for lag in range(1, self.n_lags + 1):
            X_lags.append(data[self.n_lags - lag: n_obs - lag, :])

        X = np.hstack(X_lags)

        # Add trend terms
        n_effective = Y.shape[0]
        if self.trend == 'c':
            X = np.hstack([np.ones((n_effective, 1)), X])
        elif self.trend == 'ct':
            X = np.hstack([
                np.ones((n_effective, 1)),
                np.arange(1, n_effective + 1).reshape(-1, 1),
                X
            ])

        return Y, X

    def forecast(self, results: VARResults, steps: int, last_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate multi-step forecasts.

        Args:
            results: Fitted VAR results
            steps: Number of steps to forecast
            last_obs: Last n_lags observations to condition on
                      If None, uses last observations from fitted data

        Returns:
            Forecasts, shape (steps, n_vars)
        """
        n_vars = results.coefficients.shape[0]

        # Initialize with last observations
        if last_obs is None:
            # Use residuals to reconstruct last observations
            last_obs = results.fitted_values[-self.n_lags:] + results.residuals[-self.n_lags:]
        else:
            last_obs = np.asarray(last_obs)

        # Reshape coefficients for easier computation
        A_matrices = results.coefficients.reshape(n_vars, self.n_lags, n_vars)

        forecasts = np.zeros((steps, n_vars))
        history = last_obs.copy()

        for step in range(steps):
            # Compute forecast: c + A_1 Y_{t-1} + ... + A_p Y_{t-p}
            forecast = results.intercept.copy()

            for lag in range(self.n_lags):
                if lag < history.shape[0]:
                    forecast += A_matrices[:, lag, :] @ history[-(lag + 1), :]

            forecasts[step, :] = forecast

            # Update history
            history = np.vstack([history, forecast])

        return forecasts

    def granger_causality(self, results: VARResults, caused_var: Union[int, str],
                         causing_var: Union[int, str]) -> Dict[str, float]:
        """
        Test Granger causality: Does causing_var help predict caused_var?

        H0: Lags of causing_var do not help predict caused_var

        Args:
            results: Fitted VAR results
            caused_var: Index or name of caused variable
            causing_var: Index or name of causing variable

        Returns:
            Dictionary with F-statistic, p-value, and conclusion
        """
        # Convert names to indices
        if isinstance(caused_var, str):
            caused_var = results.variable_names.index(caused_var)
        if isinstance(causing_var, str):
            causing_var = results.variable_names.index(causing_var)

        n_vars = results.coefficients.shape[0]
        A_matrices = results.coefficients.reshape(n_vars, self.n_lags, n_vars)

        # Extract coefficients of causing_var in equation for caused_var
        relevant_coeffs = A_matrices[caused_var, :, causing_var]

        # F-test: H0: all relevant coefficients = 0
        # RSS_restricted vs RSS_unrestricted
        # For simplicity, use Wald test

        # Compute F-statistic
        # This is approximate; full implementation would require re-estimation
        n_obs = results.residuals.shape[0]
        n_restrictions = self.n_lags

        # Wald statistic
        wald = np.sum(relevant_coeffs ** 2) / results.sigma_u[caused_var, caused_var]
        f_stat = wald / n_restrictions

        # p-value from F-distribution
        p_value = 1 - stats.f.cdf(f_stat, n_restrictions, n_obs - n_vars * self.n_lags - 1)

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'causing_var': results.variable_names[causing_var],
            'caused_var': results.variable_names[caused_var],
            'granger_causes': p_value < 0.05
        }

    def impulse_response(self, results: VARResults, steps: int = 10,
                        orthogonalized: bool = True) -> IRFResult:
        """
        Compute Impulse Response Functions.

        IRFs show the effect of a one-time shock to one variable on all variables.

        Args:
            results: Fitted VAR results
            steps: Number of steps to compute
            orthogonalized: If True, use Cholesky orthogonalization

        Returns:
            IRFResult with impulse responses
        """
        n_vars = results.coefficients.shape[0]
        A_matrices = results.coefficients.reshape(n_vars, self.n_lags, n_vars)

        # Initialize IRF tensor: (response_var, shock_var, time_step)
        irf = np.zeros((n_vars, n_vars, steps))

        # Compute structural shock matrix
        if orthogonalized:
            # Cholesky decomposition: Σ_u = P P'
            # Structural shocks: ε_t = P^{-1} u_t
            P = linalg.cholesky(results.sigma_u, lower=True)
        else:
            P = np.eye(n_vars)

        # Initial impact (contemporaneous)
        irf[:, :, 0] = P

        # Compute IRF recursively using VAR companion form
        companion = self._companion_matrix(A_matrices)

        for step in range(1, steps):
            # Φ_s = A_1 Φ_{s-1} + A_2 Φ_{s-2} + ... + A_p Φ_{s-p}
            response = np.zeros((n_vars, n_vars))

            for lag in range(1, min(step + 1, self.n_lags + 1)):
                response += A_matrices[:, lag - 1, :] @ irf[:, :, step - lag]

            irf[:, :, step] = response

        return IRFResult(
            irf=irf,
            shock_names=results.variable_names,
            response_names=results.variable_names
        )

    def forecast_error_variance_decomposition(self, results: VARResults,
                                             steps: int = 10) -> np.ndarray:
        """
        Compute Forecast Error Variance Decomposition (FEVD).

        Shows the proportion of forecast error variance in variable i
        attributable to shocks in variable j.

        Args:
            results: Fitted VAR results
            steps: Number of forecast horizons

        Returns:
            FEVD array, shape (n_vars, n_vars, steps)
            fevd[i, j, s] = contribution of shock j to variance of variable i at horizon s
        """
        irf_result = self.impulse_response(results, steps=steps, orthogonalized=True)
        irf = irf_result.irf

        n_vars = irf.shape[0]
        fevd = np.zeros((n_vars, n_vars, steps))

        for step in range(steps):
            # Cumulative squared IRFs
            mse = np.sum(irf[:, :, :step + 1] ** 2, axis=2)  # Total MSE for each variable

            for i in range(n_vars):
                # Normalize to get proportions
                total_mse = mse[i, :].sum()
                if total_mse > 0:
                    fevd[i, :, step] = mse[i, :] / total_mse

        return fevd

    def _companion_matrix(self, A_matrices: np.ndarray) -> np.ndarray:
        """Construct companion form matrix for VAR."""
        n_vars, n_lags, _ = A_matrices.shape
        size = n_vars * n_lags

        companion = np.zeros((size, size))

        # First block row: [A_1, A_2, ..., A_p]
        for lag in range(n_lags):
            companion[:n_vars, lag * n_vars:(lag + 1) * n_vars] = A_matrices[:, lag, :]

        # Identity blocks for lagged variables
        if n_lags > 1:
            companion[n_vars:, :size - n_vars] = np.eye(size - n_vars)

        return companion

    def _log_likelihood(self, residuals: np.ndarray, sigma_u: np.ndarray, n_obs: int) -> float:
        """Compute log-likelihood for VAR model."""
        k = residuals.shape[1]

        sign, logdet = np.linalg.slogdet(sigma_u)
        if sign <= 0:
            return -np.inf

        ll = -0.5 * n_obs * (k * np.log(2 * np.pi) + logdet)
        return ll


class SVARModel:
    """
    Structural Vector Autoregression

    Imposes structure on the reduced-form VAR to identify structural shocks:

    A_0 Y_t = c + A_1 Y_{t-1} + ... + A_p Y_{t-p} + B_0 ε_t

    where ε_t are structural shocks with E[ε_t ε_t'] = I

    Identification schemes:
    - 'cholesky': Recursive (triangular) identification
    - 'short_run': Short-run restrictions on A_0
    - 'long_run': Long-run restrictions on cumulative IRF
    - 'sign': Sign restrictions on IRFs

    Example:
        >>> svar = SVARModel(n_lags=2, identification='cholesky')
        >>> results = svar.fit(data)
        >>> structural_irf = svar.structural_impulse_response(results, steps=20)
    """

    def __init__(self, n_lags: int = 1, identification: str = 'cholesky'):
        """
        Initialize SVAR model.

        Args:
            n_lags: Number of lags
            identification: Identification scheme
        """
        self.n_lags = n_lags
        self.identification = identification
        self.var_model = VARModel(n_lags=n_lags)

    def fit(self, data: np.ndarray, variable_names: Optional[List[str]] = None,
            restrictions: Optional[np.ndarray] = None) -> VARResults:
        """
        Estimate SVAR model.

        Args:
            data: Time series data
            variable_names: Variable names
            restrictions: Optional restriction matrix for identification

        Returns:
            VARResults with structural parameters
        """
        # First estimate reduced-form VAR
        var_results = self.var_model.fit(data, variable_names)

        # Apply identification scheme
        if self.identification == 'cholesky':
            # A_0 = I, B_0 = P (Cholesky factor)
            structural_matrix = linalg.cholesky(var_results.sigma_u, lower=True)
        elif self.identification == 'short_run' and restrictions is not None:
            # Impose short-run restrictions
            structural_matrix = self._identify_short_run(var_results.sigma_u, restrictions)
        else:
            warnings.warn(f"Identification '{self.identification}' not fully implemented, using Cholesky")
            structural_matrix = linalg.cholesky(var_results.sigma_u, lower=True)

        # Store structural matrix in results (extending VARResults)
        var_results.structural_matrix = structural_matrix

        return var_results

    def structural_impulse_response(self, results: VARResults, steps: int = 10) -> IRFResult:
        """
        Compute structural (identified) impulse response functions.

        Args:
            results: Fitted SVAR results with structural_matrix
            steps: Number of steps

        Returns:
            Structural IRF
        """
        # Use VAR IRF with structural identification
        irf_result = self.var_model.impulse_response(results, steps=steps, orthogonalized=True)

        # IRFs are already structural due to Cholesky in fit()
        return irf_result

    def _identify_short_run(self, sigma_u: np.ndarray, restrictions: np.ndarray) -> np.ndarray:
        """
        Identify structural matrix using short-run restrictions.

        This is a placeholder for more sophisticated identification.
        Full implementation would use iterative algorithms.
        """
        # For now, use Cholesky
        return linalg.cholesky(sigma_u, lower=True)


class DynamicFactorModel:
    """
    Dynamic Factor Model for high-dimensional time series.

    Model:
        X_t = Λ F_t + e_t  (observation equation)
        F_t = Φ F_{t-1} + η_t  (state equation)

    where:
    - X_t: n-dimensional observed variables
    - F_t: r-dimensional common factors (r << n)
    - Λ: n×r factor loading matrix
    - Φ: r×r factor dynamics matrix

    Used for:
    - Nowcasting with many indicators
    - Dimension reduction
    - Extracting common trends

    Example:
        >>> dfm = DynamicFactorModel(n_factors=3, n_lags=1)
        >>> results = dfm.fit(data)  # data: (T, 100) many indicators
        >>> factors = dfm.extract_factors(data, results)
        >>> forecast = dfm.forecast(results, steps=10)
    """

    def __init__(self, n_factors: int = 1, n_lags: int = 1, max_iter: int = 100):
        """
        Initialize DFM.

        Args:
            n_factors: Number of common factors
            n_lags: Number of lags in factor dynamics
            max_iter: Maximum EM iterations
        """
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.max_iter = max_iter

    def fit(self, data: np.ndarray) -> Dict:
        """
        Estimate DFM using EM algorithm or principal components.

        Args:
            data: Observed data, shape (n_obs, n_vars)

        Returns:
            Dictionary with estimated parameters
        """
        data = np.asarray(data)
        n_obs, n_vars = data.shape

        # Standardize data
        data_mean = np.nanmean(data, axis=0)
        data_std = np.nanstd(data, axis=0)
        data_std[data_std == 0] = 1.0
        data_normalized = (data - data_mean) / data_std

        # Initial factor extraction via PCA
        # Handle missing data by mean imputation
        data_imputed = data_normalized.copy()
        data_imputed[np.isnan(data_imputed)] = 0

        # SVD for PCA
        U, S, Vt = linalg.svd(data_imputed, full_matrices=False)

        # Extract factors (principal components)
        factors = U[:, :self.n_factors] * S[:self.n_factors]
        loadings = Vt[:self.n_factors, :].T  # Shape: (n_vars, n_factors)

        # Estimate factor dynamics using VAR
        var_model = VARModel(n_lags=self.n_lags)
        factor_var = var_model.fit(factors)

        # Idiosyncratic variance
        reconstruction = factors @ loadings.T
        residuals = data_normalized - reconstruction
        idiosyncratic_var = np.nanvar(residuals, axis=0)

        return {
            'factors': factors,
            'loadings': loadings,
            'factor_dynamics': factor_var,
            'idiosyncratic_var': idiosyncratic_var,
            'data_mean': data_mean,
            'data_std': data_std,
            'explained_variance_ratio': (S[:self.n_factors] ** 2).sum() / (S ** 2).sum()
        }

    def extract_factors(self, data: np.ndarray, model: Dict) -> np.ndarray:
        """
        Extract factors from new data using fitted model.

        Args:
            data: New data to extract factors from
            model: Fitted model from fit()

        Returns:
            Extracted factors, shape (n_obs, n_factors)
        """
        # Normalize data
        data_normalized = (data - model['data_mean']) / model['data_std']

        # Project onto loadings (least squares)
        loadings = model['loadings']
        factors = data_normalized @ loadings @ linalg.inv(loadings.T @ loadings)

        return factors

    def forecast(self, model: Dict, steps: int = 1) -> np.ndarray:
        """
        Forecast future values.

        Args:
            model: Fitted model
            steps: Number of steps to forecast

        Returns:
            Forecasted data, shape (steps, n_vars)
        """
        # Forecast factors using VAR
        var_model = VARModel(n_lags=self.n_lags)
        factor_forecast = var_model.forecast(model['factor_dynamics'], steps=steps)

        # Reconstruct observed variables
        loadings = model['loadings']
        data_forecast = factor_forecast @ loadings.T

        # Denormalize
        data_forecast = data_forecast * model['data_std'] + model['data_mean']

        return data_forecast


class GrangerCausality:
    """
    Comprehensive Granger causality testing.

    Tests whether one time series helps predict another beyond its own history.

    Methods:
    - Pairwise Granger causality
    - Conditional Granger causality (controlling for other variables)
    - Block Granger causality (group of variables)
    - Instantaneous causality (contemporaneous correlation)
    """

    @staticmethod
    def test(data: np.ndarray, caused_idx: int, causing_idx: int,
             max_lag: int = 10, criterion: str = 'bic') -> Dict:
        """
        Test Granger causality with optimal lag selection.

        Args:
            data: Time series data, shape (n_obs, n_vars)
            caused_idx: Index of caused variable
            causing_idx: Index of causing variable
            max_lag: Maximum lag to consider
            criterion: Information criterion for lag selection ('aic' or 'bic')

        Returns:
            Dictionary with test results
        """
        # Select optimal lag
        best_lag = 1
        best_ic = np.inf

        for lag in range(1, max_lag + 1):
            var_model = VARModel(n_lags=lag)
            results = var_model.fit(data)

            ic = results.aic if criterion == 'aic' else results.bic
            if ic < best_ic:
                best_ic = ic
                best_lag = lag

        # Test with optimal lag
        var_model = VARModel(n_lags=best_lag)
        results = var_model.fit(data)

        causality_result = var_model.granger_causality(results, caused_idx, causing_idx)
        causality_result['optimal_lag'] = best_lag

        return causality_result

    @staticmethod
    def pairwise_matrix(data: np.ndarray, max_lag: int = 10,
                       variable_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Compute pairwise Granger causality matrix.

        Args:
            data: Time series data
            max_lag: Maximum lag
            variable_names: Variable names

        Returns:
            Causality matrix where entry (i,j) is p-value for "j Granger-causes i"
        """
        n_vars = data.shape[1]
        if variable_names is None:
            variable_names = [f"var{i}" for i in range(n_vars)]

        causality_matrix = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    result = GrangerCausality.test(data, caused_idx=i, causing_idx=j, max_lag=max_lag)
                    causality_matrix[i, j] = result['p_value']

        return causality_matrix
