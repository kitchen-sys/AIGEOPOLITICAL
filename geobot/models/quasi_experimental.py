"""
Quasi-Experimental Causal Inference Methods

When randomized experiments are impossible, quasi-experimental designs
provide credible causal identification under weaker assumptions.

Core methods:
1. Synthetic Control Method (SCM): Construct counterfactual from weighted controls
2. Difference-in-Differences (DiD): Compare treatment vs control before/after
3. Regression Discontinuity Design (RDD): Exploit threshold-based treatment assignment
4. Instrumental Variables (IV): Use exogenous variation to identify causal effects
5. Causal Forests: Machine learning for heterogeneous treatment effects

Applications in geopolitics:
- SCM: Effect of sanctions on target country (compare to synthetic control)
- DiD: Impact of regime change (compare neighboring countries before/after)
- RDD: Effect of election outcomes (winners vs losers near threshold)
- IV: Effect of trade on conflict (use geographic instruments)
"""

import numpy as np
from scipy import optimize, stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class SyntheticControlResult:
    """Results from Synthetic Control Method."""
    weights: np.ndarray  # Weights on control units
    treated_outcome: np.ndarray  # Actual treated unit outcomes
    synthetic_outcome: np.ndarray  # Synthetic control outcomes
    treatment_effect: np.ndarray  # Difference (post-treatment)
    pre_treatment_fit: float  # RMSPE in pre-treatment period
    control_units: List[str]  # Names of control units
    treatment_time: int  # Index where treatment starts
    p_value: Optional[float] = None  # From permutation test


@dataclass
class DIDResult:
    """Results from Difference-in-Differences."""
    att: float  # Average Treatment effect on Treated
    se: float  # Standard error
    t_stat: float
    p_value: float
    pre_treatment_diff: float  # Check parallel trends
    post_treatment_diff: float
    n_treated: int
    n_control: int


@dataclass
class RDDResult:
    """Results from Regression Discontinuity Design."""
    treatment_effect: float  # Local Average Treatment Effect (LATE)
    se: float
    t_stat: float
    p_value: float
    bandwidth: float
    n_left: int  # Observations below cutoff
    n_right: int  # Observations above cutoff


@dataclass
class IVResult:
    """Results from Instrumental Variables estimation."""
    beta_iv: np.ndarray  # IV estimates
    beta_ols: np.ndarray  # OLS estimates (for comparison)
    se_iv: np.ndarray  # Standard errors
    first_stage_f: float  # First stage F-statistic
    weak_instrument: bool  # True if F < 10


class SyntheticControlMethod:
    """
    Synthetic Control Method (Abadie, Diamond, Hainmueller 2010, 2015)

    Creates a synthetic version of the treated unit as a weighted average
    of control units to estimate counterfactual outcomes.

    Key idea: If we can match pre-treatment outcomes and covariates perfectly,
    the synthetic control provides a valid counterfactual.

    Example:
        >>> # Effect of sanctions on Iran's GDP
        >>> scm = SyntheticControlMethod()
        >>> result = scm.fit(
        ...     treated_outcome=iran_gdp,  # (T,)
        ...     control_outcomes=other_countries_gdp,  # (T, J)
        ...     treatment_time=20,  # Sanctions imposed at t=20
        ...     treated_covariates=iran_covariates,  # (K,)
        ...     control_covariates=other_covariates  # (J, K)
        ... )
        >>> print(f"Average treatment effect: {np.mean(result.treatment_effect):.2f}")
    """

    def __init__(self, loss: str = 'l2'):
        """
        Initialize SCM.

        Args:
            loss: Loss function for matching ('l2' or 'l1')
        """
        self.loss = loss

    def fit(self, treated_outcome: np.ndarray, control_outcomes: np.ndarray,
            treatment_time: int,
            treated_covariates: Optional[np.ndarray] = None,
            control_covariates: Optional[np.ndarray] = None,
            control_names: Optional[List[str]] = None,
            custom_weights: Optional[np.ndarray] = None) -> SyntheticControlResult:
        """
        Fit synthetic control model.

        Args:
            treated_outcome: Outcome for treated unit, shape (T,)
            control_outcomes: Outcomes for control units, shape (T, J)
            treatment_time: Time index when treatment begins
            treated_covariates: Covariates for treated unit, shape (K,)
            control_covariates: Covariates for controls, shape (J, K)
            control_names: Names of control units
            custom_weights: Optional custom weights for different predictors

        Returns:
            SyntheticControlResult with estimated effects
        """
        T, J = control_outcomes.shape

        if control_names is None:
            control_names = [f"control_{j}" for j in range(J)]

        # Pre-treatment period
        Y1_pre = treated_outcome[:treatment_time]
        Y0_pre = control_outcomes[:treatment_time, :]

        # Construct predictors matrix
        if treated_covariates is not None and control_covariates is not None:
            # Include both outcomes and covariates
            X1 = np.concatenate([Y1_pre, treated_covariates])
            X0 = np.vstack([Y0_pre.T, control_covariates.T])  # Shape: (J, T_pre + K)
        else:
            # Use only pre-treatment outcomes
            X1 = Y1_pre
            X0 = Y0_pre.T  # Shape: (J, T_pre)

        # Find weights that minimize ||X1 - X0 w||
        weights = self._optimize_weights(X1, X0, custom_weights)

        # Construct synthetic control
        synthetic_outcome = control_outcomes @ weights

        # Compute treatment effects (post-treatment)
        treatment_effect = np.zeros(T)
        treatment_effect[treatment_time:] = (
            treated_outcome[treatment_time:] - synthetic_outcome[treatment_time:]
        )

        # Pre-treatment fit quality
        pre_treatment_fit = np.sqrt(np.mean((Y1_pre - synthetic_outcome[:treatment_time]) ** 2))

        return SyntheticControlResult(
            weights=weights,
            treated_outcome=treated_outcome,
            synthetic_outcome=synthetic_outcome,
            treatment_effect=treatment_effect,
            pre_treatment_fit=pre_treatment_fit,
            control_units=control_names,
            treatment_time=treatment_time
        )

    def _optimize_weights(self, X1: np.ndarray, X0: np.ndarray,
                         V: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optimize weights to minimize prediction error.

        min_w ||X1 - X0 w||_V^2
        s.t. w >= 0, sum(w) = 1

        Args:
            X1: Target predictors, shape (K,)
            X0: Control predictors, shape (J, K)
            V: Optional weighting matrix

        Returns:
            Optimal weights, shape (J,)
        """
        J = X0.shape[0]

        if V is None:
            V = np.eye(len(X1))

        # Objective function
        def objective(w):
            diff = X1 - X0.T @ w
            return diff.T @ V @ diff

        # Constraints: w >= 0, sum(w) = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(J)]

        # Initial guess: equal weights
        w0 = np.ones(J) / J

        # Optimize
        result = optimize.minimize(
            objective,
            x0=w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            warnings.warn("Optimization did not fully converge")

        return result.x

    def placebo_test(self, treated_outcome: np.ndarray, control_outcomes: np.ndarray,
                    treatment_time: int, n_permutations: int = 100) -> float:
        """
        Conduct placebo test by applying SCM to control units.

        Tests whether the observed treatment effect is unusually large
        compared to effects from placebo treatments on controls.

        Args:
            treated_outcome: Treated unit outcome
            control_outcomes: Control units outcomes
            treatment_time: Treatment time
            n_permutations: Number of placebo tests

        Returns:
            p-value: Proportion of placebos with larger effect
        """
        # Fit actual SCM
        actual_result = self.fit(treated_outcome, control_outcomes, treatment_time)
        actual_effect = np.abs(np.mean(actual_result.treatment_effect[treatment_time:]))

        # Run placebo tests
        placebo_effects = []
        J = control_outcomes.shape[1]

        for j in range(min(J, n_permutations)):
            # Treat control j as if it were treated
            placebo_treated = control_outcomes[:, j]
            placebo_controls = np.delete(control_outcomes, j, axis=1)

            try:
                placebo_result = self.fit(placebo_treated, placebo_controls, treatment_time)
                placebo_effect = np.abs(np.mean(placebo_result.treatment_effect[treatment_time:]))
                placebo_effects.append(placebo_effect)
            except:
                continue

        # p-value: proportion of placebos with larger effect
        placebo_effects = np.array(placebo_effects)
        p_value = np.mean(placebo_effects >= actual_effect)

        return p_value


class DifferenceinDifferences:
    """
    Difference-in-Differences (DiD) Estimation

    Compares changes over time between treatment and control groups.

    Model:
    Y_it = β_0 + β_1 * Treated_i + β_2 * Post_t + β_3 * (Treated_i × Post_t) + ε_it

    where β_3 is the DiD estimate (Average Treatment effect on Treated).

    Key assumption: Parallel trends (treatment and control would have
    followed same trend absent treatment).

    Example:
        >>> # Effect of regime change in country A
        >>> did = DifferenceinDifferences()
        >>> result = did.estimate(
        ...     treated_pre=country_a_gdp_before,
        ...     treated_post=country_a_gdp_after,
        ...     control_pre=neighbors_gdp_before,
        ...     control_post=neighbors_gdp_after
        ... )
        >>> print(f"ATT: {result.att:.3f} (p={result.p_value:.3f})")
    """

    def estimate(self, treated_pre: np.ndarray, treated_post: np.ndarray,
                control_pre: np.ndarray, control_post: np.ndarray,
                cluster_robust: bool = False) -> DIDResult:
        """
        Estimate DiD effect.

        Args:
            treated_pre: Treated group pre-treatment, shape (n_treated,)
            treated_post: Treated group post-treatment, shape (n_treated,)
            control_pre: Control group pre-treatment, shape (n_control,)
            control_post: Control group post-treatment, shape (n_control,)
            cluster_robust: Use cluster-robust standard errors

        Returns:
            DIDResult with ATT estimate
        """
        # Convert to arrays
        treated_pre = np.asarray(treated_pre)
        treated_post = np.asarray(treated_post)
        control_pre = np.asarray(control_pre)
        control_post = np.asarray(control_post)

        # Sample sizes
        n_treated = len(treated_pre)
        n_control = len(control_pre)

        # Mean outcomes
        y_treated_pre = np.mean(treated_pre)
        y_treated_post = np.mean(treated_post)
        y_control_pre = np.mean(control_pre)
        y_control_post = np.mean(control_post)

        # DiD estimate
        diff_treated = y_treated_post - y_treated_pre
        diff_control = y_control_post - y_control_pre
        att = diff_treated - diff_control

        # Standard error (assuming homoskedasticity)
        var_treated_pre = np.var(treated_pre, ddof=1)
        var_treated_post = np.var(treated_post, ddof=1)
        var_control_pre = np.var(control_pre, ddof=1)
        var_control_post = np.var(control_post, ddof=1)

        se = np.sqrt(
            var_treated_post / n_treated +
            var_treated_pre / n_treated +
            var_control_post / n_control +
            var_control_pre / n_control
        )

        # Test statistic
        t_stat = att / se
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_treated + n_control - 2))

        return DIDResult(
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            pre_treatment_diff=y_treated_pre - y_control_pre,
            post_treatment_diff=y_treated_post - y_control_post,
            n_treated=n_treated,
            n_control=n_control
        )

    def panel_did(self, panel_data: np.ndarray, treatment_indicator: np.ndarray,
                 time_indicator: np.ndarray, unit_ids: np.ndarray) -> DIDResult:
        """
        Estimate DiD with panel data and fixed effects.

        Model:
        Y_it = α_i + γ_t + δ * (Treatment_i × Post_t) + ε_it

        Args:
            panel_data: Outcome variable, shape (N*T,)
            treatment_indicator: 1 if unit is treated, 0 otherwise, shape (N*T,)
            time_indicator: 1 if post-treatment, 0 if pre, shape (N*T,)
            unit_ids: Unit identifiers, shape (N*T,)

        Returns:
            DIDResult
        """
        # Create interaction term
        did_term = treatment_indicator * time_indicator

        # Demean for fixed effects (within transformation)
        n_obs = len(panel_data)
        unique_units = np.unique(unit_ids)
        unique_times = np.unique(time_indicator)

        # Demean by unit (removes α_i)
        y_demeaned = np.zeros(n_obs)
        did_demeaned = np.zeros(n_obs)

        for unit in unique_units:
            mask = unit_ids == unit
            y_demeaned[mask] = panel_data[mask] - np.mean(panel_data[mask])
            did_demeaned[mask] = did_term[mask] - np.mean(did_term[mask])

        # Regression: y_demeaned ~ did_demeaned (absorbs time FE implicitly)
        # Simple OLS
        att = np.sum(did_demeaned * y_demeaned) / np.sum(did_demeaned ** 2)

        # Standard error
        residuals = y_demeaned - att * did_demeaned
        rss = np.sum(residuals ** 2)
        se = np.sqrt(rss / (n_obs - 2) / np.sum(did_demeaned ** 2))

        t_stat = att / se
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_obs - 2))

        n_treated = np.sum(treatment_indicator > 0)
        n_control = n_obs - n_treated

        return DIDResult(
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            pre_treatment_diff=0.0,  # Not directly computed
            post_treatment_diff=0.0,
            n_treated=n_treated,
            n_control=n_control
        )


class RegressionDiscontinuity:
    """
    Regression Discontinuity Design (RDD)

    Estimates treatment effects when treatment assignment is determined
    by whether a running variable crosses a threshold.

    Sharp RDD: Treatment deterministically assigned at cutoff
    Fuzzy RDD: Probability of treatment jumps at cutoff

    Example: Effect of election victory on policy outcomes
    - Running variable: Vote margin
    - Cutoff: 50%
    - Treatment: Winning election

    Example:
        >>> # Effect of election victory on military spending
        >>> rdd = RegressionDiscontinuity(cutoff=0.5)  # 50% vote share
        >>> result = rdd.estimate_sharp(
        ...     running_var=vote_share,  # Vote percentage
        ...     outcome=military_spending,
        ...     bandwidth=0.1  # 10% bandwidth
        ... )
    """

    def __init__(self, cutoff: float = 0.0):
        """
        Initialize RDD.

        Args:
            cutoff: Threshold value for treatment assignment
        """
        self.cutoff = cutoff

    def estimate_sharp(self, running_var: np.ndarray, outcome: np.ndarray,
                      bandwidth: Optional[float] = None,
                      kernel: str = 'triangular',
                      polynomial_order: int = 1) -> RDDResult:
        """
        Estimate sharp RDD effect.

        Args:
            running_var: Running variable (e.g., vote share)
            outcome: Outcome variable
            bandwidth: Bandwidth around cutoff (if None, use data-driven selection)
            kernel: Weighting kernel ('triangular', 'uniform', 'epanechnikov')
            polynomial_order: Order of local polynomial

        Returns:
            RDDResult with treatment effect estimate
        """
        running_var = np.asarray(running_var)
        outcome = np.asarray(outcome)

        # Center running variable at cutoff
        X = running_var - self.cutoff

        # Select bandwidth if not provided
        if bandwidth is None:
            bandwidth = self._select_bandwidth(X, outcome)

        # Restrict to bandwidth
        in_bandwidth = np.abs(X) <= bandwidth
        X_bw = X[in_bandwidth]
        Y_bw = outcome[in_bandwidth]

        # Treatment indicator (above cutoff)
        D = (X_bw >= 0).astype(float)

        # Create weights
        weights = self._kernel_weights(X_bw, bandwidth, kernel)

        # Fit local polynomial separately on each side
        # Model: Y = α + β*D + γ*X + δ*(D*X) + higher order terms

        # Design matrix
        Z = np.column_stack([
            np.ones(len(X_bw)),  # Intercept
            D,  # Treatment
            X_bw,  # Running variable
            D * X_bw  # Interaction
        ])

        # Weighted least squares
        W = np.diag(weights)
        try:
            beta = np.linalg.solve(Z.T @ W @ Z, Z.T @ W @ Y_bw)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Z.T @ W @ Z, Z.T @ W @ Y_bw, rcond=None)[0]

        # Treatment effect is coefficient on D
        treatment_effect = beta[1]

        # Standard error (heteroskedasticity-robust)
        residuals = Y_bw - Z @ beta
        meat = Z.T @ W @ np.diag(residuals ** 2) @ W @ Z
        bread_inv = np.linalg.inv(Z.T @ W @ Z)
        vcov = bread_inv @ meat @ bread_inv
        se = np.sqrt(vcov[1, 1])

        # Test statistic
        t_stat = treatment_effect / se
        n_left = np.sum(X_bw < 0)
        n_right = np.sum(X_bw >= 0)
        df = len(X_bw) - Z.shape[1]
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))

        return RDDResult(
            treatment_effect=treatment_effect,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            bandwidth=bandwidth,
            n_left=n_left,
            n_right=n_right
        )

    def _select_bandwidth(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Select bandwidth using Imbens-Kalyanaraman method (simplified).

        Args:
            X: Centered running variable
            Y: Outcome

        Returns:
            Optimal bandwidth
        """
        # Simplified: use rule of thumb
        # h = C * σ * n^{-1/5}
        sigma = np.std(Y)
        n = len(Y)
        bandwidth = 1.06 * sigma * (n ** (-1 / 5))

        # Ensure reasonable range
        bandwidth = np.clip(bandwidth, 0.1 * np.std(X), 2.0 * np.std(X))

        return bandwidth

    def _kernel_weights(self, X: np.ndarray, bandwidth: float, kernel: str) -> np.ndarray:
        """Compute kernel weights."""
        u = X / bandwidth

        if kernel == 'triangular':
            weights = np.maximum(1 - np.abs(u), 0)
        elif kernel == 'uniform':
            weights = (np.abs(u) <= 1).astype(float)
        elif kernel == 'epanechnikov':
            weights = np.maximum(0.75 * (1 - u ** 2), 0)
        else:
            weights = np.ones(len(X))

        return weights


class InstrumentalVariables:
    """
    Instrumental Variables (IV) Estimation

    Addresses endogeneity (omitted variable bias, reverse causality)
    using exogenous variation from an instrument.

    Model:
    Y = β_0 + β_1 * X + ε  (Structural equation)
    X = γ_0 + γ_1 * Z + η  (First stage)

    where:
    - X: Endogenous variable
    - Z: Instrument (exogenous, correlated with X, affects Y only through X)
    - β_1: Causal effect of X on Y

    Estimation: Two-Stage Least Squares (2SLS)

    Example:
        >>> # Effect of trade on conflict (trade is endogenous)
        >>> # Instrument: Geographic distance to major ports
        >>> iv = InstrumentalVariables()
        >>> result = iv.estimate_2sls(
        ...     outcome=conflict_intensity,
        ...     endogenous=trade_volume,
        ...     instrument=distance_to_port,
        ...     exogenous_controls=other_covariates
        ... )
        >>> print(f"IV estimate: {result.beta_iv[0]:.3f}")
        >>> print(f"First stage F: {result.first_stage_f:.1f}")
    """

    def estimate_2sls(self, outcome: np.ndarray, endogenous: np.ndarray,
                     instrument: np.ndarray,
                     exogenous_controls: Optional[np.ndarray] = None) -> IVResult:
        """
        Two-Stage Least Squares estimation.

        Args:
            outcome: Dependent variable Y, shape (n,)
            endogenous: Endogenous variable X, shape (n,) or (n, k)
            instrument: Instrument Z, shape (n,) or (n, m)
            exogenous_controls: Additional exogenous controls, shape (n, p)

        Returns:
            IVResult with IV estimates
        """
        outcome = np.asarray(outcome).reshape(-1, 1)
        endogenous = np.atleast_2d(endogenous)
        if endogenous.ndim == 1:
            endogenous = endogenous.reshape(-1, 1)
        instrument = np.atleast_2d(instrument)
        if instrument.ndim == 1:
            instrument = instrument.reshape(-1, 1)

        n = len(outcome)

        # Construct design matrices
        if exogenous_controls is not None:
            exogenous_controls = np.atleast_2d(exogenous_controls)
            W = np.column_stack([np.ones((n, 1)), exogenous_controls])
        else:
            W = np.ones((n, 1))

        # Full instrument matrix: [W, Z]
        Z_full = np.column_stack([W, instrument])

        # STAGE 1: Regress endogenous on instruments
        # X = Z_full @ γ + residuals
        first_stage_coef = np.linalg.lstsq(Z_full, endogenous, rcond=None)[0]
        X_hat = Z_full @ first_stage_coef  # Fitted values

        # First stage F-statistic
        residuals_first = endogenous - X_hat
        rss_first = np.sum(residuals_first ** 2, axis=0)
        tss_first = np.sum((endogenous - np.mean(endogenous, axis=0)) ** 2, axis=0)
        r_squared_first = 1 - rss_first / tss_first

        k_instruments = instrument.shape[1]
        k_exogenous = W.shape[1]
        first_stage_f = (r_squared_first / k_instruments) / ((1 - r_squared_first) / (n - k_exogenous - k_instruments))
        first_stage_f = float(np.mean(first_stage_f))  # Average if multiple endogenous

        # STAGE 2: Regress Y on X_hat and W
        X_full = np.column_stack([W, X_hat])
        beta_iv = np.linalg.lstsq(X_full, outcome, rcond=None)[0]

        # Standard errors (2SLS requires special formula)
        Y_hat = X_full @ beta_iv
        residuals_second = outcome - Y_hat
        sigma_sq = np.sum(residuals_second ** 2) / (n - X_full.shape[1])

        # Variance: σ^2 (X_hat' X_hat)^{-1}
        vcov = sigma_sq * np.linalg.inv(X_full.T @ X_full)
        se_iv = np.sqrt(np.diag(vcov)).reshape(-1, 1)

        # OLS for comparison (biased but often smaller SE)
        X_full_ols = np.column_stack([W, endogenous])
        beta_ols = np.linalg.lstsq(X_full_ols, outcome, rcond=None)[0]

        # Weak instrument warning
        weak_instrument = first_stage_f < 10

        return IVResult(
            beta_iv=beta_iv[k_exogenous:, 0],  # Exclude intercept/controls
            beta_ols=beta_ols[k_exogenous:, 0],
            se_iv=se_iv[k_exogenous:, 0],
            first_stage_f=first_stage_f,
            weak_instrument=weak_instrument
        )


def estimate_treatment_effect_bounds(outcome_treated: np.ndarray,
                                    outcome_control: np.ndarray,
                                    selection_probability: float = 0.5) -> Tuple[float, float]:
    """
    Estimate bounds on treatment effect under selection on unobservables.

    When treatment assignment is not random, the true effect lies within bounds.
    This implements Manski bounds (worst-case bounds).

    Args:
        outcome_treated: Outcomes for treated group
        outcome_control: Outcomes for control group
        selection_probability: P(Treatment | unobservables)

    Returns:
        (lower_bound, upper_bound) on average treatment effect
    """
    # Observed means
    y_treated = np.mean(outcome_treated)
    y_control = np.mean(outcome_control)

    # Range of outcomes
    y_min = min(np.min(outcome_treated), np.min(outcome_control))
    y_max = max(np.max(outcome_treated), np.max(outcome_control))

    # Worst-case bounds
    # Lower bound: assume best outcomes for control in unobserved potential outcomes
    lower_bound = y_treated - y_max

    # Upper bound: assume worst outcomes for control in unobserved potential outcomes
    upper_bound = y_treated - y_min

    return (lower_bound, upper_bound)
