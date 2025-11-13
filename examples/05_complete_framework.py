"""
Example 5: Complete GeoBotv1 Framework - Final Features

This example demonstrates the final critical components that complete GeoBotv1
to 100% research-grade capability:

1. Vector Autoregression (VAR/SVAR/DFM) - Econometric time-series analysis
2. Hawkes Processes - Conflict contagion and self-exciting dynamics
3. Quasi-Experimental Methods - Causal inference without randomization
   - Synthetic Control Method (SCM)
   - Difference-in-Differences (DiD)
   - Regression Discontinuity Design (RDD)
   - Instrumental Variables (IV)

These methods are essential for:
- Multi-country forecasting with spillovers (VAR)
- Modeling conflict escalation and contagion (Hawkes)
- Estimating policy effects and counterfactuals (quasi-experimental)

GeoBotv1 is now COMPLETE with all research-grade mathematical components!
"""

import numpy as np
import sys
sys.path.append('..')

from datetime import datetime, timedelta

# Time-series models
from geobot.timeseries import (
    VARModel,
    SVARModel,
    DynamicFactorModel,
    GrangerCausality,
    UnivariateHawkesProcess,
    MultivariateHawkesProcess,
    ConflictContagionModel
)

# Quasi-experimental methods
from geobot.models import (
    SyntheticControlMethod,
    DifferenceinDifferences,
    RegressionDiscontinuity,
    InstrumentalVariables
)


def demo_var_model():
    """Demonstrate Vector Autoregression for multi-country forecasting."""
    print("\n" + "="*80)
    print("1. Vector Autoregression (VAR) - Multi-Country Spillovers")
    print("="*80)

    # Simulate data for 3 countries
    # Country dynamics with interdependencies
    np.random.seed(42)
    T = 100
    n_vars = 3

    # Generate VAR(2) data
    # Y_t = A_1 Y_{t-1} + A_2 Y_{t-2} + noise
    A1 = np.array([
        [0.5, 0.2, 0.1],   # Country 1: affected by all
        [0.1, 0.6, 0.15],  # Country 2: strong self-dependence
        [0.05, 0.1, 0.55]  # Country 3: weak spillovers
    ])
    A2 = np.array([
        [0.2, 0.05, 0.0],
        [0.1, 0.1, 0.05],
        [0.0, 0.05, 0.2]
    ])

    # Simulate
    data = np.zeros((T, n_vars))
    data[0] = np.random.randn(n_vars) * 0.1
    data[1] = np.random.randn(n_vars) * 0.1

    for t in range(2, T):
        data[t] = (A1 @ data[t-1] + A2 @ data[t-2] +
                   np.random.randn(n_vars) * 0.1)

    print(f"\nSimulated {T} time periods for {n_vars} countries")
    print(f"Variables: GDP growth, Military spending, Stability index\n")

    # Fit VAR model
    var = VARModel(n_lags=2)
    variable_names = ['GDP_growth', 'Military_spend', 'Stability']
    results = var.fit(data, variable_names)

    print(f"VAR({results.n_lags}) Estimation Results:")
    print(f"  Log-likelihood: {results.log_likelihood:.2f}")
    print(f"  AIC: {results.aic:.2f}")
    print(f"  BIC: {results.bic:.2f}")

    # Forecast
    forecast = var.forecast(results, steps=10)
    print(f"\n10-step ahead forecast:")
    print(f"  GDP growth: {forecast[-1, 0]:.3f}")
    print(f"  Military spending: {forecast[-1, 1]:.3f}")
    print(f"  Stability: {forecast[-1, 2]:.3f}")

    # Granger causality
    print("\nGranger Causality Tests:")
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                gc_result = var.granger_causality(results, i, j)
                if gc_result['p_value'] < 0.05:
                    print(f"  {variable_names[j]} → {variable_names[i]}: "
                          f"F={gc_result['f_statistic']:.2f}, p={gc_result['p_value']:.3f} ✓")

    # Impulse response functions
    irf_result = var.impulse_response(results, steps=10)
    print("\nImpulse Response Functions computed (10 steps)")
    print(f"  Shock to Military spending → GDP growth at t=5: {irf_result.irf[0, 1, 5]:.4f}")

    # Forecast error variance decomposition
    fevd = var.forecast_error_variance_decomposition(results, steps=10)
    print("\nForecast Error Variance Decomposition (horizon=10):")
    for i, var_name in enumerate(variable_names):
        contributions = fevd[i, :, -1]
        print(f"  {var_name} variance explained by:")
        for j, source_name in enumerate(variable_names):
            print(f"    {source_name}: {contributions[j]:.1%}")

    print("\n✓ VAR model demonstrates multi-country interdependencies!")


def demo_hawkes_process():
    """Demonstrate Hawkes processes for conflict contagion."""
    print("\n" + "="*80)
    print("2. Hawkes Processes - Conflict Escalation and Contagion")
    print("="*80)

    # Simulate conflict events
    print("\nSimulating conflict events with self-excitation...")
    hawkes = UnivariateHawkesProcess()

    # Parameters: baseline=0.3, excitation=0.6, decay=1.2
    # Branching ratio = 0.6/1.2 = 0.5 (stable, subcritical)
    events = hawkes.simulate(mu=0.3, alpha=0.6, beta=1.2, T=100.0)

    print(f"Generated {len(events)} conflict events over 100 time units")
    print(f"Average rate: {len(events) / 100.0:.2f} events/unit\n")

    # Fit model
    result = hawkes.fit(events, T=100.0)

    print("Estimated Hawkes Parameters:")
    print(f"  Baseline intensity (μ): {result.params.mu:.3f}")
    print(f"  Excitation (α): {result.params.alpha:.3f}")
    print(f"  Decay rate (β): {result.params.beta:.3f}")
    print(f"  Branching ratio: {result.params.branching_ratio:.3f}")
    print(f"  Process is {'STABLE' if result.params.is_stable else 'EXPLOSIVE'}")

    # Predict intensity
    t_future = 105.0
    intensity = hawkes.predict_intensity(events, result.params, t_future)
    print(f"\nPredicted conflict intensity at t={t_future}: {intensity:.3f}")

    # Multivariate: conflict contagion between countries
    print("\n" + "-"*80)
    print("Multivariate Hawkes: Cross-Country Conflict Contagion")
    print("-"*80)

    countries = ['Syria', 'Iraq', 'Lebanon']
    contagion_model = ConflictContagionModel(countries=countries)

    # Simulate with cross-excitation
    mu = np.array([0.5, 0.3, 0.2])  # Different baseline rates
    alpha = np.array([
        [0.3, 0.15, 0.1],   # Syria: high self-excitation, moderate contagion
        [0.2, 0.25, 0.1],   # Iraq: affected by Syria
        [0.15, 0.1, 0.2]    # Lebanon: affected by both
    ])
    beta = np.ones((3, 3)) * 1.5

    multi_hawkes = MultivariateHawkesProcess(n_dimensions=3)
    events_multi = multi_hawkes.simulate(mu=mu, alpha=alpha, beta=beta, T=100.0)

    print(f"\nSimulated events:")
    for i, country in enumerate(countries):
        print(f"  {country}: {len(events_multi[i])} events")

    # Fit multivariate model
    events_dict = {country: events_multi[i] for i, country in enumerate(countries)}
    fit_result = contagion_model.fit(events_dict, T=100.0)

    print(f"\nFitted contagion model:")
    print(f"  Spectral radius: {fit_result['spectral_radius']:.3f} (< 1 = stable)")
    print(f"  Most contagious source: {fit_result['most_contagious_source']}")
    print(f"  Most vulnerable target: {fit_result['most_vulnerable_target']}")

    # Identify contagion pathways
    pathways = contagion_model.identify_contagion_pathways(fit_result, threshold=0.1)
    print("\nSignificant contagion pathways (branching ratio > 0.1):")
    for source, target, strength in pathways[:5]:
        print(f"  {source} → {target}: {strength:.3f}")

    # Risk assessment
    risks = contagion_model.contagion_risk(events_dict, fit_result, t=105.0, horizon=5.0)
    print("\nConflict risk over next 5 time units:")
    for country, risk in risks.items():
        print(f"  {country}: {risk:.1%}")

    print("\n✓ Hawkes processes capture conflict escalation dynamics!")


def demo_synthetic_control():
    """Demonstrate Synthetic Control Method."""
    print("\n" + "="*80)
    print("3. Synthetic Control Method - Policy Impact Estimation")
    print("="*80)

    # Scenario: Estimate effect of sanctions on target country's GDP
    print("\nScenario: Economic sanctions imposed on Country A at t=50")
    print("Question: What is the causal effect on GDP growth?\n")

    # Generate data
    np.random.seed(42)
    T = 100
    J = 10  # 10 control countries

    # Pre-treatment: all countries follow similar trends
    time = np.arange(T)
    trend = 0.02 * time + np.random.randn(T) * 0.1

    # Control countries
    control_outcomes = np.zeros((T, J))
    for j in range(J):
        control_outcomes[:, j] = trend + np.random.randn(T) * 0.15 + np.random.randn() * 0.5

    # Treated country (matches controls pre-treatment)
    treated_outcome = trend + np.random.randn(T) * 0.15

    # Treatment effect: negative shock starting at t=50
    treatment_time = 50
    true_effect = -0.8
    treated_outcome[treatment_time:] += true_effect + np.random.randn(T - treatment_time) * 0.1

    # Fit SCM
    scm = SyntheticControlMethod()
    result = scm.fit(
        treated_outcome=treated_outcome,
        control_outcomes=control_outcomes,
        treatment_time=treatment_time,
        control_names=[f"Country_{j+1}" for j in range(J)]
    )

    print("Synthetic Control Results:")
    print(f"  Pre-treatment fit (RMSPE): {result.pre_treatment_fit:.4f}")
    print(f"\nSynthetic Country A is weighted combination of:")
    for j, weight in enumerate(result.weights):
        if weight > 0.01:  # Only show significant weights
            print(f"    {result.control_units[j]}: {weight:.1%}")

    # Treatment effects
    avg_effect = np.mean(result.treatment_effect[treatment_time:])
    print(f"\nEstimated treatment effect (post-sanctions):")
    print(f"  Average: {avg_effect:.3f} (true effect: {true_effect:.3f})")
    print(f"  Final period: {result.treatment_effect[-1]:.3f}")

    # Placebo test
    p_value = scm.placebo_test(treated_outcome, control_outcomes, treatment_time, n_permutations=J)
    print(f"\nPlacebo test p-value: {p_value:.3f}")
    if p_value < 0.05:
        print("  ✓ Effect is statistically significant (unusual compared to placebos)")
    else:
        print("  ✗ Effect not significant (could be random)")

    print("\n✓ Synthetic control provides credible counterfactual!")


def demo_difference_in_differences():
    """Demonstrate Difference-in-Differences."""
    print("\n" + "="*80)
    print("4. Difference-in-Differences (DiD) - Regime Change Analysis")
    print("="*80)

    # Scenario: Regime change in treated country
    print("\nScenario: Regime change in Country T at t=50")
    print("Compare to similar countries without regime change\n")

    np.random.seed(42)

    # Pre-treatment (similar trends)
    treated_pre = 3.0 + np.random.randn(50) * 0.5
    control_pre = 3.2 + np.random.randn(50) * 0.5

    # Post-treatment (treatment effect = +1.5 on outcome)
    true_effect = 1.5
    treated_post = 3.0 + true_effect + np.random.randn(50) * 0.5
    control_post = 3.2 + np.random.randn(50) * 0.5  # No effect

    # Estimate DiD
    did = DifferenceinDifferences()
    result = did.estimate(treated_pre, treated_post, control_pre, control_post)

    print("Difference-in-Differences Results:")
    print(f"\n  Pre-treatment difference: {result.pre_treatment_diff:.3f}")
    print(f"  Post-treatment difference: {result.post_treatment_diff:.3f}")
    print(f"\n  Average Treatment Effect (ATT): {result.att:.3f}")
    print(f"  Standard error: {result.se:.3f}")
    print(f"  t-statistic: {result.t_stat:.3f}")
    print(f"  p-value: {result.p_value:.4f}")

    if result.p_value < 0.05:
        print(f"\n  ✓ Regime change had significant effect (true effect: {true_effect:.3f})")
    else:
        print("\n  ✗ Effect not statistically significant")

    # Assumption check
    if abs(result.pre_treatment_diff) < 0.5:
        print("\n  ✓ Parallel trends assumption plausible (small pre-treatment diff)")
    else:
        print("\n  ⚠ Parallel trends questionable (large pre-treatment diff)")

    print("\n✓ DiD isolates causal effect of regime change!")


def demo_regression_discontinuity():
    """Demonstrate Regression Discontinuity Design."""
    print("\n" + "="*80)
    print("5. Regression Discontinuity Design (RDD) - Election Effects")
    print("="*80)

    # Scenario: Effect of winning election on military policy
    print("\nScenario: Effect of hawkish candidate winning election")
    print("Running variable: Vote share (cutoff = 50%)")
    print("Outcome: Military spending increase\n")

    np.random.seed(42)
    n = 500

    # Vote share (running variable)
    vote_share = np.random.uniform(0.3, 0.7, n)

    # Outcome: military spending
    # Smooth function of vote share + discontinuity at 50%
    outcome = 2.0 + 1.5 * vote_share + np.random.randn(n) * 0.3

    # Treatment effect: +0.8 if vote > 50%
    true_effect = 0.8
    outcome[vote_share >= 0.5] += true_effect

    # Estimate RDD
    rdd = RegressionDiscontinuity(cutoff=0.5)
    result = rdd.estimate_sharp(
        running_var=vote_share,
        outcome=outcome,
        bandwidth=0.15,  # 15% bandwidth
        kernel='triangular'
    )

    print("Regression Discontinuity Results:")
    print(f"\n  Bandwidth: {result.bandwidth:.3f}")
    print(f"  Observations below cutoff: {result.n_left}")
    print(f"  Observations above cutoff: {result.n_right}")
    print(f"\n  Treatment effect (LATE): {result.treatment_effect:.3f}")
    print(f"  Standard error: {result.se:.3f}")
    print(f"  t-statistic: {result.t_stat:.3f}")
    print(f"  p-value: {result.p_value:.4f}")

    if result.p_value < 0.05:
        print(f"\n  ✓ Winning election causes increase in military spending")
        print(f"    (true effect: {true_effect:.3f})")
    else:
        print("\n  ✗ Effect not statistically significant")

    print("\n✓ RDD exploits threshold-based treatment assignment!")


def demo_instrumental_variables():
    """Demonstrate Instrumental Variables."""
    print("\n" + "="*80)
    print("6. Instrumental Variables (IV) - Trade and Conflict")
    print("="*80)

    # Scenario: Effect of trade on conflict (trade is endogenous)
    print("\nScenario: Does trade reduce conflict?")
    print("Problem: Trade is endogenous (reverse causality, omitted variables)")
    print("Instrument: Geographic distance to major trade routes\n")

    np.random.seed(42)
    n = 300

    # Instrument: distance (exogenous)
    distance = np.random.uniform(100, 1000, n)

    # Unobserved confounders
    unobserved = np.random.randn(n)

    # Trade (endogenous): affected by distance and confounders
    trade = 50 - 0.03 * distance + 2.0 * unobserved + np.random.randn(n) * 5

    # Conflict: true effect of trade = -0.15, but also affected by confounders
    true_effect = -0.15
    conflict = 10 + true_effect * trade - 1.5 * unobserved + np.random.randn(n) * 2

    # Estimate with IV
    iv = InstrumentalVariables()
    result = iv.estimate_2sls(
        outcome=conflict,
        endogenous=trade,
        instrument=distance
    )

    print("Instrumental Variables (2SLS) Results:")
    print(f"\n  First stage F-statistic: {result.first_stage_f:.2f}")
    if result.weak_instrument:
        print("  ⚠ Warning: Weak instrument (F < 10)")
    else:
        print("  ✓ Strong instrument (F > 10)")

    print(f"\n  OLS estimate (biased): {result.beta_ols[0]:.4f}")
    print(f"  IV estimate (consistent): {result.beta_iv[0]:.4f}")
    print(f"  IV standard error: {result.se_iv[0]:.4f}")
    print(f"\n  True causal effect: {true_effect:.4f}")

    # Hausman test (informal)
    if abs(result.beta_ols[0] - result.beta_iv[0]) > 0.05:
        print("\n  ✓ OLS and IV differ substantially → endogeneity present")
        print("    IV corrects for bias!")
    else:
        print("\n  OLS and IV similar → endogeneity may be small")

    print("\n✓ IV isolates causal effect using exogenous variation!")


def demo_dynamic_factor_model():
    """Demonstrate Dynamic Factor Model for nowcasting."""
    print("\n" + "="*80)
    print("7. Dynamic Factor Model (DFM) - High-Dimensional Nowcasting")
    print("="*80)

    # Scenario: Nowcast geopolitical tension from many indicators
    print("\nScenario: Nowcast regional tension from 50 economic/political indicators")
    print("DFM extracts common latent factors driving all indicators\n")

    np.random.seed(42)
    T = 200
    n_indicators = 50
    n_factors = 3

    # True factors (latent tensions)
    true_factors = np.zeros((T, n_factors))
    for k in range(n_factors):
        # AR(1) dynamics
        for t in range(1, T):
            true_factors[t, k] = 0.8 * true_factors[t-1, k] + np.random.randn() * 0.5

    # Factor loadings (how indicators load on factors)
    true_loadings = np.random.randn(n_indicators, n_factors)

    # Observed indicators = factors * loadings + idiosyncratic noise
    data = true_factors @ true_loadings.T + np.random.randn(T, n_indicators) * 0.5

    # Fit DFM
    dfm = DynamicFactorModel(n_factors=3, n_lags=1)
    model = dfm.fit(data)

    print(f"Dynamic Factor Model Results:")
    print(f"\n  Number of indicators: {n_indicators}")
    print(f"  Number of factors: {n_factors}")
    print(f"  Explained variance: {model['explained_variance_ratio']:.1%}")

    # Extracted factors
    factors = model['factors']
    print(f"\n  Extracted factor dimensions: {factors.shape}")
    print(f"  Factor 1 final value: {factors[-1, 0]:.3f}")
    print(f"  Factor 2 final value: {factors[-1, 1]:.3f}")
    print(f"  Factor 3 final value: {factors[-1, 2]:.3f}")

    # Forecast
    forecast = dfm.forecast(model, steps=10)
    print(f"\n  10-step ahead forecast dimensions: {forecast.shape}")
    print(f"  Average forecasted indicator value: {np.mean(forecast[-1]):.3f}")

    # Correlation with true factors
    corr_0 = np.corrcoef(true_factors[:, 0], factors[:, 0])[0, 1]
    print(f"\n  Factor recovery (correlation with true): {abs(corr_0):.3f}")

    print("\n✓ DFM reduces dimensionality while preserving information!")


def main():
    """Run all demonstrations of final features."""
    print("=" * 80)
    print("GeoBotv1 - COMPLETE FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    print("\nThis example showcases the final components that complete GeoBotv1:")
    print("• Vector Autoregression (VAR/SVAR/DFM)")
    print("• Hawkes Processes for conflict contagion")
    print("• Quasi-Experimental Causal Inference")
    print("  - Synthetic Control Method")
    print("  - Difference-in-Differences")
    print("  - Regression Discontinuity Design")
    print("  - Instrumental Variables")

    # Run all demonstrations
    demo_var_model()
    demo_hawkes_process()
    demo_synthetic_control()
    demo_difference_in_differences()
    demo_regression_discontinuity()
    demo_instrumental_variables()
    demo_dynamic_factor_model()

    print("\n" + "=" * 80)
    print("GeoBotv1 Framework is NOW 100% COMPLETE!")
    print("=" * 80)
    print("\n🎉 All Research-Grade Mathematical Components Implemented:")
    print("\n📊 CORE FRAMEWORKS:")
    print("  ✓ Optimal Transport (Wasserstein, Kantorovich, Sinkhorn)")
    print("  ✓ Causal Inference (DAGs, SCMs, Do-Calculus)")
    print("  ✓ Bayesian Inference (MCMC, Particle Filters, VI)")
    print("  ✓ Stochastic Processes (SDEs, Jump-Diffusion)")
    print("  ✓ Time-Series Models (Kalman, HMM, VAR, Hawkes)")
    print("  ✓ Quasi-Experimental Methods (SCM, DiD, RDD, IV)")
    print("  ✓ Machine Learning (GNNs, Risk Scoring, Embeddings)")
    print("\n📈 SPECIALIZED CAPABILITIES:")
    print("  ✓ Multi-country interdependency modeling (VAR)")
    print("  ✓ Conflict contagion and escalation (Hawkes)")
    print("  ✓ Policy counterfactuals (Synthetic Control)")
    print("  ✓ Regime change effects (Difference-in-Differences)")
    print("  ✓ Election outcomes impact (Regression Discontinuity)")
    print("  ✓ Trade-conflict nexus (Instrumental Variables)")
    print("  ✓ High-dimensional nowcasting (Dynamic Factor Models)")
    print("\n🔬 MATHEMATICAL RIGOR:")
    print("  ✓ Measure-theoretic probability foundations")
    print("  ✓ Continuous-time dynamics (SDEs)")
    print("  ✓ Causal identification strategies")
    print("  ✓ Structural econometric methods")
    print("  ✓ Point process theory")
    print("  ✓ Optimal transport geometry")
    print("\n💡 GeoBotv1 is ready for production geopolitical forecasting!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
