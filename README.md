# GeoBotv1: Research-Grade Geopolitical Forecasting Framework

[![Status](https://img.shields.io/badge/status-production-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.9+-blue)]() [![License](https://img.shields.io/badge/license-MIT-blue)]()

**GeoBotv1** is a complete, research-grade framework for geopolitical forecasting, conflict prediction, and causal policy analysis. Built on rigorous mathematical foundations, it combines optimal transport theory, structural causal inference, Bayesian reasoning, stochastic processes, econometric methods, and machine learning to provide actionable intelligence on regime shifts, conflict escalation, and intervention outcomes.

**Status: ✅ 100% Complete** - All core mathematical frameworks implemented and production-ready.

---

## 🎯 Key Capabilities

- **Causal Policy Analysis**: Simulate interventions (sanctions, deployments, regime changes) and estimate counterfactual outcomes
- **Conflict Contagion Modeling**: Model self-exciting escalation dynamics and cross-country spillovers using Hawkes processes
- **Multi-Country Forecasting**: Capture interdependencies and shock propagation with Vector Autoregression (VAR/SVAR)
- **Quasi-Experimental Inference**: Estimate treatment effects from observational data (Synthetic Control, DiD, RDD, IV)
- **Regime Detection**: Identify structural breaks and state transitions in real-time
- **Scenario Comparison**: Measure distances between geopolitical futures using optimal transport geometry
- **Intelligence Integration**: Bayesian belief updating from text, events, and structured data
- **Nowcasting**: High-dimensional factor models for real-time situational awareness

---

## 📊 Complete Mathematical Framework

### ✅ Implemented Core Components

<table>
<tr>
<td width="50%">

**1. Optimal Transport Theory**
- Wasserstein distances (W1, W2, W∞)
- Kantorovich duality (primal/dual formulations)
- Sinkhorn algorithm (entropic regularization)
- Unbalanced optimal transport
- Gromov-Wasserstein for network comparison
- Gradient-based OT optimization

**2. Causal Inference**
- Directed Acyclic Graphs (DAGs)
- Structural Causal Models (SCMs)
- Pearl's Do-Calculus for interventions
- Backdoor/frontdoor adjustment
- Counterfactual computation
- Causal discovery algorithms

**3. Bayesian Inference**
- Markov Chain Monte Carlo (MCMC)
- Sequential Monte Carlo (Particle Filters)
  - Bootstrap, Auxiliary, Rao-Blackwellized
- Variational Inference (ELBO, CAVI, ADVI)
- Belief updating from intelligence
- Posterior predictive distributions

**4. Stochastic Processes**
- Stochastic Differential Equations (SDEs)
  - Euler-Maruyama, Milstein, Stochastic RK
- Jump-diffusion processes (Merton model)
- Ornstein-Uhlenbeck processes
- GeopoliticalSDE framework
- Continuous-time dynamics

</td>
<td width="50%">

**5. Time-Series & Econometrics**
- Kalman Filters (Linear & Extended)
- Hidden Markov Models (HMM)
- Regime-Switching Models
- **Vector Autoregression (VAR)**
- **Structural VAR (SVAR)**
- **Dynamic Factor Models (DFM)**
- **Granger Causality Testing**
- Impulse Response Functions (IRF)
- Forecast Error Variance Decomposition

**6. Point Processes**
- **Univariate Hawkes Processes**
- **Multivariate Hawkes Processes**
- **Conflict Contagion Models**
- Branching ratio estimation
- Self-excitation dynamics
- Cross-country excitation matrices

**7. Quasi-Experimental Methods**
- **Synthetic Control Method (SCM)**
- **Difference-in-Differences (DiD)**
- **Regression Discontinuity Design (RDD)**
- **Instrumental Variables (2SLS)**
- Placebo tests and robustness checks
- Treatment effect bounds

**8. Machine Learning**
- Graph Neural Networks (GNN, GAT)
- CausalGNN for directed graphs
- Risk scoring and classification
- Feature discovery and embeddings
- Transformer-based text encoding

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/AIGEOPOLITICAL.git
cd AIGEOPOLITICAL

# Install dependencies
pip install -r requirements.txt

# Optional: Install Graph Neural Network support
# (Requires matching torch version)
pip install torch-geometric torch-scatter torch-sparse
```

### Basic Usage Example

```python
from geobot.core import Scenario, ScenarioComparator
from geobot.models import CausalGraph
from geobot.inference import DoCalculus
from geobot.timeseries import VARModel
import numpy as np

# 1. Create geopolitical scenarios
scenario_baseline = Scenario(
    name="Baseline",
    features={
        "military_readiness": np.array([0.6, 0.4, 0.5]),
        "economic_stability": np.array([0.7, 0.6, 0.5])
    }
)

scenario_intervention = Scenario(
    name="Post-Sanctions",
    features={
        "military_readiness": np.array([0.8, 0.4, 0.5]),
        "economic_stability": np.array([0.3, 0.6, 0.5])
    }
)

# 2. Compare scenarios using optimal transport
comparator = ScenarioComparator()
distance = comparator.compare(scenario_baseline, scenario_intervention)
print(f"Wasserstein distance: {distance:.4f}")

# 3. Build causal model
causal_graph = CausalGraph()
causal_graph.add_edge("sanctions", "economy", strength=0.8)
causal_graph.add_edge("economy", "stability", strength=0.6)
causal_graph.add_edge("stability", "conflict_risk", strength=-0.7)

# 4. Simulate intervention
do_calc = DoCalculus(causal_graph)
intervention_effect = do_calc.compute_intervention_effect(
    intervention={"sanctions": 1.0},
    outcome="conflict_risk"
)
print(f"Estimated effect on conflict risk: {intervention_effect:.3f}")

# 5. Multi-country VAR forecasting
# (Simulated data for demonstration)
data = np.random.randn(100, 3)  # 100 time periods, 3 countries
var = VARModel(n_lags=2)
results = var.fit(data, variable_names=['Country_A', 'Country_B', 'Country_C'])
forecast = var.forecast(results, steps=10)
print(f"10-step forecast:\n{forecast}")
```

### Conflict Contagion Example

```python
from geobot.timeseries import ConflictContagionModel

# Model conflict spread between countries
countries = ['Syria', 'Iraq', 'Lebanon', 'Turkey']
model = ConflictContagionModel(countries=countries)

# Historical conflict events (times when conflicts occurred)
events = {
    'Syria': [1.2, 5.3, 10.1, 15.2, 22.3],
    'Iraq': [3.4, 8.9, 12.1, 18.5],
    'Lebanon': [12.3, 19.8, 25.1],
    'Turkey': [28.2, 30.5]
}

# Fit contagion model
result = model.fit(events, T=365.0)  # 1 year of data

print(f"Most contagious source: {result['most_contagious_source']}")
print(f"Most vulnerable target: {result['most_vulnerable_target']}")

# Assess future risk
risks = model.contagion_risk(events, result, t=370.0, horizon=30.0)
for country, risk in risks.items():
    print(f"{country} conflict risk (next 30 days): {risk:.1%}")
```

### Synthetic Control for Policy Evaluation

```python
from geobot.models import SyntheticControlMethod
import numpy as np

# Evaluate impact of sanctions on target country
scm = SyntheticControlMethod()

# Data: treated country vs. control countries
treated_outcome = gdp_growth_target_country  # Shape: (T,)
control_outcomes = gdp_growth_other_countries  # Shape: (T, J)

result = scm.fit(
    treated_outcome=treated_outcome,
    control_outcomes=control_outcomes,
    treatment_time=50,  # Sanctions imposed at t=50
    control_names=['Country_1', 'Country_2', ..., 'Country_J']
)

# Estimated treatment effect
avg_effect = np.mean(result.treatment_effect[50:])
print(f"Average treatment effect: {avg_effect:.3f}")

# Statistical significance via placebo test
p_value = scm.placebo_test(treated_outcome, control_outcomes,
                           treatment_time=50, n_permutations=100)
print(f"Placebo test p-value: {p_value:.3f}")
```

---

## 📁 Project Structure

```
AIGEOPOLITICAL/
├── geobot/                          # Main package
│   ├── core/                        # Core mathematical primitives
│   │   ├── optimal_transport.py     # Wasserstein distances
│   │   ├── advanced_optimal_transport.py  # Kantorovich, Sinkhorn, Gromov-W
│   │   └── scenario.py              # Scenario representations
│   │
│   ├── models/                      # Causal models
│   │   ├── causal_graph.py          # DAGs and SCMs
│   │   ├── causal_discovery.py      # Causal structure learning
│   │   └── quasi_experimental.py    # SCM, DiD, RDD, IV methods
│   │
│   ├── inference/                   # Probabilistic inference
│   │   ├── do_calculus.py           # Interventions and counterfactuals
│   │   ├── bayesian_engine.py       # Bayesian belief updating
│   │   ├── particle_filter.py       # Sequential Monte Carlo
│   │   └── variational_inference.py # VI, ELBO, ADVI
│   │
│   ├── simulation/                  # Stochastic simulation
│   │   ├── monte_carlo.py           # Basic Monte Carlo
│   │   ├── sde_solver.py            # Stochastic differential equations
│   │   └── agent_based.py           # Agent-based models
│   │
│   ├── timeseries/                  # Time-series models
│   │   ├── kalman_filter.py         # Kalman filters
│   │   ├── hmm.py                   # Hidden Markov Models
│   │   ├── regime_switching.py      # Regime-switching models
│   │   ├── var_models.py            # VAR, SVAR, DFM, Granger causality
│   │   └── point_processes.py       # Hawkes processes, conflict contagion
│   │
│   ├── ml/                          # Machine learning
│   │   ├── risk_models.py           # Risk scoring
│   │   ├── graph_neural_networks.py # GNNs for causal/geopolitical networks
│   │   └── embeddings.py            # Feature embeddings
│   │
│   ├── data_ingestion/              # Data processing
│   │   ├── pdf_reader.py            # PDF intelligence extraction
│   │   ├── web_scraper.py           # Web scraping
│   │   ├── event_extraction.py      # NLP-based event structuring
│   │   └── event_database.py        # Event storage and querying
│   │
│   ├── utils/                       # Utilities
│   │   ├── data_processing.py       # Data preprocessing
│   │   └── visualization.py         # Plotting and visualization
│   │
│   └── config/                      # Configuration
│       └── settings.py              # System configuration
│
├── examples/                        # Comprehensive examples
│   ├── 01_basic_usage.py            # Scenarios, causal graphs, Monte Carlo
│   ├── 02_data_ingestion.py        # PDF/web scraping pipeline
│   ├── 03_intervention_simulation.py # Do-calculus and counterfactuals
│   ├── 04_advanced_features.py     # Particle filters, VI, SDEs, GNNs
│   └── 05_complete_framework.py    # VAR, Hawkes, quasi-experimental
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🔬 Mathematical Foundations

### Causality-First Principle

GeoBotv1 grounds all forecasting in **explicit causal structure**. This prevents spurious correlations and enables simulation of interventions that have never been observed.

**Structural Causal Model (SCM):**
```
X := f_X(Pa_X, U_X)
Y := f_Y(Pa_Y, U_Y)
```

**Pearl's Do-Calculus** enables reasoning about interventions `do(X = x)` even when only observational data is available.

### Optimal Transport for Scenario Comparison

The Wasserstein distance measures the "cost" of transforming one probability distribution into another:

```
W_2(μ, ν) = inf_{π ∈ Π(μ,ν)} √(∫∫ ||x - y||² dπ(x,y))
```

**Kantorovich Duality** provides computational efficiency and geometric interpretation.

### Hawkes Processes for Conflict Dynamics

Self-exciting point processes capture escalation and contagion:

```
λ_k(t) = μ_k + ∑_{j=1}^K α_{kj} ∑_{t_i^j < t} exp(-β_{kj}(t - t_i^j))
```

**Branching ratio** `n = α/β` determines stability:
- `n < 1`: Process is stable (subcritical)
- `n ≥ 1`: Process is explosive (supercritical)

### Stochastic Differential Equations

Continuous-time geopolitical dynamics:

```
dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t + dJ_t
```

Where:
- `μ`: Drift (deterministic component)
- `σ dW_t`: Diffusion (continuous shocks)
- `dJ_t`: Jumps (discrete events)

---

## 💡 Use Cases

### 1. Strategic Intelligence & Risk Assessment
- **Multi-country risk dashboards** with VAR-based interdependency analysis
- **Real-time belief updating** as intelligence arrives (Bayesian inference)
- **Regime detection** using HMM and particle filters

### 2. Conflict Prediction & Early Warning
- **Hawkes process models** for escalation dynamics and contagion
- **Structural break detection** in military/economic indicators
- **Cross-country spillover analysis** with SVAR impulse responses

### 3. Policy Impact Analysis
- **Synthetic control** for sanctions/intervention evaluation
- **Difference-in-differences** for regime change effects
- **Regression discontinuity** for election outcome impacts
- **Do-calculus** for counterfactual policy simulation

### 4. Resource Allocation & Logistics
- **Optimal transport** for supply chain optimization under uncertainty
- **Monte Carlo simulation** for contingency planning
- **Stochastic optimization** with SDE-based constraints

### 5. Diplomatic Forecasting
- **Game-theoretic extensions** (can be added via causal graphs)
- **Network centrality** analysis with GNNs
- **Alliance dynamics** using multivariate Hawkes processes

---

## 🛠️ Technical Requirements

### Core Dependencies
- **Python**: 3.9+
- **NumPy**: 1.24+
- **SciPy**: 1.10+
- **Pandas**: 2.0+
- **NetworkX**: 3.0+ (causal graphs)
- **POT**: 0.9+ (optimal transport)
- **PyMC**: 5.0+ (Bayesian inference)
- **Statsmodels**: 0.14+ (econometrics)

### Optional but Recommended
- **PyTorch**: 2.0+ (for GNNs, deep learning)
- **PyTorch Geometric**: 2.3+ (graph neural networks)
- **spaCy**: 3.6+ (NLP for event extraction)
- **Transformers**: 4.30+ (text embeddings)

### Data Ingestion
- **Beautiful Soup**: 4.12+ (web scraping)
- **PDFPlumber**: 0.10+ (PDF extraction)
- **Newspaper3k**: 0.2.8+ (article extraction)
- **Feedparser**: 6.0+ (RSS feeds)

See `requirements.txt` for complete dependency list.

---

## 📖 Documentation & Examples

### Example Scripts
All examples are fully functional and demonstrate end-to-end workflows:

1. **`01_basic_usage.py`**: Scenarios, causal graphs, Monte Carlo basics, Bayesian updating
2. **`02_data_ingestion.py`**: PDF extraction, web scraping, event databases
3. **`03_intervention_simulation.py`**: Do-calculus, counterfactuals, policy simulation
4. **`04_advanced_features.py`**: Particle filters, VI, SDEs, GNNs, event extraction
5. **`05_complete_framework.py`**: VAR/SVAR/DFM, Hawkes processes, quasi-experimental methods

### Running Examples

```bash
cd examples

# Basic usage
python 01_basic_usage.py

# Data ingestion pipeline
python 02_data_ingestion.py

# Intervention simulation
python 03_intervention_simulation.py

# Advanced mathematical features
python 04_advanced_features.py

# Complete framework demonstration (VAR, Hawkes, quasi-experimental)
python 05_complete_framework.py
```

---

## 🎓 Theoretical Background

GeoBotv1 synthesizes methods from multiple fields:

### Economics & Econometrics
- Vector Autoregression (Sims, 1980)
- Structural VAR identification (Blanchard & Quah, 1989)
- Synthetic Control Method (Abadie et al., 2010, 2015)
- Difference-in-Differences (Card & Krueger, 1994)
- Regression Discontinuity (Thistlethwaite & Campbell, 1960)

### Statistics & Probability
- Optimal Transport (Villani, 2003, 2009)
- Hawkes Processes (Hawkes, 1971; Hawkes & Oakes, 1974)
- Sequential Monte Carlo (Doucet et al., 2001)
- Variational Inference (Jordan et al., 1999; Blei et al., 2017)

### Computer Science & AI
- Causal Inference (Pearl, 2000, 2009)
- Do-Calculus (Pearl, 1995)
- Graph Neural Networks (Kipf & Welling, 2017; Veličković et al., 2018)
- Structural Causal Models (Pearl & Mackenzie, 2018)

### Geopolitics & Conflict Studies
- Conflict contagion (Gleditsch, 2007; Braithwaite, 2010)
- Regime change dynamics (Acemoglu & Robinson, 2006)
- Economic sanctions effects (Hufbauer et al., 2007)

---

## 🧪 Testing & Validation

### Mathematical Correctness
- Kantorovich duality: Verify primal-dual gap ≈ 0
- Hawkes stability: Check branching ratio < 1 for subcritical processes
- Causal identification: Validate backdoor/frontdoor criteria
- Particle filter: Monitor Effective Sample Size (ESS)

### Statistical Properties
- VAR stationarity: Check eigenvalues of companion matrix
- SVAR identification: Verify order conditions satisfied
- Synthetic control: Pre-treatment fit quality (RMSPE)
- RDD: Bandwidth sensitivity analysis

### Reproducibility
All examples include `np.random.seed()` for deterministic results.

---

## 🚧 Extensions & Future Work

While GeoBotv1 is complete for core forecasting tasks, potential extensions include:

- **Game-Theoretic Modules**: Strategic interaction between actors
- **Spatial Statistics**: Geographic contagion with distance decay
- **Network Effects**: Centrality measures, community detection
- **Deep Learning**: Attention mechanisms for text-to-risk pipelines
- **Real-Time Data Feeds**: API integrations for live intelligence
- **Uncertainty Quantification**: Conformal prediction, calibration
- **Ensemble Methods**: Model averaging across multiple frameworks

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

GeoBotv1 builds on decades of research in:
- Causal inference (Judea Pearl, Donald Rubin)
- Optimal transport theory (Cédric Villani, Leonid Kantorovich)
- Econometrics (Christopher Sims, Alberto Abadie)
- Point processes (Alan Hawkes, Daryl Daley)
- Bayesian statistics (Andrew Gelman, Michael Jordan)
- Stochastic calculus (Kiyosi Itô, Bernt Øksendal)

---

## 📧 Contact & Support

For questions, issues, or collaboration:
- **Issues**: [GitHub Issues](https://github.com/your-org/AIGEOPOLITICAL/issues)
- **Documentation**: [Full documentation](https://your-org.github.io/AIGEOPOLITICAL)
- **Citation**: If you use GeoBotv1 in research, please cite this repository

---

<div align="center">

**GeoBotv1** - Where rigorous mathematics meets geopolitical forecasting

*Built with causality, powered by probability, validated by theory*

</div>
