# Getting Started with GeoBotv1

This guide will help you get up and running with GeoBotv1.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/AIGEOPOLITICAL.git
cd AIGEOPOLITICAL
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

### 4. Install optional dependencies
For full data ingestion capabilities:
```bash
pip install pypdf pdfplumber beautifulsoup4 newspaper3k trafilatura feedparser
```

## Quick Start

### Example 1: Basic Scenario Analysis

```python
from geobot.core.scenario import Scenario
from geobot.simulation.monte_carlo import MonteCarloEngine, SimulationConfig
import numpy as np

# Create a scenario
scenario = Scenario(
    name="tension_scenario",
    features={
        'military_tension': np.array([0.7]),
        'diplomatic_relations': np.array([0.3]),
    }
)

# Run Monte Carlo simulation
config = SimulationConfig(n_simulations=1000, time_horizon=50)
engine = MonteCarloEngine(config)

# Define dynamics
def transition_fn(state, t, noise):
    new_state = {}
    new_state['tension'] = state.get('tension', 0.5) + noise.get('tension', 0)
    return new_state

def noise_fn(t):
    return {'tension': np.random.normal(0, 0.05)}

initial_state = {'tension': 0.3}
trajectories = engine.run_basic_simulation(initial_state, transition_fn, noise_fn)

# Analyze results
stats = engine.compute_statistics(trajectories)
print(f"Mean tension at end: {stats['tension']['mean'][-1]:.3f}")
```

### Example 2: Causal Inference

```python
from geobot.models.causal_graph import CausalGraph

# Build causal graph
graph = CausalGraph(name="conflict_model")

# Add variables
graph.add_node('sanctions', node_type='policy')
graph.add_node('tension', node_type='state')
graph.add_node('conflict', node_type='outcome')

# Define causal relationships
graph.add_edge('sanctions', 'tension',
              strength=0.7,
              mechanism="Sanctions increase tension")
graph.add_edge('tension', 'conflict',
              strength=0.8,
              mechanism="Tension leads to conflict")

# Visualize
graph.visualize('causal_graph.png')
```

### Example 3: Intervention Simulation

```python
from geobot.inference.do_calculus import InterventionSimulator
from geobot.models.causal_graph import StructuralCausalModel

# Create SCM with your causal graph
scm = StructuralCausalModel(graph)

# Define structural equations
# (See examples/03_intervention_simulation.py for full details)

# Create simulator
simulator = InterventionSimulator(scm)

# Simulate intervention
result = simulator.simulate_intervention(
    intervention={'sanctions': 0.8},
    n_samples=1000,
    outcomes=['conflict']
)

print(f"Expected conflict under sanctions: {result['conflict'].mean():.3f}")
```

### Example 4: Bayesian Belief Updating

```python
from geobot.inference.bayesian_engine import BeliefUpdater

# Create updater
updater = BeliefUpdater()

# Initialize belief
updater.initialize_belief(
    name='conflict_risk',
    prior_mean=0.3,
    prior_std=0.1,
    belief_type='probability'
)

# Update with new intelligence
posterior = updater.update_from_intelligence(
    belief='conflict_risk',
    observation=0.6,
    reliability=0.8
)

print(f"Updated risk: {posterior['mean']:.3f} ± {posterior['std']:.3f}")
```

### Example 5: PDF Processing

```python
from geobot.data_ingestion.pdf_reader import PDFProcessor

# Create processor
processor = PDFProcessor()

# Process document
result = processor.extract_intelligence('report.pdf')

print(f"Risk Level: {result['intelligence']['risk_level']}")
print(f"Countries: {result['intelligence']['mentioned_countries']}")
print(f"Conflict Indicators: {result['intelligence']['conflict_indicators']}")
```

### Example 6: Web Scraping

```python
from geobot.data_ingestion.web_scraper import ArticleExtractor

# Create extractor
extractor = ArticleExtractor()

# Extract article
article = extractor.extract_article('https://example.com/article')

print(f"Title: {article['title']}")
print(f"Summary: {article['text'][:200]}...")
```

## Running Examples

The `examples/` directory contains comprehensive demonstrations:

```bash
cd examples

# Basic usage
python 01_basic_usage.py

# Data ingestion
python 02_data_ingestion.py

# Intervention simulation
python 03_intervention_simulation.py
```

## Core Concepts

### 1. Scenarios
Scenarios represent geopolitical states with features and probabilities.

### 2. Causal Graphs
DAGs that model causal relationships between variables.

### 3. Structural Causal Models
Mathematical models with functional equations for each variable.

### 4. Monte Carlo Simulation
Stochastic simulation for uncertainty quantification.

### 5. Bayesian Inference
Principled belief updating as new evidence arrives.

### 6. Do-Calculus
Intervention reasoning for "what if" questions.

### 7. Optimal Transport
Measuring distances between probability distributions.

## Next Steps

1. Read the full README.md
2. Explore the examples directory
3. Check out the module documentation
4. Build your own models!

## Need Help?

- Check the examples directory for detailed code
- Review module docstrings for API documentation
- Open an issue on GitHub

## Tips

1. Start with simple models and gradually add complexity
2. Always validate your causal assumptions
3. Use Monte Carlo for uncertainty quantification
4. Combine multiple methods for robust forecasting
5. Document your assumptions and data sources

Happy forecasting!
