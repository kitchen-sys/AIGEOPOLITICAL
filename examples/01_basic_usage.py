"""
Example 1: Basic Usage of GeoBotv1

This example demonstrates the core components of the framework:
- Creating scenarios
- Building causal graphs
- Running Monte Carlo simulations
- Bayesian belief updating
"""

import numpy as np
import sys
sys.path.append('..')

from geobot.core.scenario import Scenario, ScenarioDistribution
from geobot.models.causal_graph import CausalGraph, StructuralCausalModel
from geobot.simulation.monte_carlo import MonteCarloEngine, SimulationConfig
from geobot.inference.bayesian_engine import BayesianEngine, Prior, Evidence, BeliefUpdater
from scipy import stats


def main():
    print("=" * 80)
    print("GeoBotv1 - Basic Usage Example")
    print("=" * 80)

    # 1. Create a simple scenario
    print("\n1. Creating a geopolitical scenario...")
    scenario = Scenario(
        name="baseline_scenario",
        features={
            'military_tension': np.array([0.5]),
            'economic_sanctions': np.array([0.3]),
            'diplomatic_relations': np.array([0.6]),
        },
        probability=1.0
    )
    print(f"   Created scenario: {scenario.name}")
    print(f"   Features: {list(scenario.features.keys())}")

    # 2. Build a causal graph
    print("\n2. Building causal graph...")
    causal_graph = CausalGraph(name="geopolitical_dag")

    # Add nodes
    causal_graph.add_node('sanctions', node_type='policy')
    causal_graph.add_node('tension', node_type='state')
    causal_graph.add_node('conflict_risk', node_type='outcome')

    # Add causal edges
    causal_graph.add_edge('sanctions', 'tension',
                         strength=0.7,
                         mechanism="Sanctions increase military tension")
    causal_graph.add_edge('tension', 'conflict_risk',
                         strength=0.8,
                         mechanism="Tension increases conflict probability")

    print(f"   Created graph with {len(causal_graph.graph.nodes)} nodes")
    print(f"   Causal relationships: sanctions -> tension -> conflict_risk")

    # 3. Run Monte Carlo simulation
    print("\n3. Running Monte Carlo simulation...")
    config = SimulationConfig(n_simulations=100, time_horizon=50)
    mc_engine = MonteCarloEngine(config)

    def transition_fn(state, t, noise):
        # Simple dynamics
        new_state = {}
        new_state['tension'] = state.get('tension', 0.5) + \
                              0.1 * state.get('sanctions', 0) + \
                              noise.get('tension', 0)
        new_state['conflict_risk'] = 0.5 * new_state['tension'] + \
                                    noise.get('conflict_risk', 0)
        # Clip values
        new_state['tension'] = np.clip(new_state['tension'], 0, 1)
        new_state['conflict_risk'] = np.clip(new_state['conflict_risk'], 0, 1)
        return new_state

    def noise_fn(t):
        return {
            'tension': np.random.normal(0, 0.05),
            'conflict_risk': np.random.normal(0, 0.05)
        }

    initial_state = {'tension': 0.3, 'sanctions': 0.2, 'conflict_risk': 0.1}
    trajectories = mc_engine.run_basic_simulation(initial_state, transition_fn, noise_fn)

    # Compute statistics
    stats = mc_engine.compute_statistics(trajectories)
    print(f"   Ran {config.n_simulations} simulations")
    print(f"   Final conflict risk (mean): {stats['conflict_risk']['mean'][-1]:.3f}")
    print(f"   Final conflict risk (95% CI): [{stats['conflict_risk']['q5'][-1]:.3f}, {stats['conflict_risk']['q95'][-1]:.3f}]")

    # 4. Bayesian belief updating
    print("\n4. Bayesian belief updating...")
    updater = BeliefUpdater()

    # Initialize belief about conflict risk
    updater.initialize_belief(
        name='conflict_risk',
        prior_mean=0.3,
        prior_std=0.1,
        belief_type='probability'
    )

    # Receive intelligence report suggesting higher risk
    print("   Received intelligence: conflict risk = 0.6 (reliability: 0.7)")
    posterior = updater.update_from_intelligence(
        belief='conflict_risk',
        observation=0.6,
        reliability=0.7
    )

    print(f"   Updated belief - Mean: {posterior['mean']:.3f}, Std: {posterior['std']:.3f}")
    print(f"   95% Credible Interval: [{posterior['q5']:.3f}, {posterior['q95']:.3f}]")

    # 5. Probability of high risk
    prob_high_risk = updater.get_belief_probability(
        'conflict_risk',
        threshold=0.5,
        direction='greater'
    )
    print(f"   Probability of high risk (>0.5): {prob_high_risk:.3f}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
