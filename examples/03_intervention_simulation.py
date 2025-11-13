"""
Example 3: Intervention Simulation and Counterfactual Analysis

This example demonstrates:
- Policy intervention simulation
- Counterfactual reasoning ("what if" scenarios)
- Comparing multiple interventions
- Optimal intervention finding
"""

import numpy as np
import sys
sys.path.append('..')

from geobot.models.causal_graph import CausalGraph, StructuralCausalModel
from geobot.inference.do_calculus import DoCalculus, InterventionSimulator


def create_geopolitical_scm():
    """Create a structural causal model for geopolitical scenarios."""
    print("\n1. Creating Structural Causal Model...")

    # Create causal graph
    graph = CausalGraph(name="geopolitical_system")

    # Add nodes
    graph.add_node('economic_sanctions', node_type='policy')
    graph.add_node('diplomatic_pressure', node_type='policy')
    graph.add_node('domestic_stability', node_type='state')
    graph.add_node('military_mobilization', node_type='state')
    graph.add_node('conflict_probability', node_type='outcome')

    # Add causal edges
    graph.add_edge('economic_sanctions', 'domestic_stability',
                  strength=-0.6, mechanism="Sanctions reduce stability")
    graph.add_edge('diplomatic_pressure', 'domestic_stability',
                  strength=-0.3, mechanism="Pressure affects stability")
    graph.add_edge('domestic_stability', 'military_mobilization',
                  strength=-0.7, mechanism="Instability drives mobilization")
    graph.add_edge('military_mobilization', 'conflict_probability',
                  strength=0.8, mechanism="Mobilization increases conflict risk")
    graph.add_edge('economic_sanctions', 'conflict_probability',
                  strength=0.4, mechanism="Direct deterrence effect")

    print(f"   Created graph with {len(graph.graph.nodes)} nodes and {len(graph.edges)} edges")

    # Create SCM
    scm = StructuralCausalModel(graph)

    # Define structural equations
    def sanctions_fn(parents, noise):
        return 0.5 + noise  # Baseline policy level

    def pressure_fn(parents, noise):
        return 0.3 + noise

    def stability_fn(parents, noise):
        sanctions = parents.get('economic_sanctions', np.zeros(1))[0]
        pressure = parents.get('diplomatic_pressure', np.zeros(1))[0]
        return np.clip(0.7 - 0.6 * sanctions - 0.3 * pressure + noise, 0, 1)

    def mobilization_fn(parents, noise):
        stability = parents.get('domestic_stability', np.zeros(1))[0]
        return np.clip(0.3 - 0.7 * stability + noise, 0, 1)

    def conflict_fn(parents, noise):
        mobilization = parents.get('military_mobilization', np.zeros(1))[0]
        sanctions = parents.get('economic_sanctions', np.zeros(1))[0]
        return np.clip(0.8 * mobilization + 0.4 * sanctions + noise, 0, 1)

    # Set functions
    from scipy import stats
    scm.set_function('economic_sanctions', sanctions_fn, stats.norm(0, 0.1))
    scm.set_function('diplomatic_pressure', pressure_fn, stats.norm(0, 0.1))
    scm.set_function('domestic_stability', stability_fn, stats.norm(0, 0.05))
    scm.set_function('military_mobilization', mobilization_fn, stats.norm(0, 0.05))
    scm.set_function('conflict_probability', conflict_fn, stats.norm(0, 0.05))

    print("   Structural equations defined")

    return scm


def simulate_baseline(simulator):
    """Simulate baseline (no intervention) scenario."""
    print("\n2. Simulating Baseline Scenario...")

    baseline = simulator.simulate_intervention(
        intervention={},
        n_samples=1000,
        outcomes=['conflict_probability']
    )

    conflict_mean = np.mean(baseline['conflict_probability'])
    conflict_std = np.std(baseline['conflict_probability'])

    print(f"   Baseline conflict probability: {conflict_mean:.3f} ± {conflict_std:.3f}")

    return baseline


def simulate_interventions(simulator):
    """Simulate different policy interventions."""
    print("\n3. Simulating Policy Interventions...")

    interventions = [
        {'economic_sanctions': 0.8, 'diplomatic_pressure': 0.3},  # Heavy sanctions
        {'economic_sanctions': 0.3, 'diplomatic_pressure': 0.8},  # Heavy diplomacy
        {'economic_sanctions': 0.6, 'diplomatic_pressure': 0.6},  # Balanced approach
    ]

    intervention_names = [
        "Heavy Sanctions",
        "Heavy Diplomacy",
        "Balanced Approach"
    ]

    results = simulator.compare_interventions(
        interventions,
        outcome='conflict_probability',
        n_samples=1000
    )

    print("\n   Intervention Results:")
    print("   " + "-" * 60)

    for i, name in enumerate(intervention_names):
        result = results[f'intervention_{i}']
        print(f"\n   {name}:")
        print(f"     Mean conflict probability: {result['mean']:.3f}")
        print(f"     Std deviation: {result['std']:.3f}")
        print(f"     95% CI: [{result['q25']:.3f}, {result['q75']:.3f}]")

    return results


def find_optimal_intervention(simulator):
    """Find optimal intervention to minimize conflict."""
    print("\n4. Finding Optimal Intervention...")

    optimal = simulator.optimal_intervention(
        target_var='conflict_probability',
        intervention_vars=['economic_sanctions', 'diplomatic_pressure'],
        intervention_ranges={
            'economic_sanctions': (0.0, 1.0),
            'diplomatic_pressure': (0.0, 1.0)
        },
        objective='minimize',
        n_trials=50,
        n_samples=1000
    )

    print(f"\n   Optimal intervention found:")
    print(f"     Economic Sanctions: {optimal['optimal_intervention']['economic_sanctions']:.3f}")
    print(f"     Diplomatic Pressure: {optimal['optimal_intervention']['diplomatic_pressure']:.3f}")
    print(f"     Expected conflict probability: {optimal['optimal_value']:.3f}")

    return optimal


def counterfactual_analysis(simulator):
    """Perform counterfactual analysis."""
    print("\n5. Counterfactual Analysis...")

    # Observed scenario
    observed = {
        'economic_sanctions': 0.7,
        'diplomatic_pressure': 0.2,
        'domestic_stability': 0.4,
        'military_mobilization': 0.6,
        'conflict_probability': 0.65
    }

    print("\n   Observed scenario:")
    print(f"     Sanctions: {observed['economic_sanctions']}")
    print(f"     Diplomacy: {observed['diplomatic_pressure']}")
    print(f"     Conflict: {observed['conflict_probability']}")

    # Counterfactual: What if we had used more diplomacy?
    counterfactual_intervention = {
        'diplomatic_pressure': 0.8,
        'economic_sanctions': 0.3
    }

    result = simulator.counterfactual_analysis(
        observed=observed,
        intervention=counterfactual_intervention,
        outcome='conflict_probability'
    )

    print("\n   Counterfactual: 'What if we had emphasized diplomacy?'")
    print(f"     Counterfactual conflict: {result['counterfactual_outcome']:.3f}")
    print(f"     Effect of intervention: {result['effect']:.3f}")

    if result['effect'] < 0:
        print(f"     Conclusion: Diplomacy would have REDUCED conflict by {abs(result['effect']):.3f}")
    else:
        print(f"     Conclusion: Diplomacy would have INCREASED conflict by {result['effect']:.3f}")


def main():
    print("=" * 80)
    print("GeoBotv1 - Intervention Simulation & Counterfactual Analysis")
    print("=" * 80)
    print("\nThis example demonstrates answering 'what if' questions:")
    print("- 'What if the U.S. increases sanctions?'")
    print("- 'What if we emphasize diplomacy over sanctions?'")
    print("- 'What is the optimal policy mix?'")
    print("- 'What would have happened if we had acted differently?'")

    # Create SCM
    scm = create_geopolitical_scm()

    # Create intervention simulator
    simulator = InterventionSimulator(scm)

    # Run analyses
    baseline = simulate_baseline(simulator)
    interventions = simulate_interventions(simulator)
    optimal = find_optimal_intervention(simulator)
    counterfactual_analysis(simulator)

    print("\n" + "=" * 80)
    print("Key Insights:")
    print("=" * 80)
    print("\n1. Different interventions have different effects on conflict probability")
    print("2. Optimal policy can be discovered through systematic search")
    print("3. Counterfactual reasoning enables learning from alternative scenarios")
    print("4. Causal models enable principled 'what if' analysis")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
