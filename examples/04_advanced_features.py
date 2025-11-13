"""
Example 4: Advanced Mathematical Features

This example demonstrates the research-grade advanced features:
- Sequential Monte Carlo (particle filtering)
- Variational Inference
- Stochastic Differential Equations (SDEs)
- Gradient-based Optimal Transport
- Kantorovich Duality
- Event Extraction and Database
- Continuous-time dynamics

These features enable measure-theoretic, rigorous forecasting.
"""

import numpy as np
import sys
sys.path.append('..')

from datetime import datetime, timedelta
from scipy import stats

# Advanced inference
from geobot.inference.particle_filter import SequentialMonteCarlo
from geobot.inference.variational_inference import VariationalInference

# SDE solvers
from geobot.simulation.sde_solver import (
    EulerMaruyama,
    Milstein,
    JumpDiffusionProcess,
    GeopoliticalSDE
)

# Advanced optimal transport
from geobot.core.advanced_optimal_transport import (
    KantorovichDuality,
    EntropicOT,
    GradientBasedOT
)

# Event extraction
from geobot.data_ingestion.event_extraction import EventExtractor, EventType
from geobot.data_ingestion.event_database import EventDatabase


def demo_particle_filter():
    """Demonstrate Sequential Monte Carlo / Particle Filter."""
    print("\n" + "="*80)
    print("1. Sequential Monte Carlo (Particle Filter)")
    print("="*80)

    # Define nonlinear dynamics
    def dynamics_fn(x, noise):
        # Nonlinear geopolitical dynamics
        # x[0] = tension, x[1] = stability
        tension = x[0]
        stability = x[1]

        new_tension = tension + 0.1 * (1 - stability) + noise[0]
        new_stability = stability - 0.05 * tension + noise[1]

        return np.array([
            np.clip(new_tension, 0, 1),
            np.clip(new_stability, 0, 1)
        ])

    def observation_fn(y, x):
        # Log-likelihood of observation given state
        # Observe tension with noise
        predicted = x[0]
        return stats.norm.logpdf(y[0], loc=predicted, scale=0.1)

    # Create particle filter
    pf = SequentialMonteCarlo(
        n_particles=500,
        state_dim=2,
        dynamics_fn=dynamics_fn,
        observation_fn=observation_fn
    )

    # Initialize from prior
    pf.initialize_from_prior(lambda: np.array([0.3, 0.7]))

    # Generate synthetic observations
    observations = np.array([
        [0.35], [0.40], [0.45], [0.50], [0.55],
        [0.60], [0.65], [0.70], [0.75], [0.80]
    ])

    print(f"\nRunning particle filter with {pf.n_particles} particles...")
    print("Tracking hidden geopolitical state from noisy observations\n")

    # Filter
    states = pf.filter(observations)

    # Show results
    for i, state in enumerate(states[-5:]):  # Last 5 steps
        mean, cov = pf.get_state_estimate()
        print(f"Step {i+6}: Tension={mean[0]:.3f}±{np.sqrt(cov[0,0]):.3f}, "
              f"Stability={mean[1]:.3f}±{np.sqrt(cov[1,1]):.3f}, "
              f"ESS={state.ess:.1f}")

    print("\n✓ Particle filter successfully tracked nonlinear hidden states!")


def demo_sde_solver():
    """Demonstrate Stochastic Differential Equations."""
    print("\n" + "="*80)
    print("2. Stochastic Differential Equations (Continuous-Time Dynamics)")
    print("="*80)

    # Define SDE: dx = f(x,t)dt + g(x,t)dW
    def drift(x, t):
        # Mean-reverting to 0.5 (long-term stability)
        return 0.2 * (0.5 - x)

    def diffusion(x, t):
        # Volatility increases with tension
        return 0.1 * (1 + x)

    # Create SDE solver
    solver = EulerMaruyama(
        drift=drift,
        diffusion=diffusion,
        x0=np.array([0.7]),  # Start with high tension
        t0=0.0
    )

    print("\nSimulating continuous-time geopolitical tension dynamics...")
    print("SDE: dx = 0.2(0.5 - x)dt + 0.1(1 + x)dW\n")

    # Integrate
    solution = solver.integrate(T=10.0, dt=0.01, n_paths=5)

    # Show statistics
    final_values = solution.x[:, -1, 0]
    print(f"After T=10.0 time units:")
    print(f"  Mean tension: {np.mean(final_values):.3f}")
    print(f"  Std deviation: {np.std(final_values):.3f}")
    print(f"  Min/Max: [{np.min(final_values):.3f}, {np.max(final_values):.3f}]")

    print("\n✓ SDE solver successfully simulated continuous-time dynamics!")


def demo_jump_diffusion():
    """Demonstrate Jump-Diffusion Process."""
    print("\n" + "="*80)
    print("3. Jump-Diffusion Process (Modeling Black Swan Events)")
    print("="*80)

    # Create jump-diffusion process
    jdp = JumpDiffusionProcess(
        drift=0.05,  # Slow drift
        diffusion=0.1,  # Normal volatility
        jump_intensity=0.5,  # 0.5 jumps per unit time (on average)
        jump_mean=-0.2,  # Negative jumps (crises)
        jump_std=0.1,
        x0=np.array([0.5])
    )

    print("\nSimulating conflict escalation with discrete shock events...")
    print("Model: Continuous diffusion + Poisson jumps (λ=0.5, μ=-0.2)\n")

    # Simulate
    solution = jdp.simulate(T=20.0, dt=0.1, n_paths=3)

    # Count jumps (approximately)
    for path in range(3):
        # Detect jumps as large changes
        diffs = np.diff(solution.x[path, :, 0])
        n_jumps = np.sum(np.abs(diffs) > 0.15)
        final_value = solution.x[path, -1, 0]
        print(f"Path {path+1}: {n_jumps} jumps detected, Final value: {final_value:.3f}")

    print("\n✓ Jump-diffusion successfully modeled rare shock events!")


def demo_kantorovich_duality():
    """Demonstrate Kantorovich Duality."""
    print("\n" + "="*80)
    print("4. Kantorovich Duality (Optimal Transport Theory)")
    print("="*80)

    # Create two distributions (scenarios)
    n, m = 10, 10
    mu = np.ones(n) / n  # Uniform source
    nu = np.ones(m) / m  # Uniform target

    # Cost matrix (Euclidean distance)
    X_source = np.random.rand(n, 2)
    X_target = np.random.rand(m, 2) + np.array([0.5, 0.5])  # Shifted
    from scipy.spatial.distance import cdist
    C = cdist(X_source, X_target, metric='sqeuclidean')

    # Solve primal and dual
    kantorovich = KantorovichDuality()

    print("\nComputing optimal transport between two geopolitical scenarios...")
    print(f"Source: {n} points, Target: {m} points\n")

    # Primal solution
    coupling, primal_cost = kantorovich.solve_primal(mu, nu, C, method='emd')
    print(f"Primal optimal cost: {primal_cost:.6f}")

    # Dual solution
    f, g, dual_value = kantorovich.solve_dual(mu, nu, C, max_iter=100)
    print(f"Dual optimal value: {dual_value:.6f}")

    # Verify duality gap
    gap = kantorovich.verify_duality_gap(mu, nu, C)
    print(f"Duality gap: {gap:.8f} (should be ≈ 0)")

    if abs(gap) < 1e-4:
        print("\n✓ Strong duality verified! Primal = Dual")
    else:
        print("\n⚠ Duality gap present (numerical approximation)")


def demo_entropic_ot():
    """Demonstrate Entropic Optimal Transport (Sinkhorn)."""
    print("\n" + "="*80)
    print("5. Entropic Optimal Transport (Sinkhorn Algorithm)")
    print("="*80)

    # Create distributions
    n, m = 20, 20
    mu = np.random.dirichlet(np.ones(n))  # Random distribution
    nu = np.random.dirichlet(np.ones(m))

    # Cost matrix
    X = np.random.rand(n, 2)
    Y = np.random.rand(m, 2)
    from scipy.spatial.distance import cdist
    C = cdist(X, Y, metric='euclidean')

    # Entropic OT with different regularization
    epsilons = [0.01, 0.05, 0.1]

    print("\nComparing regularization levels for Sinkhorn algorithm...\n")

    for eps in epsilons:
        eot = EntropicOT(epsilon=eps)
        coupling, cost = eot.sinkhorn(mu, nu, C, max_iter=500)

        print(f"ε = {eps:0.2f}: Cost = {cost:.6f}, "
              f"Entropy = {-np.sum(coupling * np.log(coupling + 1e-10)):.4f}")

    print("\n✓ Entropic OT computed with fast Sinkhorn iterations!")


def demo_event_extraction():
    """Demonstrate Event Extraction Pipeline."""
    print("\n" + "="*80)
    print("6. Structured Event Extraction from Intelligence")
    print("="*80)

    # Sample intelligence text
    intelligence_text = """
    On March 15, 2024, tensions escalated between the United States and China
    following a major military mobilization in the Taiwan Strait. NATO issued
    a statement expressing concern. Russia announced sanctions on European Union
    member states. India maintained diplomatic neutrality while calling for
    de-escalation talks.

    The United Nations Security Council convened an emergency session on March 16,
    2024. Economic sanctions were proposed against China by the United States,
    but Russia exercised its veto power.
    """

    # Extract events
    extractor = EventExtractor()

    print("\nExtracting structured events from intelligence report...\n")
    print("Input text:")
    print("-" * 60)
    print(intelligence_text[:200] + "...")
    print("-" * 60)

    events = extractor.extract_events(
        intelligence_text,
        source="intel_report_001",
        default_timestamp=datetime(2024, 3, 15)
    )

    print(f"\n✓ Extracted {len(events)} geopolitical events:")
    print()

    for i, event in enumerate(events):
        print(f"Event {i+1}:")
        print(f"  Type: {event.event_type.value}")
        print(f"  Actors: {', '.join(event.actors)}")
        print(f"  Magnitude: {event.magnitude:.2f}")
        print(f"  Timestamp: {event.timestamp.date()}")
        print()

    # Store in database
    print("Storing events in database...")
    with EventDatabase("demo_events.db") as db:
        db.insert_events(events)

        # Query back
        conflict_events = db.query_events(
            event_types=[EventType.CONFLICT, EventType.MILITARY_MOBILIZATION]
        )

        print(f"✓ Database contains {len(conflict_events)} conflict-related events")

    print("\n✓ Event extraction and storage pipeline operational!")


def main():
    """Run all advanced feature demonstrations."""
    print("=" * 80)
    print("GeoBotv1 - Advanced Mathematical Features Demonstration")
    print("=" * 80)
    print("\nThis example showcases research-grade capabilities:")
    print("• Sequential Monte Carlo (particle filtering)")
    print("• Stochastic Differential Equations")
    print("• Jump-Diffusion Processes")
    print("• Kantorovich Duality in Optimal Transport")
    print("• Entropic OT with Sinkhorn")
    print("• Structured Event Extraction")

    # Run demonstrations
    demo_particle_filter()
    demo_sde_solver()
    demo_jump_diffusion()
    demo_kantorovich_duality()
    demo_entropic_ot()
    demo_event_extraction()

    print("\n" + "=" * 80)
    print("All Advanced Features Demonstrated Successfully!")
    print("=" * 80)
    print("\nKey Insights:")
    print("1. Particle filters handle nonlinear/non-Gaussian state estimation")
    print("2. SDEs model continuous-time geopolitical dynamics rigorously")
    print("3. Jump-diffusion captures both gradual change and sudden shocks")
    print("4. Kantorovich duality provides theoretical foundation for OT")
    print("5. Entropic OT enables fast computation via Sinkhorn")
    print("6. Event extraction creates structured data for causal modeling")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
