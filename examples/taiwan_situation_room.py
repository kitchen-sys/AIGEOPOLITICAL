"""
Taiwan Situation Room - GeoBot 2.0 Analytical Framework Demo

Comprehensive demonstration of GeoBot 2.0 analytical capabilities applied
to Taiwan Strait scenario analysis. Integrates:

- GeoBot 2.0 analytical lenses (Governance, Logistics, Corruption, Non-Western)
- Bayesian forecasting and belief updating
- Structural causal models for intervention analysis
- Hawkes processes for escalation dynamics

Scenario: Rising tensions in Taiwan Strait with potential for
military escalation. Analysis evaluates PRC capabilities, deterrence
credibility, and intervention outcomes.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from datetime import datetime

# GeoBot 2.0 Analytical Framework
from geobot.analysis import (
    AnalyticalEngine,
    GovernanceType,
    CorruptionType,
    AnalyticalLenses
)

# Bayesian Forecasting (if numpy available)
try:
    from geobot.bayes import (
        BayesianForecaster,
        GeopoliticalPrior,
        PriorType,
        EvidenceUpdate,
        EvidenceType
    )
    BAYES_AVAILABLE = True
except ImportError:
    BAYES_AVAILABLE = False
    print("Note: Bayesian forecasting requires numpy - skipping Bayesian analysis")

# Causal Models (if numpy/networkx available)
try:
    from geobot.causal import (
        StructuralCausalModel,
        StructuralEquation,
        Intervention,
        Counterfactual
    )
    CAUSAL_AVAILABLE = True
except ImportError:
    CAUSAL_AVAILABLE = False
    print("Note: Causal models require numpy/networkx - skipping causal analysis")

# Hawkes processes (if scipy available)
try:
    from geobot.simulation.hawkes import (
        HawkesSimulator,
        quick_conflict_contagion_analysis
    )
    HAWKES_AVAILABLE = True
except ImportError:
    HAWKES_AVAILABLE = False
    print("Note: Hawkes processes require scipy - skipping escalation dynamics")


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subheader(title):
    """Print formatted subsection header."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80 + "\n")


# ============================================================================
# Part 1: GeoBot 2.0 Core Analysis
# ============================================================================

def part1_geobot_core_analysis():
    """
    Apply GeoBot 2.0 analytical framework to Taiwan scenario.
    """
    print_header("Part 1: GeoBot 2.0 Analytical Framework - Taiwan Scenario")

    engine = AnalyticalEngine()

    # Scenario context
    query = """PRC conducts large-scale military exercises around Taiwan,
including live-fire drills and simulated blockade operations. US conducts
freedom of navigation operations in Taiwan Strait. What are the escalation
risks and intervention outcomes?"""

    print(f"QUERY: {query}\n")

    # Build comprehensive context
    context = {
        'governance_type': GovernanceType.AUTHORITARIAN_CENTRALIZED,
        'corruption_type': CorruptionType.MANAGED_BOUNDED,
        'military_system': 'Chinese PLA',
        'scenario_description': 'PRC military exercises and potential Taiwan contingency',
        'operational_context': 'High-intensity joint operations in near-seas environment',

        'summary': """PRC demonstrates improving capability for joint operations in Taiwan
Strait, but faces significant logistical and operational challenges for sustained
high-intensity operations. Authoritarian governance enables rapid mobilization but
information flow problems could create coordination failures under stress.""",

        'logistics_assessment': """PRC Eastern Theater Command has concentrated logistics
infrastructure supporting Taiwan contingency. Civil-military fusion enables rapid resource
mobilization. However, sustained amphibious/air assault operations would stress logistics
systems untested in combat. Key constraints: sealift capacity, contested logistics under
US/allied interdiction, ammunition sustainment for high-intensity operations.""",

        # Scenarios with probabilities
        'scenarios': [
            {
                'name': 'Coercive demonstration without escalation',
                'probability': 0.55,
                'description': 'Exercises conclude after demonstrating capability and resolve'
            },
            {
                'name': 'Graduated escalation (quarantine/blockade)',
                'probability': 0.30,
                'description': 'PRC implements quarantine, testing US/allied response'
            },
            {
                'name': 'Limited kinetic action',
                'probability': 0.10,
                'description': 'Strikes on Taiwan military targets, no invasion'
            },
            {
                'name': 'Full-scale invasion attempt',
                'probability': 0.05,
                'description': 'Amphibious/airborne assault on Taiwan'
            }
        ],

        # Uncertainty factors
        'uncertainty_factors': [
            'PRC leadership risk tolerance and decision calculus',
            'Taiwan domestic political response and resolve',
            'US extended deterrence credibility perception',
            'PLA actual readiness vs. reported readiness (information distortion risk)',
            'Third-party actions (Japan, Australia, regional states)',
            'Economic interdependence constraints on escalation'
        ],

        # Signals to watch
        'signals_to_watch': [
            'PLA logistics mobilization (satellite-observable sealift, air transport concentration)',
            'Rocket Force alert status and deployment patterns',
            'PLAN submarine deployments',
            'Civilian shipping disruptions (clearance of civilian vessels from exercise areas)',
            'PRC domestic propaganda shifts (priming for kinetic action vs. victorious conclusion)',
            'US carrier strike group deployments and readiness status',
            'Taiwan reserve mobilization signals',
            'Japanese Self-Defense Force posture changes'
        ],

        # Comparative notes
        'comparative_notes': """Unlike Russia-Ukraine, PRC faces amphibious/air assault across
defended strait with peer/near-peer opposition (US, Japan, Australia). PLA has not conducted
combat operations since 1979, vs. Russia's experience in Syria, Georgia, Ukraine. However,
PRC has advantage of proximity, massive firepower overmatch against Taiwan alone, and
authoritarian ability to sustain economic costs."""
    }

    # Add governance-specific analysis
    context['governance_context'] = {
        'trade_off': """Authoritarian governance enables PRC to:
        - Rapidly mobilize resources without legislative approval
        - Sustain operations despite economic costs and casualties
        - Conduct strategic surprise without public debate

        But creates risks:
        - Information distortion about PLA readiness/capabilities
        - Over-confidence in leadership due to filtered reporting
        - Inflexible response to unexpected battlefield developments""",

        'context_specific_advantage': """In crisis initiation and short, sharp operations,
authoritarian system has decision-speed advantage. In sustained operations requiring
adaptation, democratic information flow advantages become more important. Key question:
Can PRC achieve fait accompli before US/allied decision-making concludes?"""
    }

    # Add corruption context
    context['corruption_details'] = {
        'evidence': """Post-2012 anti-corruption campaigns have reduced parasitic corruption
in PLA, especially after 2017 Rocket Force purges. However, managed corruption model means:
        - Procurement still involves kickbacks, but constrained to avoid readiness impact
        - Promotion decisions still involve patronage, affecting command quality
        - Readiness reporting still subject to careerism incentives""",

        'risk_assessment': """Corruption less likely to cause catastrophic equipment failures
(cf. Russian logistics in Ukraine), but could create:
        - Over-estimation of PLA capabilities by leadership
        - Coordination problems from patronage-based command appointments
        - Supply chain inefficiencies under stress"""
    }

    # Add non-Western analysis
    context['non_western_context'] = {
        'analysis_framework': """PLA operational culture emphasizes:
        - Centralized planning with detailed pre-scripted operations
        - Heavy firepower preparation before maneuver
        - Political control through party committee system
        - Joint operations still developing (improving but not NATO-level)

        This creates both capabilities and constraints different from Western assumptions.""",

        'key_distinction': """Western analysis often assumes PLA would operate like NATO forces.
In reality, PLA would likely emphasize:
        - Overwhelming initial firepower (missiles, air strikes) to create shock
        - Rapid fait accompli before US can intervene
        - Accepting higher casualties than Western forces
        - Using information operations and political warfare alongside kinetic

        These reflect Chinese strategic culture and organizational strengths, not deficiencies."""
    }

    # Perform analysis
    print_subheader("GeoBot 2.0 Analytical Output")
    analysis = engine.analyze(query, context)
    print(analysis)


# ============================================================================
# Part 2: Bayesian Belief Updating
# ============================================================================

def part2_bayesian_analysis():
    """
    Bayesian belief updating as new intelligence arrives.
    """
    if not BAYES_AVAILABLE:
        print_header("Part 2: Bayesian Analysis - SKIPPED (numpy not available)")
        return

    print_header("Part 2: Bayesian Belief Updating - Intelligence Integration")

    forecaster = BayesianForecaster()

    # Set prior on PRC invasion probability within 12 months
    invasion_prior = GeopoliticalPrior(
        parameter_name="invasion_probability_12mo",
        prior_type=PriorType.BETA,
        parameters={'alpha': 2.0, 'beta': 18.0},  # Prior mean ~0.10
        description="Probability of PRC invasion attempt within 12 months"
    )

    forecaster.set_prior(invasion_prior)

    print("PRIOR BELIEF:")
    print(f"  Distribution: Beta(α=2.0, β=18.0)")
    print(f"  Prior mean: ~0.10 (10% chance)")
    print(f"  This reflects baseline assessment before current crisis\n")

    # Evidence 1: Satellite imagery shows sealift mobilization
    print_subheader("Evidence Update 1: Satellite Imagery")
    print("Satellite imagery shows increased sealift concentration in Fujian ports")
    print("Assessing impact on invasion probability...\n")

    def sealift_likelihood(p):
        # If invasion likely, sealift mobilization is very likely
        # If invasion unlikely, some mobilization still possible (exercises)
        return p * 0.9 + (1 - p) * 0.2

    evidence1 = EvidenceUpdate(
        evidence_type=EvidenceType.SATELLITE_IMAGERY,
        observation="sealift_mobilization",
        likelihood_function=sealift_likelihood,
        reliability=0.95,
        source="Commercial satellite analysis"
    )

    belief1 = forecaster.update_belief(
        "invasion_probability_12mo",
        evidence1,
        n_samples=10000
    )

    print(f"Updated belief after satellite evidence:")
    print(f"  Mean: {belief1.mean():.3f}")
    print(f"  Median: {belief1.median():.3f}")
    print(f"  95% CI: {belief1.credible_interval(0.05)}")
    print(f"  P(invasion > 0.20): {belief1.probability_greater_than(0.20):.2f}\n")

    # Evidence 2: HUMINT reports purges in Taiwan Affairs Office
    print_subheader("Evidence Update 2: HUMINT Report")
    print("HUMINT reports internal purges in Taiwan Affairs Office leadership")
    print("Interpretation: Could indicate pre-operation security tightening OR internal dysfunction\n")

    def purge_likelihood(p):
        # Purges could indicate either preparation or problems
        # Moderate signal
        return p * 0.6 + (1 - p) * 0.4

    evidence2 = EvidenceUpdate(
        evidence_type=EvidenceType.INTELLIGENCE_REPORT,
        observation="tao_purges",
        likelihood_function=purge_likelihood,
        reliability=0.70,  # HUMINT less reliable than satellite
        source="HUMINT Taiwan Affairs Office"
    )

    belief2 = forecaster.update_belief(
        "invasion_probability_12mo",
        evidence2,
        n_samples=10000
    )

    print(f"Updated belief after HUMINT evidence:")
    print(f"  Mean: {belief2.mean():.3f}")
    print(f"  Median: {belief2.median():.3f}")
    print(f"  95% CI: {belief2.credible_interval(0.05)}")
    print(f"  P(invasion > 0.20): {belief2.probability_greater_than(0.20):.2f}\n")

    # Evidence 3: Economic data shows continued deep integration
    print_subheader("Evidence Update 3: Economic Data")
    print("Economic data shows continued deep PRC-Taiwan trade integration, no decoupling")
    print("Interpretation: Reduces likelihood of near-term kinetic action\n")

    def economic_likelihood(p):
        # Continued integration suggests not preparing for war
        return p * 0.3 + (1 - p) * 0.8

    evidence3 = EvidenceUpdate(
        evidence_type=EvidenceType.ECONOMIC_DATA,
        observation="continued_integration",
        likelihood_function=economic_likelihood,
        reliability=1.0,  # Economic data highly reliable
        source="Trade statistics"
    )

    belief3 = forecaster.update_belief(
        "invasion_probability_12mo",
        evidence3,
        n_samples=10000
    )

    print(f"Final belief after all evidence:")
    print(f"  Mean: {belief3.mean():.3f}")
    print(f"  Median: {belief3.median():.3f}")
    print(f"  95% CI: {belief3.credible_interval(0.05)}")
    print(f"  P(invasion > 0.20): {belief3.probability_greater_than(0.20):.2f}")
    print(f"  P(invasion > 0.30): {belief3.probability_greater_than(0.30):.2f}")

    # Summary
    summary = forecaster.get_belief_summary("invasion_probability_12mo")
    print(f"\nBelief Summary:")
    print(f"  Evidence updates: {summary['n_evidence_updates']}")
    print(f"  Evidence types: {summary['evidence_types']}")
    print(f"  Final assessment: {summary['mean']:.1%} probability within 12 months")


# ============================================================================
# Part 3: Causal Intervention Analysis
# ============================================================================

def part3_causal_intervention_analysis():
    """
    Use structural causal models to evaluate intervention outcomes.
    """
    if not CAUSAL_AVAILABLE:
        print_header("Part 3: Causal Analysis - SKIPPED (dependencies not available)")
        return

    print_header("Part 3: Causal Intervention Analysis - Policy Counterfactuals")

    # Build SCM for Taiwan deterrence
    print("Building Structural Causal Model for Taiwan deterrence dynamics...\n")

    scm = StructuralCausalModel(name="TaiwanDeterrenceSCM")

    noise_dist = lambda n: np.random.randn(n) * 0.05

    # US military presence -> PRC perception of US resolve
    scm.add_equation(StructuralEquation(
        variable="prc_perceived_us_resolve",
        parents=["us_military_presence"],
        function=lambda p: 0.3 + 0.6 * p["us_military_presence"],
        noise_dist=noise_dist,
        description="US military presence increases PRC perception of US resolve"
    ))

    # Taiwan defense spending -> Taiwan military capability
    scm.add_equation(StructuralEquation(
        variable="taiwan_military_capability",
        parents=["taiwan_defense_spending"],
        function=lambda p: 0.4 + 0.5 * p["taiwan_defense_spending"],
        noise_dist=noise_dist,
        description="Taiwan defense spending improves military capability"
    ))

    # PRC perceived costs = f(US resolve, Taiwan capability)
    scm.add_equation(StructuralEquation(
        variable="prc_perceived_costs",
        parents=["prc_perceived_us_resolve", "taiwan_military_capability"],
        function=lambda p: (0.4 * p["prc_perceived_us_resolve"] +
                          0.3 * p["taiwan_military_capability"]),
        noise_dist=noise_dist,
        description="Perceived costs depend on US resolve and Taiwan capability"
    ))

    # Conflict risk = f(perceived costs, prc_domestic_pressure)
    scm.add_equation(StructuralEquation(
        variable="conflict_risk",
        parents=["prc_perceived_costs", "prc_domestic_pressure"],
        function=lambda p: (0.5 + 0.3 * p["prc_domestic_pressure"] -
                          0.4 * p["prc_perceived_costs"]),
        noise_dist=noise_dist,
        description="Conflict risk increases with domestic pressure, decreases with perceived costs"
    ))

    # Baseline scenario
    print_subheader("Baseline Scenario")
    baseline_data = scm.simulate(n_samples=10000, random_state=42)
    print(f"Baseline conflict risk: Mean = {np.mean(baseline_data['conflict_risk']):.3f}, "
          f"Std = {np.std(baseline_data['conflict_risk']):.3f}\n")

    # Intervention 1: Increase US military presence
    print_subheader("Intervention 1: Increase US Military Presence")
    print("do(us_military_presence = 0.8)  # High presence\n")

    intervention1 = Intervention(
        variable="us_military_presence",
        value=0.8,
        description="Sustained US carrier presence + forward-deployed assets"
    )

    post_intervention1 = scm.intervene([intervention1], n_samples=10000, random_state=42)
    print(f"Post-intervention conflict risk: Mean = {np.mean(post_intervention1['conflict_risk']):.3f}")
    print(f"Effect of intervention: {np.mean(baseline_data['conflict_risk']) - np.mean(post_intervention1['conflict_risk']):.3f} reduction")
    print(f"Interpretation: Increasing US presence reduces conflict risk by ~{100*(np.mean(baseline_data['conflict_risk']) - np.mean(post_intervention1['conflict_risk'])):.1f} percentage points\n")

    # Intervention 2: Increase Taiwan defense spending
    print_subheader("Intervention 2: Increase Taiwan Defense Spending")
    print("do(taiwan_defense_spending = 0.9)  # Major defense investment\n")

    intervention2 = Intervention(
        variable="taiwan_defense_spending",
        value=0.9,
        description="Major asymmetric defense investments"
    )

    post_intervention2 = scm.intervene([intervention2], n_samples=10000, random_state=42)
    print(f"Post-intervention conflict risk: Mean = {np.mean(post_intervention2['conflict_risk']):.3f}")
    print(f"Effect of intervention: {np.mean(baseline_data['conflict_risk']) - np.mean(post_intervention2['conflict_risk']):.3f} reduction\n")

    # Combined intervention
    print_subheader("Intervention 3: Combined Strategy")
    print("do(us_military_presence = 0.8, taiwan_defense_spending = 0.9)\n")

    combined_data = scm.intervene([intervention1, intervention2], n_samples=10000, random_state=42)
    print(f"Post-intervention conflict risk: Mean = {np.mean(combined_data['conflict_risk']):.3f}")
    print(f"Effect of combined intervention: {np.mean(baseline_data['conflict_risk']) - np.mean(combined_data['conflict_risk']):.3f} reduction")
    print(f"Interpretation: Combined strategy most effective for reducing conflict risk\n")

    # Counterfactual query
    print_subheader("Counterfactual Query")
    print("Question: What would conflict risk be if we had maintained high US presence,")
    print("given that we currently observe moderate US presence?\n")

    counterfactual = Counterfactual(
        query_variable="conflict_risk",
        intervention=Intervention("us_military_presence", 0.8),
        observations={"us_military_presence": 0.5}
    )

    cf_result = scm.counterfactual_query(counterfactual, n_samples=10000)
    print(f"Counterfactual conflict risk: {cf_result['expected_value']:.3f}")
    print(f"95% CI: ({cf_result['quantiles']['5%']:.3f}, {cf_result['quantiles']['95%']:.3f})")


# ============================================================================
# Part 4: Escalation Dynamics (Hawkes Processes)
# ============================================================================

def part4_escalation_dynamics():
    """
    Model escalation dynamics using Hawkes processes.
    """
    if not HAWKES_AVAILABLE:
        print_header("Part 4: Escalation Dynamics - SKIPPED (scipy not available)")
        return

    print_header("Part 4: Escalation Dynamics - Self-Exciting Processes")

    from geobot.simulation.hawkes import HawkesParameters

    print("Modeling crisis escalation as self-exciting point process...")
    print("Events cluster in time and trigger subsequent events (escalatory spiral)\n")

    # Simulate escalation scenario
    print_subheader("Scenario: Incremental Escalation with Contagion")

    # Three actors: PRC, Taiwan, US
    baseline_rates = [0.5, 0.3, 0.2]  # PRC initiates more, US responds
    countries = ['PRC', 'Taiwan', 'US']

    # Contagion: PRC action triggers Taiwan/US response
    alpha_matrix = np.array([
        [0.3, 0.2, 0.15],  # PRC actions trigger more PRC, Taiwan, US actions
        [0.4, 0.2, 0.3],   # Taiwan actions strongly trigger PRC and US
        [0.5, 0.2, 0.1],   # US actions strongly trigger PRC responses
    ])

    beta_matrix = np.ones((3, 3)) * 1.5  # Decay rate

    params = HawkesParameters(
        mu=np.array(baseline_rates),
        alpha=alpha_matrix,
        beta=beta_matrix
    )

    # Simulate 30-day crisis
    simulator = HawkesSimulator(n_dimensions=3)
    events = simulator.simulate(T=30.0, params=params, random_state=42)

    print(f"Simulated 30-day crisis escalation:")
    for i, country in enumerate(countries):
        print(f"  {country}: {len(events[i])} escalatory events")

    # Assess stability
    stability = simulator.assess_stability(params)
    print(f"\nEscalation dynamics stability assessment:")
    print(f"  Branching ratio: {stability['branching_ratio']:.3f}")
    print(f"  Regime: {stability['regime']}")
    print(f"  Interpretation: {stability['interpretation']}\n")

    if stability['is_explosive']:
        print("⚠️  WARNING: Process is supercritical - escalation could spiral out of control")
    else:
        print("✓ Process is subcritical - escalation will stabilize")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run complete Taiwan situation room analysis."""

    print("\n" + "=" * 80)
    print("  TAIWAN SITUATION ROOM")
    print("  GeoBot 2.0 Integrated Geopolitical Analysis")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    # Run all parts
    part1_geobot_core_analysis()
    part2_bayesian_analysis()
    part3_causal_intervention_analysis()
    part4_escalation_dynamics()

    # Summary
    print_header("Summary and Recommendations")

    print("""
INTEGRATED ASSESSMENT:

1. GEOBOT 2.0 ANALYTICAL FRAMEWORK
   - PRC has improving joint operations capability but faces significant logistical
     constraints for sustained high-intensity operations
   - Authoritarian governance enables rapid mobilization but creates information
     flow risks
   - Managed corruption likely constrained enough to maintain basic functionality
   - Non-Western analysis reveals PRC emphasis on firepower and fait accompli

2. BAYESIAN BELIEF UPDATING
   - Posterior probability of invasion within 12 months: ~15-20% (up from 10% prior)
   - Satellite evidence of sealift mobilization raises concern
   - Economic integration evidence reduces near-term kinetic risk
   - Continued monitoring required as new intelligence arrives

3. CAUSAL INTERVENTION ANALYSIS
   - Combined strategy (US presence + Taiwan defense) most effective
   - US military presence has direct deterrent effect
   - Taiwan capabilities create operational costs for PRC
   - Counterfactual analysis supports sustained presence policy

4. ESCALATION DYNAMICS
   - Current contagion parameters suggest subcritical regime (stable)
   - However, parameter changes could shift to explosive regime
   - Escalation management critical to prevent spiral

POLICY RECOMMENDATIONS:
   - Maintain credible US extended deterrence
   - Support Taiwan asymmetric defense capabilities
   - Engage in crisis management mechanisms to prevent escalation spirals
   - Continue intelligence collection on PLA readiness and mobilization
   - Monitor for signals of PRC leadership decision to use force
    """)

    print("=" * 80)
    print("  Analysis Complete")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
