"""
GeoBot 2.0 Analytical Framework Example

Demonstrates the complete GeoBot 2.0 framework for clinical systems analysis
with geopolitical nuance. Includes:

1. Framework overview
2. Analytical lenses demonstration
3. China Rocket Force purge analysis (example from specification)
4. Governance system comparison
5. Corruption type analysis
6. Non-Western military assessment
"""

import sys
sys.path.insert(0, '..')

from geobot.analysis import (
    AnalyticalEngine,
    GeoBotFramework,
    AnalyticalLenses,
    GovernanceType,
    CorruptionType
)


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def example_1_framework_overview():
    """Demonstrate GeoBot 2.0 framework overview."""
    print_section("Example 1: GeoBot 2.0 Framework Overview")

    framework = GeoBotFramework()
    summary = framework.get_framework_summary()

    print(f"Version: {summary['version']}")
    print(f"Description: {summary['description']}\n")

    print("Core Identity:")
    print(f"  Focus: {summary['identity']['focus']}")
    print(f"  Key Shift: {summary['identity']['key_shift']}\n")

    print("Integration Elements:")
    for element in summary['identity']['integration_elements']:
        print(f"  - {element}")

    print("\nTone:")
    print(summary['tone'])

    print("\nAnalytical Principles:")
    for i, principle in enumerate(summary['principles'], 1):
        print(f"  {i}. {principle}")


def example_2_china_rocket_force_analysis():
    """
    Demonstrate complete analysis using China Rocket Force purge example
    from the GeoBot 2.0 specification.
    """
    print_section("Example 2: China Rocket Force Purge Analysis")

    engine = AnalyticalEngine()

    query = "China removed several top Rocket Force generals. What does this mean?"

    context = {
        'governance_type': GovernanceType.AUTHORITARIAN_CENTRALIZED,
        'corruption_type': CorruptionType.MANAGED_BOUNDED,
        'military_system': 'Chinese PLA',
        'scenario_description': 'Leadership purge in strategic forces',
        'operational_context': 'Strategic nuclear forces readiness',

        'summary': """The purge indicates internal accountability enforcement within strategic forces
command, with mixed implications for near-term readiness and decision coherence.""",

        'logistics_assessment': """Rocket Force maintenance, silo integration, and inventory control
are likely under audit. Purges typically follow discovery of procurement irregularities or
readiness misreporting. However, unlike Russian corruption patterns, Chinese anti-corruption
campaigns since 2012 have successfully constrained (though not eliminated) defense sector
embezzlement. The PLA's civil-military logistics integration provides redundancy that mitigates
some supply chain risks.""",

        'scenarios': [
            {
                'name': 'Routine institutional maintenance',
                'probability': 0.50,
                'description': 'Temporary disruption, return to baseline within 6-12 months'
            },
            {
                'name': 'Deeper procurement crisis',
                'probability': 0.30,
                'description': 'Extended degradation of readiness reporting reliability'
            },
            {
                'name': 'Factional conflict',
                'probability': 0.15,
                'description': 'Prolonged command instability'
            },
            {
                'name': 'Major reorganization',
                'probability': 0.05,
                'description': 'Strategic forces restructure'
            }
        ],

        'uncertainty_factors': [
            'Limited visibility into CCP internal dynamics and audit findings'
        ],

        'signals_to_watch': [
            'Promotion patterns (meritocratic vs. factional indicators)',
            'Training tempo changes (satellite observable)',
            'Procurement contract patterns',
            'Whether purges expand beyond Rocket Force'
        ],

        'comparative_notes': """Russia's similar aerospace purges (2015-2017) resulted in sustained
degradation because underlying corruption was never addressed—only individuals were replaced.
China's systemic anti-corruption infrastructure suggests different trajectory is possible."""
    }

    # Add governance context
    context['governance_context'] = {
        'trade_off': """China gains long-term institutional integrity at the cost of
short-term command continuity.""",
        'context_specific_advantage': """Authoritarian systems can execute rapid leadership
replacement without legislative constraints, allowing faster course correction than
consensus-based systems. However, this creates temporary communication disruption and
institutional memory loss."""
    }

    # Add corruption details
    context['corruption_details'] = {
        'evidence': """Evidence suggests managed corruption model rather than parasitic:
purges indicate the system detected and acted on problems, rather than tolerating systemic decay.
This is structurally different from militaries where corruption goes unaddressed.""",
        'risk_assessment': """Purge-induced fear may cause temporary over-reporting conservatism,
slowing mobilization responses."""
    }

    # Add non-Western context
    context['non_western_context'] = {
        'analysis_framework': """Western analysis often treats purges as pure weakness signals.
In Chinese institutional context, periodic purges are a maintenance mechanism for regime stability.
The question is whether this specific purge reflects routine enforcement or deeper structural crisis.""",
        'key_distinction': """Indicators distinguishing routine vs. crisis:
  - Scope: limited to RF or expanding to other services?
  - Timing: related to specific audit cycle or sudden?
  - Replacements: technocratic or factional?"""
    }

    # Perform analysis
    analysis = engine.analyze(query, context)
    print(analysis)


def example_3_governance_comparison():
    """Demonstrate governance system comparison."""
    print_section("Example 3: Governance System Comparison")

    engine = AnalyticalEngine()

    scenario = "Rapid military mobilization in response to regional crisis"

    comparison = engine.compare_governance_systems(
        scenario=scenario,
        authoritarian_context={},
        democratic_context={}
    )

    print(comparison)


def example_4_corruption_assessment():
    """Demonstrate corruption type assessment."""
    print_section("Example 4: Corruption Type Assessment")

    lenses = AnalyticalLenses()

    print("Corruption Type Analysis:\n")

    # Assess different corruption types
    corruption_types = [
        (CorruptionType.PARASITIC, "Russia"),
        (CorruptionType.MANAGED_BOUNDED, "China"),
        (CorruptionType.INSTITUTIONALIZED_PATRONAGE, "Iran IRGC"),
        (CorruptionType.LOW_CORRUPTION, "NATO countries")
    ]

    for corr_type, example in corruption_types:
        analysis = lenses.corruption.analyze(
            corr_type,
            operational_context="Sustained high-intensity conventional operations"
        )

        print(f"\n{corr_type.value.upper()} ({example}):")
        print(f"  Operational Impact: {analysis['operational_impact']}")
        print(f"  Characteristics:")
        for char in analysis['characteristics']:
            print(f"    - {char}")


def example_5_non_western_military_assessment():
    """Demonstrate non-Western military analysis."""
    print_section("Example 5: Non-Western Military Assessment")

    lenses = AnalyticalLenses()

    militaries = ["Chinese PLA", "Russian Military", "Iranian Systems"]

    for military in militaries:
        analysis = lenses.non_western.analyze(military)

        print(f"\n{military.upper()}:")
        print(f"\nOperational Culture: {analysis['operational_culture']}")

        print("\nStrengths:")
        for strength in analysis['strengths']:
            print(f"  - {strength}")

        print("\nWeaknesses:")
        for weakness in analysis['weaknesses']:
            print(f"  - {weakness}")

        print(f"\nKey Insight: {analysis['key_insight']}")


def example_6_all_lenses_integration():
    """Demonstrate integration of all four lenses."""
    print_section("Example 6: All Lenses Integration")

    engine = AnalyticalEngine()

    print("Analytical Priorities:\n")
    priorities = engine.get_analytical_priorities()
    for i, priority in enumerate(priorities, 1):
        print(f"{i}. {priority}")

    print("\n" + "-" * 80)

    # Quick analysis example
    print("\nQuick Analysis Example:")
    print("Query: 'Iran's ability to sustain proxy operations in Syria'\n")

    analysis = engine.quick_analysis(
        query="Iran's ability to sustain proxy operations in Syria",
        governance_type=GovernanceType.AUTHORITARIAN_CENTRALIZED,
        corruption_type=CorruptionType.INSTITUTIONALIZED_PATRONAGE,
        military_system="Iranian Systems",
        summary="""Iran demonstrates structural advantages in proxy coordination despite
conventional military limitations. IRGC Quds Force maintains effective command and control
through patronage networks that double as operational infrastructure.""",
        logistics_assessment="""Supply lines to Syria rely on air bridge through Iraq and
maritime routes. Vulnerable to interdiction but demonstrated resilience through redundancy.
Sanctions impact advanced systems but not basic sustainment.""",
        scenarios=[
            {
                'name': 'Sustained proxy presence',
                'probability': 0.60,
                'description': 'Current operational tempo maintained'
            },
            {
                'name': 'Degraded operations',
                'probability': 0.30,
                'description': 'Israeli interdiction reduces capability'
            },
            {
                'name': 'Expansion',
                'probability': 0.10,
                'description': 'Regional instability creates opportunities'
            }
        ]
    )

    print(analysis)


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("  GeoBot 2.0: Cold Systems Analysis with Geopolitical Nuance")
    print("=" * 80)

    examples = [
        ("Framework Overview", example_1_framework_overview),
        ("China Rocket Force Analysis", example_2_china_rocket_force_analysis),
        ("Governance Comparison", example_3_governance_comparison),
        ("Corruption Assessment", example_4_corruption_assessment),
        ("Non-Western Military Assessment", example_5_non_western_military_assessment),
        ("All Lenses Integration", example_6_all_lenses_integration),
    ]

    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...\n")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("  All examples completed")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
