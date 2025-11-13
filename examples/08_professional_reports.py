"""
GeoBotv1 Example 08: Professional Intelligence Reports

Demonstrates how to generate professional intelligence reports in the style of:
- CIA National Intelligence Estimates (NIE)
- DIA Intelligence Assessments
- RAND Corporation Analysis

Features:
- ICD 203 confidence levels and likelihood estimators
- Classification markings (UNCLASSIFIED to TOP SECRET)
- BLUF (Bottom Line Up Front) executive summaries
- Key judgments with confidence assessments
- Scenarios with probability and indicators
- Indicators and Warnings (I&W)
- Information gaps and collection requirements
- Multiple output formats (text, markdown, JSON)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from geobot.interface import (
    IntelligenceReport,
    ReportBuilder,
    Classification,
    ConfidenceLevel,
    Likelihood,
    ReportType,
    KeyJudgment,
    Scenario,
    Indicator,
    AnalystAgent,
    AnalysisResult
)


def example_1_create_basic_report():
    """Example 1: Create a basic intelligence report from scratch."""
    print("="*80)
    print("EXAMPLE 1: Creating a Basic Intelligence Report")
    print("="*80)

    # Create report components
    key_judgments = [
        KeyJudgment(
            judgment="Iran is likely to continue nuclear enrichment activities despite sanctions",
            confidence=ConfidenceLevel.MODERATE,
            basis="Based on satellite imagery showing continued centrifuge operation, intercepted communications, and historical pattern analysis"
        ),
        KeyJudgment(
            judgment="Regional tensions will escalate if enrichment reaches 90% purity",
            confidence=ConfidenceLevel.HIGH,
            basis="Game-theoretic analysis shows deterrence breakdown at weapons-grade threshold, supported by ally statements and military posturing"
        ),
        KeyJudgment(
            judgment="Diplomatic negotiations may resume within 60-90 days",
            confidence=ConfidenceLevel.LOW,
            basis="Limited diplomatic signaling detected, but historical precedent suggests openness during economic pressure"
        )
    ]

    scenarios = [
        Scenario(
            name="Status Quo Continuation",
            description="Iran continues enrichment at current 60% level while engaging in limited diplomatic talks",
            probability=0.55,
            likelihood=Likelihood.ROUGHLY_EVEN,
            key_indicators=[
                "Centrifuge count remains stable at ~5,000 IR-6 units",
                "Enriched uranium stockpile grows 5-10 kg/month",
                "Back-channel communications continue",
                "No new sanctions imposed"
            ],
            timeline="Next 3-6 months"
        ),
        Scenario(
            name="Escalation to Weapons-Grade",
            description="Iran enriches to 90% purity, triggering regional crisis",
            probability=0.25,
            likelihood=Likelihood.UNLIKELY,
            key_indicators=[
                "Advanced centrifuge installation accelerates",
                "IAEA inspectors denied access",
                "Hardline rhetoric intensifies",
                "Regional allies move to high alert"
            ],
            timeline="60-120 days if triggered"
        ),
        Scenario(
            name="Diplomatic Breakthrough",
            description="Renewed JCPOA negotiations lead to temporary freeze",
            probability=0.20,
            likelihood=Likelihood.UNLIKELY,
            key_indicators=[
                "High-level diplomatic contacts increase",
                "Iranian officials make conciliatory statements",
                "Sanctions relief signals from US/EU",
                "Enrichment rate slows"
            ],
            timeline="90-180 days"
        )
    ]

    indicators = [
        Indicator(
            indicator="IR-6 centrifuge installation rate exceeds 200/month",
            significance="HIGH",
            current_status="Baseline: 150/month average",
            threshold="200/month sustained over 60 days",
            collection_method="Satellite imagery + SIGINT"
        ),
        Indicator(
            indicator="Enriched uranium stockpile reaches 120kg at 60% purity",
            significance="CRITICAL",
            current_status="Current: 85kg (as of last IAEA report)",
            threshold="120kg (sufficient for quick breakout to weapons-grade)",
            collection_method="IAEA inspections"
        ),
        Indicator(
            indicator="Back-channel diplomatic contacts",
            significance="MEDIUM",
            current_status="Sporadic communications via Oman, Switzerland",
            threshold="Weekly meetings sustained over 30 days",
            collection_method="HUMINT + liaison reporting"
        )
    ]

    # Create the report
    report = IntelligenceReport(
        classification=Classification.SECRET,
        report_type=ReportType.ASSESSMENT,
        title="Iran Nuclear Program: 90-Day Assessment",
        subject="Iranian Nuclear Enrichment Trajectory and Regional Implications",
        originator="GeoBotv1 Analysis System",
        date_produced=datetime.utcnow(),
        intelligence_cut_off=datetime.utcnow() - timedelta(hours=24),
        executive_summary=(
            "Iran continues uranium enrichment at 60% purity with approximately "
            "85kg stockpile, below but approaching breakout threshold. Mathematical "
            "modeling indicates 55% probability of status quo continuation over next "
            "90 days, with escalation and diplomatic scenarios each at 20-25% probability. "
            "Key indicators are centrifuge installation rate and enriched uranium stockpile "
            "growth. Recommend enhanced collection on enrichment facilities and diplomatic channels."
        ),
        key_judgments=key_judgments,
        scenarios=scenarios,
        indicators=indicators,
        information_gaps=[
            {
                'gap': 'Precise enrichment rate at Fordow facility',
                'impact': 'HIGH',
                'collection_requirement': 'Enhanced satellite coverage + SIGINT targeting facility power consumption'
            },
            {
                'gap': 'Supreme Leader intent on weapons development vs leverage',
                'impact': 'CRITICAL',
                'collection_requirement': 'HUMINT access to SNSC deliberations'
            },
            {
                'gap': 'Regional ally (Saudi, Israel) red lines and response plans',
                'impact': 'HIGH',
                'collection_requirement': 'Liaison engagement + SIGINT on military planning'
            }
        ],
        methodology=(
            "This assessment integrates multiple analytical methods: (1) Hawkes process "
            "modeling for conflict escalation dynamics, (2) Vector Autoregression for "
            "multi-country interdependencies, (3) Bayesian inference with satellite imagery "
            "and SIGINT priors, (4) Game-theoretic scenario analysis. Confidence levels "
            "follow ICD 203 standards based on source reliability and analytical rigor."
        ),
        sources=[
            "SIGINT: Intercepted communications from AEOI facilities (HIGH confidence)",
            "IMINT: Commercial satellite imagery from Natanz, Fordow (MODERATE confidence)",
            "OSINT: IAEA quarterly reports, Iranian state media (MODERATE confidence)",
            "HUMINT: Regional liaison reporting (LOW-MODERATE confidence due to access limitations)"
        ],
        confidence_overall=ConfidenceLevel.MODERATE
    )

    # Display in different formats
    print("\n" + "="*80)
    print("TEXT FORMAT (Intelligence Community Standard)")
    print("="*80)
    print(report.format_text(width=80))

    print("\n\n" + "="*80)
    print("MARKDOWN FORMAT (For Digital Distribution)")
    print("="*80)
    print(report.format_markdown())

    print("\n\n" + "="*80)
    print("JSON FORMAT (For Machine Processing)")
    print("="*80)
    import json
    print(json.dumps(report.format_json(), indent=2))

    return report


def example_2_build_from_analysis_result():
    """Example 2: Build intelligence report from GeoBotv1 analysis result."""
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Building Report from Analysis Result")
    print("="*80)

    # Simulate an AnalysisResult (in production, this comes from AnalystAgent)
    analysis_result = AnalysisResult(
        query="What is the risk of Russia-Ukraine conflict escalation in next 30 days?",
        query_intent={
            'entities': ['Russia', 'Ukraine'],
            'timeframe': '30 days',
            'analysis_type': 'risk_forecast'
        },
        narrative_answer=(
            "Mathematical modeling indicates moderate-high risk (0.68 probability) of "
            "further escalation in Russia-Ukraine conflict over next 30 days. Hawkes "
            "process analysis shows branching ratio of 0.82, indicating self-exciting "
            "dynamics where each incident triggers additional incidents. Vector "
            "autoregression on regional tensions shows positive feedback loops. Key "
            "escalation drivers include ongoing offensive operations in Donbas and "
            "long-range strike capabilities. De-escalation scenario remains possible "
            "(0.32 probability) if diplomatic channels reopen."
        ),
        structured_analysis={
            'risk_score': 0.68,
            'confidence': 0.72,
            'forecast_horizon_days': 30,
            'escalation_probability': 0.68,
            'de_escalation_probability': 0.32,
            'key_drivers': [
                {'factor': 'Ongoing offensive operations', 'impact': 0.35},
                {'factor': 'Long-range strike capabilities', 'impact': 0.28},
                {'factor': 'Diplomatic channel closure', 'impact': 0.22},
                {'factor': 'Economic pressure on Russia', 'impact': 0.15}
            ],
            'models_used': ['hawkes', 'var', 'bayesian_forecast'],
            'hawkes_branching_ratio': 0.82,
            'var_granger_causality': {
                'russia_aggression -> ukraine_response': 0.73,
                'ukraine_response -> russia_aggression': 0.41
            }
        },
        visualizations=[],
        modules_used=['hawkes', 'var', 'bayesian'],
        timestamp=datetime.utcnow(),
        processing_time_seconds=4.7,
        confidence=0.72
    )

    # Build intelligence report using ReportBuilder
    builder = ReportBuilder()
    report = builder.from_analysis_result(
        analysis_result,
        classification=Classification.SECRET,
        report_type=ReportType.FORECAST,
        originator="GeoBotv1 Forecast Engine"
    )

    print("\nGenerated Intelligence Report:")
    print(report.format_text(width=80))

    return report


def example_3_warning_report():
    """Example 3: Create an Intelligence Warning report (highest urgency)."""
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Intelligence Warning Report")
    print("="*80)

    report = IntelligenceReport(
        classification=Classification.TOP_SECRET,
        report_type=ReportType.WARNING,
        title="CRITICAL: Taiwan Strait Crisis Indicators",
        subject="Imminent Risk of PRC Military Action Against Taiwan",
        originator="GeoBotv1 Watch Daemon",
        date_produced=datetime.utcnow(),
        intelligence_cut_off=datetime.utcnow() - timedelta(hours=1),
        executive_summary=(
            "CRITICAL WARNING: Multiple indicators suggest very high probability (85-90%) "
            "of PRC military action against Taiwan within 7-14 days. Anomaly detection "
            "systems flagged unprecedented PLAN naval movements, PLARF missile unit "
            "dispersal, and PLAAF exercise surge. Hawkes process branching ratio reached "
            "0.94 (supercritical threshold). Recommend immediate alert to national command "
            "authority and regional force posture elevation."
        ),
        key_judgments=[
            KeyJudgment(
                judgment="PRC is likely conducting final preparations for major military operation against Taiwan",
                confidence=ConfidenceLevel.HIGH,
                basis="Convergence of naval deployment (15+ amphibious ships), missile unit dispersal (85% readiness), air activity surge (3x normal), and logistics pre-positioning"
            ),
            KeyJudgment(
                judgment="Window of action is 7-14 days based on current force posture and weather windows",
                confidence=ConfidenceLevel.MODERATE,
                basis="Amphibious operations require specific tidal and weather conditions; current positioning consistent with mid-month timeline"
            ),
            KeyJudgment(
                judgment="Diplomatic off-ramps are closing rapidly",
                confidence=ConfidenceLevel.HIGH,
                basis="PRC leadership rhetoric shift, diplomatic channel silence for 72+ hours, and historical precedent analysis"
            )
        ],
        scenarios=[
            Scenario(
                name="Limited Blockade/Quarantine",
                description="PRC implements 'customs enforcement zone' around Taiwan, tests US/allied response",
                probability=0.55,
                likelihood=Likelihood.ROUGHLY_EVEN,
                key_indicators=[
                    "Coast Guard and maritime militia surge",
                    "Air defense identification zone enforcement",
                    "Limited duration (7-14 days initial)",
                    "Avoid direct engagement with US forces"
                ],
                timeline="Within 7 days"
            ),
            Scenario(
                name="Island Seizure (Kinmen/Matsu)",
                description="PRC seizes offshore islands as show of force and negotiating leverage",
                probability=0.30,
                likelihood=Likelihood.UNLIKELY,
                key_indicators=[
                    "Amphibious rehearsal near target islands",
                    "Precision strike against island defenses",
                    "Rapid airborne/amphibious assault",
                    "Consolidation within 48 hours"
                ],
                timeline="Within 10 days"
            ),
            Scenario(
                name="Full-Scale Invasion",
                description="Large-scale amphibious and airborne assault on Taiwan main island",
                probability=0.15,
                likelihood=Likelihood.REMOTE,
                key_indicators=[
                    "Massive PLARF strike to suppress air defenses",
                    "Multi-axis amphibious landings",
                    "Strategic airborne drops",
                    "Cyber/space attack to isolate Taiwan"
                ],
                timeline="Within 14 days if chosen"
            )
        ],
        indicators=[
            Indicator(
                indicator="PLAN amphibious fleet concentration",
                significance="CRITICAL",
                current_status="TRIGGERED: 15+ Type 071/075 vessels within 200nm of Taiwan",
                threshold="12+ vessels (EXCEEDED)",
                collection_method="IMINT + SIGINT"
            ),
            Indicator(
                indicator="PLARF DF-15/16/17 dispersal from garrisons",
                significance="CRITICAL",
                current_status="TRIGGERED: 85% of units deployed to field positions",
                threshold="70% deployment (EXCEEDED)",
                collection_method="Satellite imagery + SIGINT"
            ),
            Indicator(
                indicator="PLAAF surge operations",
                significance="HIGH",
                current_status="TRIGGERED: 300+ sorties/day (3x normal)",
                threshold="200+ sorties/day (EXCEEDED)",
                collection_method="Taiwan ADIZ monitoring + SIGINT"
            ),
            Indicator(
                indicator="Hawkes process branching ratio",
                significance="CRITICAL",
                current_status="TRIGGERED: 0.94 (supercritical regime)",
                threshold="0.90 (EXCEEDED)",
                collection_method="GeoBotv1 mathematical modeling"
            )
        ],
        information_gaps=[
            {
                'gap': 'PRC leadership decision timeline and final authorization',
                'impact': 'CRITICAL',
                'collection_requirement': 'HUMINT access to CMC deliberations - HIGHEST PRIORITY'
            },
            {
                'gap': 'Specific D-Day and H-Hour if operation is authorized',
                'impact': 'CRITICAL',
                'collection_requirement': 'SIGINT targeting operational orders - FLASH precedence'
            }
        ],
        methodology=(
            "URGENT WARNING based on real-time anomaly detection (Kalman filtering on "
            "military activity time series), Hawkes process supercritical detection, and "
            "multi-source intelligence fusion. Automated alert triggered by Watch Daemon "
            "at 0342Z when branching ratio exceeded 0.90 threshold."
        ),
        sources=[
            "IMINT: NRO satellite coverage of naval bases, missile garrisons (HIGH confidence)",
            "SIGINT: PLA communications intercepts (MODERATE-HIGH confidence)",
            "Taiwan MND: ADIZ incursion data, coastal radar (HIGH confidence)",
            "GeoBotv1: Mathematical anomaly detection (MODERATE confidence - model-based)"
        ],
        confidence_overall=ConfidenceLevel.HIGH,
        dissemination_controls=["NOFORN", "ORCON"]
    )

    print("\nCRITICAL INTELLIGENCE WARNING:")
    print(report.format_text(width=80))

    return report


def example_4_different_classification_levels():
    """Example 4: Demonstrate different classification levels."""
    print("\n\n" + "="*80)
    print("EXAMPLE 4: Reports at Different Classification Levels")
    print("="*80)

    classifications = [
        Classification.UNCLASSIFIED,
        Classification.CONFIDENTIAL,
        Classification.SECRET,
        Classification.TOP_SECRET
    ]

    for classification in classifications:
        report = IntelligenceReport(
            classification=classification,
            report_type=ReportType.ASSESSMENT,
            title=f"Sample Report at {classification.value} Level",
            subject="Demonstration of Classification Handling",
            originator="GeoBotv1",
            date_produced=datetime.utcnow(),
            intelligence_cut_off=datetime.utcnow(),
            executive_summary=(
                f"This is a sample report demonstrating {classification.value} "
                "classification handling with appropriate markings and dissemination controls."
            ),
            key_judgments=[
                KeyJudgment(
                    judgment="Sample judgment for classification demonstration",
                    confidence=ConfidenceLevel.MODERATE,
                    basis="Demonstration purposes"
                )
            ],
            scenarios=[],
            indicators=[],
            information_gaps=[],
            methodology="Demonstration",
            sources=["Sample sources"],
            confidence_overall=ConfidenceLevel.MODERATE
        )

        print(f"\n{classification.value} Report:")
        print("-" * 80)
        # Show just the header and classification markings
        lines = report.format_text(width=80).split('\n')
        print('\n'.join(lines[:10]))  # First 10 lines showing classification
        print("...[content truncated]...\n")


def example_5_export_for_distribution():
    """Example 5: Export report in different formats for distribution."""
    print("\n\n" + "="*80)
    print("EXAMPLE 5: Exporting Reports for Distribution")
    print("="*80)

    # Create a sample report
    report = IntelligenceReport(
        classification=Classification.SECRET,
        report_type=ReportType.ASSESSMENT,
        title="North Korea Missile Program Assessment",
        subject="DPRK ICBM Development Status",
        originator="GeoBotv1",
        date_produced=datetime.utcnow(),
        intelligence_cut_off=datetime.utcnow(),
        executive_summary="Sample assessment of DPRK missile capabilities",
        key_judgments=[
            KeyJudgment(
                judgment="DPRK has likely achieved ICBM re-entry vehicle capability",
                confidence=ConfidenceLevel.MODERATE,
                basis="Test data analysis and materials science assessment"
            )
        ],
        scenarios=[],
        indicators=[],
        information_gaps=[],
        methodology="Multi-source intelligence fusion",
        sources=["IMINT", "OSINT", "Technical analysis"],
        confidence_overall=ConfidenceLevel.MODERATE
    )

    # Export to different formats
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Text format (for print/PDF)
    text_file = output_dir / "report_text.txt"
    with open(text_file, 'w') as f:
        f.write(report.format_text(width=80))
    print(f"✓ Text report saved: {text_file}")

    # Markdown format (for web/wiki)
    md_file = output_dir / "report_markdown.md"
    with open(md_file, 'w') as f:
        f.write(report.format_markdown())
    print(f"✓ Markdown report saved: {md_file}")

    # JSON format (for APIs/databases)
    json_file = output_dir / "report_data.json"
    import json
    with open(json_file, 'w') as f:
        json.dump(report.format_json(), f, indent=2, default=str)
    print(f"✓ JSON report saved: {json_file}")

    print(f"\nAll reports exported to: {output_dir}")


def main():
    """Run all examples."""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "GeoBotv1 - Professional Intelligence Report Generation".center(78) + "█")
    print("█" + "CIA/DIA/RAND-Style Analysis Output".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # Run examples
    example_1_create_basic_report()
    example_2_build_from_analysis_result()
    example_3_warning_report()
    example_4_different_classification_levels()
    example_5_export_for_distribution()

    print("\n\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. IntelligenceReport class supports CIA/DIA/RAND-style formatting")
    print("2. ICD 203 confidence levels and likelihood estimators")
    print("3. Classification markings from UNCLASSIFIED to TOP SECRET")
    print("4. ReportBuilder converts AnalysisResult to professional reports")
    print("5. Multiple output formats: text, markdown, JSON")
    print("6. Comprehensive report structure: BLUF, key judgments, scenarios, I&W")
    print("\nAll GeoBotv1 analysis can now be presented in professional IC format!")


if __name__ == "__main__":
    main()
