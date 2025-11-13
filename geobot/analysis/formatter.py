"""
GeoBot 2.0 Analysis Formatter

Formats analytical outputs according to GeoBot 2.0 specifications:
1. Summary conclusion
2. Governance structure analysis
3. Logistical interpretation
4. Corruption impact
5. Non-Western perspective integration
6. Scenarios (with probabilities)
7. Uncertainty factors
8. Signals to watch
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Scenario:
    """A scenario with probability estimate."""
    name: str
    probability: float
    description: str


@dataclass
class AnalysisOutput:
    """Structured analysis output."""
    summary: str
    governance_analysis: Dict[str, Any]
    logistics_analysis: Dict[str, Any]
    corruption_assessment: Dict[str, Any]
    non_western_perspective: Dict[str, Any]
    scenarios: List[Scenario] = field(default_factory=list)
    uncertainty_factors: List[str] = field(default_factory=list)
    signals_to_watch: List[str] = field(default_factory=list)
    comparative_notes: Optional[str] = None


class AnalysisFormatter:
    """
    Formatter for GeoBot 2.0 analytical outputs.

    Formats analysis according to the default output formula:
    1. Summary conclusion
    2. Governance structure analysis
    3. Logistical interpretation
    4. Corruption impact
    5. Non-Western perspective integration
    6. Scenarios (with probabilities)
    7. Uncertainty factors
    8. Signals to watch
    """

    def __init__(self):
        """Initialize the formatter."""
        self.section_separator = "\n" + "=" * 60 + "\n"
        self.subsection_separator = "\n" + "-" * 60 + "\n"

    def format_analysis(self, analysis: AnalysisOutput) -> str:
        """
        Format complete analysis output.

        Parameters
        ----------
        analysis : AnalysisOutput
            Analysis to format

        Returns
        -------
        str
            Formatted analysis
        """
        sections = []

        # 1. Summary
        sections.append(self._format_summary(analysis.summary))

        # 2. Governance Structure Analysis
        sections.append(self._format_governance(analysis.governance_analysis))

        # 3. Logistics
        sections.append(self._format_logistics(analysis.logistics_analysis))

        # 4. Corruption Dynamics
        sections.append(self._format_corruption(analysis.corruption_assessment))

        # 5. Non-Western Context
        sections.append(self._format_non_western(analysis.non_western_perspective))

        # 6. Scenarios
        sections.append(self._format_scenarios(analysis.scenarios))

        # 7. Uncertainty
        sections.append(self._format_uncertainty(analysis.uncertainty_factors))

        # 8. Signals to Watch
        sections.append(self._format_signals(analysis.signals_to_watch))

        # 9. Comparative Note (if present)
        if analysis.comparative_notes:
            sections.append(self._format_comparative(analysis.comparative_notes))

        return self.section_separator.join(sections)

    def _format_summary(self, summary: str) -> str:
        """Format assessment summary."""
        return f"""ASSESSMENT:
{summary}"""

    def _format_governance(self, governance: Dict[str, Any]) -> str:
        """Format governance structure analysis."""
        output = ["GOVERNANCE STRUCTURE CONTEXT:"]

        if 'system_type' in governance:
            output.append(f"\nSystem Type: {governance['system_type']}")

        if 'advantages' in governance:
            output.append("\nAdvantages:")
            for adv in governance['advantages']:
                output.append(f"  - {adv}")

        if 'disadvantages' in governance:
            output.append("\nDisadvantages:")
            for dis in governance['disadvantages']:
                output.append(f"  - {dis}")

        if 'trade_off' in governance:
            output.append(f"\nTrade-off: {governance['trade_off']}")

        if 'context_specific_advantage' in governance:
            output.append(f"\nContextual Advantage: {governance['context_specific_advantage']}")

        return "\n".join(output)

    def _format_logistics(self, logistics: Dict[str, Any]) -> str:
        """Format logistics analysis."""
        output = ["LOGISTICS:"]

        if 'assessment' in logistics:
            output.append(f"\n{logistics['assessment']}")

        if 'supply_chain_status' in logistics:
            output.append(f"\nSupply Chain Status: {logistics['supply_chain_status']}")

        if 'maintenance_capacity' in logistics:
            output.append(f"Maintenance Capacity: {logistics['maintenance_capacity']}")

        if 'key_constraints' in logistics:
            output.append("\nKey Constraints:")
            for constraint in logistics['key_constraints']:
                output.append(f"  - {constraint}")

        if 'mitigating_factors' in logistics:
            output.append("\nMitigating Factors:")
            for factor in logistics['mitigating_factors']:
                output.append(f"  - {factor}")

        return "\n".join(output)

    def _format_corruption(self, corruption: Dict[str, Any]) -> str:
        """Format corruption assessment."""
        output = ["CORRUPTION DYNAMICS:"]

        if 'corruption_type' in corruption:
            output.append(f"\nCorruption Type: {corruption['corruption_type']}")

        if 'evidence' in corruption:
            output.append(f"\nEvidence: {corruption['evidence']}")

        if 'operational_impact' in corruption:
            output.append(f"\nOperational Impact: {corruption['operational_impact']}")

        if 'comparison' in corruption:
            output.append(f"\nComparative Context: {corruption['comparison']}")

        if 'risk_assessment' in corruption:
            output.append(f"\nRisk: {corruption['risk_assessment']}")

        return "\n".join(output)

    def _format_non_western(self, non_western: Dict[str, Any]) -> str:
        """Format non-Western perspective."""
        output = ["NON-WESTERN CONTEXT:"]

        if 'analysis_framework' in non_western:
            output.append(f"\n{non_western['analysis_framework']}")

        if 'indigenous_strengths' in non_western:
            output.append("\nIndigenous Strengths:")
            for strength in non_western['indigenous_strengths']:
                output.append(f"  - {strength}")

        if 'structural_constraints' in non_western:
            output.append("\nStructural Constraints:")
            for constraint in non_western['structural_constraints']:
                output.append(f"  - {constraint}")

        if 'institutional_context' in non_western:
            output.append(f"\nInstitutional Context: {non_western['institutional_context']}")

        if 'key_distinction' in non_western:
            output.append(f"\nKey Distinction: {non_western['key_distinction']}")

        return "\n".join(output)

    def _format_scenarios(self, scenarios: List[Scenario]) -> str:
        """Format scenario probabilities."""
        output = ["SCENARIOS:"]

        if not scenarios:
            output.append("\n(Scenarios require additional context)")
            return "\n".join(output)

        # Sort by probability descending
        sorted_scenarios = sorted(scenarios, key=lambda s: s.probability, reverse=True)

        for scenario in sorted_scenarios:
            output.append(f"\n• {scenario.name} ({scenario.probability:.2f})")
            output.append(f"  {scenario.description}")

        return "\n".join(output)

    def _format_uncertainty(self, uncertainty_factors: List[str]) -> str:
        """Format uncertainty factors."""
        output = ["UNCERTAINTY:"]

        if not uncertainty_factors:
            output.append("\n(Standard intelligence limitations apply)")
            return "\n".join(output)

        for factor in uncertainty_factors:
            output.append(f"  - {factor}")

        return "\n".join(output)

    def _format_signals(self, signals: List[str]) -> str:
        """Format signals to watch."""
        output = ["SIGNALS TO WATCH:"]

        if not signals:
            output.append("\n(Ongoing monitoring required)")
            return "\n".join(output)

        for signal in signals:
            output.append(f"  - {signal}")

        return "\n".join(output)

    def _format_comparative(self, comparative_note: str) -> str:
        """Format comparative note."""
        return f"""COMPARATIVE NOTE:
{comparative_note}"""

    def create_structured_output(
        self,
        summary: str,
        governance: Dict[str, Any],
        logistics: Dict[str, Any],
        corruption: Dict[str, Any],
        non_western: Dict[str, Any],
        scenarios: Optional[List[Dict[str, Any]]] = None,
        uncertainty: Optional[List[str]] = None,
        signals: Optional[List[str]] = None,
        comparative: Optional[str] = None
    ) -> AnalysisOutput:
        """
        Create structured analysis output.

        Parameters
        ----------
        summary : str
            Assessment summary
        governance : Dict[str, Any]
            Governance structure analysis
        logistics : Dict[str, Any]
            Logistics analysis
        corruption : Dict[str, Any]
            Corruption assessment
        non_western : Dict[str, Any]
            Non-Western perspective
        scenarios : Optional[List[Dict[str, Any]]]
            List of scenarios with probabilities
        uncertainty : Optional[List[str]]
            Uncertainty factors
        signals : Optional[List[str]]
            Signals to watch
        comparative : Optional[str]
            Comparative note

        Returns
        -------
        AnalysisOutput
            Structured output
        """
        scenario_objects = []
        if scenarios:
            scenario_objects = [
                Scenario(
                    name=s['name'],
                    probability=s['probability'],
                    description=s['description']
                )
                for s in scenarios
            ]

        return AnalysisOutput(
            summary=summary,
            governance_analysis=governance,
            logistics_analysis=logistics,
            corruption_assessment=corruption,
            non_western_perspective=non_western,
            scenarios=scenario_objects,
            uncertainty_factors=uncertainty or [],
            signals_to_watch=signals or [],
            comparative_notes=comparative
        )
