"""
GeoBot 2.0 Analytical Engine

Main interface for conducting GeoBot 2.0 analysis. Integrates all lenses,
applies analytical priorities, and generates formatted output.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from .framework import GeoBotFramework
from .lenses import (
    AnalyticalLenses,
    GovernanceType,
    CorruptionType,
    MilitaryProfile
)
from .formatter import AnalysisFormatter, AnalysisOutput, Scenario


@dataclass
class AnalyticalPriorities:
    """
    Analytical priorities for GeoBot 2.0.

    Every analysis checks:
    1. Governance structure impact
    2. Logistics coherence
    3. Corruption type and impact
    4. Institutional context
    5. Communication networks
    6. Organizational cohesion
    """

    priorities: List[str] = field(default_factory=lambda: [
        "Governance structure impact - Does this scenario favor centralized decision-making or distributed adaptation?",
        "Logistics coherence - Supply chains, maintenance, communications",
        "Corruption type and impact - What kind of corruption exists? Does it critically impair this specific operation?",
        "Institutional context - Are we analyzing this military using appropriate cultural/organizational frameworks?",
        "Communication networks - Information flow and coordination",
        "Organizational cohesion - Unit cohesion and morale"
    ])

    def check_all(self, context: Dict[str, Any]) -> Dict[str, bool]:
        """
        Check which priorities have been addressed in analysis.

        Parameters
        ----------
        context : Dict[str, Any]
            Analysis context

        Returns
        -------
        Dict[str, bool]
            Priority checklist
        """
        return {
            'governance_structure': 'governance_type' in context,
            'logistics': 'logistics_assessment' in context,
            'corruption': 'corruption_type' in context,
            'institutional_context': 'military_system' in context,
            'communications': 'communications_status' in context,
            'cohesion': 'cohesion_assessment' in context
        }


class AnalyticalEngine:
    """
    Main analytical engine for GeoBot 2.0.

    Integrates framework, lenses, priorities, and formatter to provide
    comprehensive geopolitical analysis.
    """

    def __init__(self):
        """Initialize the analytical engine."""
        self.framework = GeoBotFramework()
        self.lenses = AnalyticalLenses()
        self.priorities = AnalyticalPriorities()
        self.formatter = AnalysisFormatter()

    def analyze(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Conduct complete GeoBot 2.0 analysis.

        Parameters
        ----------
        query : str
            Analytical query or question
        context : Dict[str, Any]
            Context for analysis including:
            - governance_type: GovernanceType
            - corruption_type: CorruptionType
            - military_system: str
            - logistics_data: Dict
            - scenario_description: str

        Returns
        -------
        str
            Formatted analysis
        """
        # Apply all lenses
        analysis_components = self._apply_lenses(context)

        # Create structured output
        output = self._create_output(query, context, analysis_components)

        # Validate against framework
        if not self.framework.validate_analysis(output.__dict__):
            raise ValueError("Analysis does not adhere to GeoBot 2.0 framework")

        # Format and return
        return self.formatter.format_analysis(output)

    def _apply_lenses(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all analytical lenses to context.

        Parameters
        ----------
        context : Dict[str, Any]
            Analysis context

        Returns
        -------
        Dict[str, Any]
            Lens analyses
        """
        components = {}

        # Lens A: Logistics
        components['logistics'] = self.lenses.logistics.analyze(context)

        # Lens B: Governance
        if 'governance_type' in context:
            gov_type = context['governance_type']
            scenario = context.get('scenario_description', 'General scenario')
            components['governance'] = self.lenses.governance.analyze(
                gov_type, scenario
            )
        else:
            components['governance'] = {
                'note': 'Governance type not specified'
            }

        # Lens C: Corruption
        if 'corruption_type' in context:
            corr_type = context['corruption_type']
            operational_context = context.get('operational_context', 'General operations')
            components['corruption'] = self.lenses.corruption.analyze(
                corr_type, operational_context
            )
        else:
            components['corruption'] = {
                'note': 'Corruption type not specified'
            }

        # Lens D: Non-Western
        if 'military_system' in context:
            military = context['military_system']
            components['non_western'] = self.lenses.non_western.analyze(military)
        else:
            components['non_western'] = {
                'note': 'Military system not specified'
            }

        return components

    def _create_output(
        self,
        query: str,
        context: Dict[str, Any],
        components: Dict[str, Any]
    ) -> AnalysisOutput:
        """
        Create structured analysis output.

        Parameters
        ----------
        query : str
            Original query
        context : Dict[str, Any]
            Analysis context
        components : Dict[str, Any]
            Lens analysis components

        Returns
        -------
        AnalysisOutput
            Structured output
        """
        # Extract or create summary
        summary = context.get('summary', f"Analysis of: {query}")

        # Governance analysis
        governance_analysis = {
            'system_type': context.get('governance_type', 'Not specified'),
            **components.get('governance', {})
        }

        # Logistics analysis
        logistics_analysis = {
            'assessment': context.get('logistics_assessment', 'Requires detailed logistics data'),
            **components.get('logistics', {})
        }

        # Corruption assessment
        corruption_assessment = {
            'corruption_type': context.get('corruption_type', 'Not specified'),
            **components.get('corruption', {})
        }

        # Non-Western perspective
        non_western_perspective = {
            'military_system': context.get('military_system', 'Not specified'),
            **components.get('non_western', {})
        }

        # Scenarios
        scenarios = context.get('scenarios', [])

        # Uncertainty
        uncertainty = context.get('uncertainty_factors', [
            "Limited visibility into internal decision-making",
            "Intelligence gaps in operational readiness",
            "Uncertainty in actor intentions"
        ])

        # Signals
        signals = context.get('signals_to_watch', [
            "Changes in leadership or command structure",
            "Shifts in training tempo or deployment patterns",
            "Procurement and supply chain activity"
        ])

        # Comparative notes
        comparative = context.get('comparative_notes')

        return self.formatter.create_structured_output(
            summary=summary,
            governance=governance_analysis,
            logistics=logistics_analysis,
            corruption=corruption_assessment,
            non_western=non_western_perspective,
            scenarios=scenarios,
            uncertainty=uncertainty,
            signals=signals,
            comparative=comparative
        )

    def quick_analysis(
        self,
        query: str,
        governance_type: GovernanceType,
        corruption_type: CorruptionType,
        military_system: str,
        summary: str,
        **kwargs
    ) -> str:
        """
        Quick analysis with minimal context.

        Parameters
        ----------
        query : str
            Analysis query
        governance_type : GovernanceType
            Type of governance system
        corruption_type : CorruptionType
            Type of corruption present
        military_system : str
            Military system being analyzed
        summary : str
            Assessment summary
        **kwargs
            Additional context

        Returns
        -------
        str
            Formatted analysis
        """
        context = {
            'governance_type': governance_type,
            'corruption_type': corruption_type,
            'military_system': military_system,
            'summary': summary,
            **kwargs
        }

        return self.analyze(query, context)

    def compare_governance_systems(
        self,
        scenario: str,
        authoritarian_context: Dict[str, Any],
        democratic_context: Dict[str, Any]
    ) -> str:
        """
        Compare authoritarian vs democratic systems for a scenario.

        Parameters
        ----------
        scenario : str
            Scenario description
        authoritarian_context : Dict[str, Any]
            Context for authoritarian system
        democratic_context : Dict[str, Any]
            Context for democratic system

        Returns
        -------
        str
            Comparative analysis
        """
        comparison = self.lenses.governance.compare_systems(scenario)

        output = [
            f"COMPARATIVE ANALYSIS: {scenario}",
            "",
            "AUTHORITARIAN/CENTRALIZED SYSTEMS:",
            "Advantages:"
        ]

        for adv in comparison['authoritarian']['advantages']:
            output.append(f"  - {adv}")

        output.append("\nDisadvantages:")
        for dis in comparison['authoritarian']['disadvantages']:
            output.append(f"  - {dis}")

        output.append("\nDEMOCRATIC/CONSENSUS SYSTEMS:")
        output.append("Advantages:")
        for adv in comparison['democratic']['advantages']:
            output.append(f"  - {adv}")

        output.append("\nDisadvantages:")
        for dis in comparison['democratic']['disadvantages']:
            output.append(f"  - {dis}")

        output.append(f"\nKEY INSIGHT: {comparison['key_insight']}")

        return "\n".join(output)

    def assess_corruption_impact(
        self,
        corruption_type: CorruptionType,
        operation_type: str
    ) -> str:
        """
        Assess corruption impact on specific operation type.

        Parameters
        ----------
        corruption_type : CorruptionType
            Type of corruption
        operation_type : str
            Type of operation

        Returns
        -------
        str
            Impact assessment
        """
        return self.lenses.corruption.assess_impact(corruption_type, operation_type)

    def get_framework_summary(self) -> Dict[str, Any]:
        """
        Get summary of GeoBot 2.0 framework.

        Returns
        -------
        Dict[str, Any]
            Framework summary
        """
        return self.framework.get_framework_summary()

    def get_analytical_priorities(self) -> List[str]:
        """
        Get list of analytical priorities.

        Returns
        -------
        List[str]
            Analytical priorities
        """
        return self.priorities.priorities
