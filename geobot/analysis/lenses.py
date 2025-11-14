"""
GeoBot 2.0 Analytical Lenses

Four complementary analytical lenses for geopolitical analysis:
- Lens A: Logistics as Power
- Lens B: Governance Structure & Decision Speed
- Lens C: Corruption as Context-Dependent Variable
- Lens D: Non-Western Military Logic
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


# ============================================================================
# Lens A: Logistics as Power
# ============================================================================

@dataclass
class LogisticsLens:
    """
    Lens A: Logistics as Power (Unchanged from GeoBot v1)

    Prioritizes supply chains, throughput, maintenance, communications infrastructure.
    Logistics remain the ultimate constraint.
    """

    name: str = "Logistics as Power"
    priority_areas: List[str] = field(default_factory=lambda: [
        "Supply chains and throughput",
        "Maintenance capacity",
        "Communications infrastructure",
        "Resource mobilization speed",
        "Sustainment capacity"
    ])

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze situation through logistics lens.

        Parameters
        ----------
        context : Dict[str, Any]
            Situational context

        Returns
        -------
        Dict[str, Any]
            Logistics analysis
        """
        return {
            'lens': self.name,
            'priority_areas': self.priority_areas,
            'assessment': "Logistics coherence analysis required",
            'context': context
        }


# ============================================================================
# Lens B: Governance Structure & Decision Speed
# ============================================================================

class GovernanceType(Enum):
    """Types of governance structures."""
    AUTHORITARIAN_CENTRALIZED = "authoritarian/centralized"
    DEMOCRATIC_CONSENSUS = "democratic/consensus"
    HYBRID = "hybrid"


@dataclass
class GovernanceAdvantages:
    """Advantages of a governance type."""
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)


@dataclass
class GovernanceLens:
    """
    Lens B: Governance Structure & Decision Speed

    Evaluates institutional agility and decision-making structures.
    Recognizes that different governance structures create different
    operational capabilities, not just deficits.
    """

    name: str = "Governance Structure & Decision Speed"

    authoritarian_profile: GovernanceAdvantages = field(default_factory=lambda: GovernanceAdvantages(
        advantages=[
            "Faster strategic pivots (no legislative/consensus delays)",
            "Rapid resource mobilization during crises",
            "Unified command structures (fewer veto points)",
            "Ability to absorb short-term costs for long-term positioning",
            "Less vulnerable to public opinion shifts"
        ],
        disadvantages=[
            "Higher corruption risk (less accountability)",
            "Information distortion (fear of reporting bad news upward)",
            "Brittleness under sustained stress (rigid hierarchies)",
            "Lower tactical initiative at junior levels",
            "Purge-induced institutional memory loss"
        ]
    ))

    democratic_profile: GovernanceAdvantages = field(default_factory=lambda: GovernanceAdvantages(
        advantages=[
            "Better information flow (less fear-based reporting)",
            "Higher tactical flexibility (NCO empowerment)",
            "More resilient under prolonged strain",
            "Transparent procurement (lower corruption)",
            "Adaptive learning cultures"
        ],
        disadvantages=[
            "Slower strategic decision-making (multiple approval layers)",
            "Political constraints on deployment/escalation",
            "Public opinion as operational constraint",
            "Bureaucratic friction in mobilization",
            "Difficulty sustaining unpopular policies"
        ]
    ))

    def analyze(self, governance_type: GovernanceType, scenario_context: str) -> Dict[str, Any]:
        """
        Analyze which governance type has structural advantage for specific scenario.

        Parameters
        ----------
        governance_type : GovernanceType
            Type of governance structure
        scenario_context : str
            Description of scenario requiring analysis

        Returns
        -------
        Dict[str, Any]
            Governance structure analysis
        """
        profile = None
        if governance_type == GovernanceType.AUTHORITARIAN_CENTRALIZED:
            profile = self.authoritarian_profile
        elif governance_type == GovernanceType.DEMOCRATIC_CONSENSUS:
            profile = self.democratic_profile

        return {
            'lens': self.name,
            'governance_type': governance_type.value,
            'advantages': profile.advantages if profile else [],
            'disadvantages': profile.disadvantages if profile else [],
            'scenario_context': scenario_context,
            'key_question': "Which governance type advantages matter for this specific scenario?"
        }

    def compare_systems(self, scenario_context: str) -> Dict[str, Any]:
        """
        Compare authoritarian vs democratic systems for a scenario.

        Parameters
        ----------
        scenario_context : str
            Description of scenario

        Returns
        -------
        Dict[str, Any]
            Comparative analysis
        """
        return {
            'lens': self.name,
            'scenario': scenario_context,
            'authoritarian': {
                'advantages': self.authoritarian_profile.advantages,
                'disadvantages': self.authoritarian_profile.disadvantages
            },
            'democratic': {
                'advantages': self.democratic_profile.advantages,
                'disadvantages': self.democratic_profile.disadvantages
            },
            'key_insight': "Governance structure creates operational trade-offs, not universal superiority"
        }


# ============================================================================
# Lens C: Corruption as Context-Dependent Variable
# ============================================================================

class CorruptionType(Enum):
    """Types of corruption and their operational impacts."""
    PARASITIC = "parasitic"  # Hollows readiness, predictably degrades performance
    MANAGED_BOUNDED = "managed/bounded"  # Limited by periodic purges
    INSTITUTIONALIZED_PATRONAGE = "institutionalized patronage"  # Loyalty networks
    LOW_CORRUPTION = "low corruption"  # Western militaries, Singapore


@dataclass
class CorruptionProfile:
    """Profile of corruption type and its impacts."""
    corruption_type: CorruptionType
    characteristics: List[str] = field(default_factory=list)
    operational_impact: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class CorruptionLens:
    """
    Lens C: Corruption as Context-Dependent Variable

    Corruption is no longer assumed to be universally crippling.
    Instead, analyzes corruption type and its context-specific impacts.
    """

    name: str = "Corruption as Context-Dependent Variable"

    corruption_profiles: Dict[CorruptionType, CorruptionProfile] = field(default_factory=lambda: {
        CorruptionType.PARASITIC: CorruptionProfile(
            corruption_type=CorruptionType.PARASITIC,
            characteristics=[
                "Hollows readiness",
                "Steals from supply chains",
                "Predictably degrades performance"
            ],
            operational_impact="Severe degradation of operational capability",
            examples=["Russia (extensive)", "Many Global South militaries"]
        ),
        CorruptionType.MANAGED_BOUNDED: CorruptionProfile(
            corruption_type=CorruptionType.MANAGED_BOUNDED,
            characteristics=[
                "Limited by periodic purges and surveillance",
                "Still present, but constrained enough to maintain basic functionality",
                "Risk: purges themselves create instability"
            ],
            operational_impact="Moderate impact, mitigated by enforcement",
            examples=["China post-Xi purges"]
        ),
        CorruptionType.INSTITUTIONALIZED_PATRONAGE: CorruptionProfile(
            corruption_type=CorruptionType.INSTITUTIONALIZED_PATRONAGE,
            characteristics=[
                "Loyalty networks provide cohesion",
                "Can coexist with effectiveness if tied to performance"
            ],
            operational_impact="Variable - depends on performance accountability",
            examples=["Iran IRGC", "Some Gulf states"]
        ),
        CorruptionType.LOW_CORRUPTION: CorruptionProfile(
            corruption_type=CorruptionType.LOW_CORRUPTION,
            characteristics=[
                "Advantage in sustained operations",
                "Can be slower to mobilize"
            ],
            operational_impact="Minimal negative impact, enables sustained operations",
            examples=["Western militaries", "Singapore"]
        )
    })

    def analyze(self, corruption_type: CorruptionType, operational_context: str) -> Dict[str, Any]:
        """
        Analyze corruption impact in specific operational context.

        Parameters
        ----------
        corruption_type : CorruptionType
            Type of corruption present
        operational_context : str
            Operational context for assessment

        Returns
        -------
        Dict[str, Any]
            Corruption impact analysis
        """
        profile = self.corruption_profiles[corruption_type]

        return {
            'lens': self.name,
            'corruption_type': corruption_type.value,
            'characteristics': profile.characteristics,
            'operational_impact': profile.operational_impact,
            'examples': profile.examples,
            'operational_context': operational_context,
            'key_question': "What type of corruption, and how does it interact with operational demands?"
        }

    def assess_impact(self, corruption_type: CorruptionType, operation_type: str) -> str:
        """
        Assess whether corruption critically impairs specific operation.

        Parameters
        ----------
        corruption_type : CorruptionType
            Type of corruption
        operation_type : str
            Type of military operation

        Returns
        -------
        str
            Impact assessment
        """
        profile = self.corruption_profiles[corruption_type]
        return f"For {operation_type}: {profile.operational_impact}"


# ============================================================================
# Lens D: Non-Western Military Logic
# ============================================================================

@dataclass
class MilitaryProfile:
    """Profile of a military system's strengths and weaknesses."""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    operational_culture: str = ""


@dataclass
class NonWesternLens:
    """
    Lens D: Non-Western Military Logic

    Incorporates indigenous operational cultures rather than measuring
    everything against NATO standards. Analyzes non-Western militaries
    using appropriate cultural/organizational frameworks.
    """

    name: str = "Non-Western Military Logic"

    military_profiles: Dict[str, MilitaryProfile] = field(default_factory=lambda: {
        "Chinese PLA": MilitaryProfile(
            strengths=[
                "Rapid infrastructure mobilization (civil-military fusion)",
                "Industrial base integration",
                "Coastal defense asymmetric advantages",
                "Improving joint operations capability",
                "Long-term strategic patience"
            ],
            weaknesses=[
                "Limited expeditionary experience",
                "Unproven complex joint operations",
                "NCO corps still developing",
                "Logistics for sustained high-intensity operations"
            ],
            operational_culture="Centralized strategic planning with improving tactical adaptation"
        ),
        "Russian Military": MilitaryProfile(
            strengths=[
                "Artillery coordination",
                "Tactical adaptation under fire (demonstrated in Ukraine)",
                "Willingness to accept casualties",
                "Deep fires integration"
            ],
            weaknesses=[
                "Logistics corruption (confirmed)",
                "Poor junior leadership initiative",
                "Industrial base constraints under sanctions"
            ],
            operational_culture="Heavy firepower doctrine with rigid tactical execution"
        ),
        "Iranian Systems": MilitaryProfile(
            strengths=[
                "Proxy warfare coordination",
                "Missile/drone saturation tactics",
                "Strategic patience",
                "Asymmetric warfare effectiveness"
            ],
            weaknesses=[
                "Air force decay",
                "Sanctions-induced technology gaps",
                "Conventional forces limitations"
            ],
            operational_culture="Asymmetric focus with strategic depth through proxies"
        )
    })

    def analyze(self, military: str) -> Dict[str, Any]:
        """
        Analyze military using appropriate cultural/organizational framework.

        Parameters
        ----------
        military : str
            Military system to analyze

        Returns
        -------
        Dict[str, Any]
            Non-Western perspective analysis
        """
        if military not in self.military_profiles:
            return {
                'lens': self.name,
                'military': military,
                'warning': "Military profile not defined - analysis requires custom framework",
                'key_question': "What are we missing if we only use Western assumptions?"
            }

        profile = self.military_profiles[military]

        return {
            'lens': self.name,
            'military': military,
            'strengths': profile.strengths,
            'weaknesses': profile.weaknesses,
            'operational_culture': profile.operational_culture,
            'key_insight': "Non-obvious strengths often missed by Western-centric analysis",
            'key_question': "Are we analyzing this military using appropriate cultural/organizational frameworks?"
        }

    def add_military_profile(self, military: str, profile: MilitaryProfile) -> None:
        """
        Add new military profile to the lens.

        Parameters
        ----------
        military : str
            Name of military system
        profile : MilitaryProfile
            Profile of the military system
        """
        self.military_profiles[military] = profile


# ============================================================================
# Combined Analytical Lenses
# ============================================================================

@dataclass
class AnalyticalLenses:
    """
    Combined analytical lenses for comprehensive GeoBot 2.0 analysis.
    """

    logistics: LogisticsLens = field(default_factory=LogisticsLens)
    governance: GovernanceLens = field(default_factory=GovernanceLens)
    corruption: CorruptionLens = field(default_factory=CorruptionLens)
    non_western: NonWesternLens = field(default_factory=NonWesternLens)

    def get_all_lenses(self) -> Dict[str, Any]:
        """
        Get all analytical lenses.

        Returns
        -------
        Dict[str, Any]
            All lenses
        """
        return {
            'Lens A': self.logistics,
            'Lens B': self.governance,
            'Lens C': self.corruption,
            'Lens D': self.non_western
        }

    def apply_all_lenses(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all lenses to a given context.

        Parameters
        ----------
        context : Dict[str, Any]
            Context to analyze

        Returns
        -------
        Dict[str, Any]
            Multi-lens analysis
        """
        return {
            'logistics_analysis': self.logistics.analyze(context),
            'governance_analysis': self.governance.compare_systems(
                context.get('scenario', 'General scenario')
            ),
            'corruption_analysis': "Requires corruption type specification",
            'non_western_analysis': "Requires military system specification",
            'integrated_assessment': "Apply all lenses for comprehensive analysis"
        }
