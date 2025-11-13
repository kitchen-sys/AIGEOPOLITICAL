"""
Professional Intelligence Report Formatting for GeoBotv1

Standardized output formats matching CIA, DIA, RAND Corporation standards:
- Classification markings
- Executive Summary (BLUF - Bottom Line Up Front)
- Key Judgments with confidence levels
- Detailed analysis
- Indicators and Warnings
- Outlook and scenarios
- Information gaps
- Sourcing and methodology

Report Types:
- Intelligence Assessment
- Forecasting Memorandum
- Warning Notice
- Policy Impact Analysis
- Situation Report (SITREP)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class Classification(Enum):
    """Classification levels."""
    UNCLASSIFIED = "UNCLASSIFIED"
    CUI = "CONTROLLED UNCLASSIFIED INFORMATION"
    CONFIDENTIAL = "CONFIDENTIAL"
    SECRET = "SECRET"
    TOP_SECRET = "TOP SECRET"
    TS_SCI = "TOP SECRET//SCI"


class ConfidenceLevel(Enum):
    """Analytic confidence levels (ICD 203 standard)."""
    HIGH = "High Confidence"
    MODERATE = "Moderate Confidence"
    LOW = "Low Confidence"


class ReportType(Enum):
    """Types of intelligence reports."""
    ASSESSMENT = "Intelligence Assessment"
    FORECAST = "Forecasting Memorandum"
    WARNING = "Warning Notice"
    POLICY_IMPACT = "Policy Impact Analysis"
    SITREP = "Situation Report"
    BRIEFING = "Intelligence Briefing"
    ESTIMATE = "National Intelligence Estimate"


class Likelihood(Enum):
    """Likelihood estimators (ICD 203 standard)."""
    REMOTE = "Remote Chance"  # < 10%
    UNLIKELY = "Unlikely"  # 10-30%
    ROUGHLY_EVEN = "Roughly Even Chance"  # 30-70%
    LIKELY = "Likely"  # 70-90%
    VERY_LIKELY = "Very Likely"  # > 90%


@dataclass
class KeyJudgment:
    """A key judgment with confidence level."""
    judgment: str
    confidence: ConfidenceLevel
    likelihood: Optional[Likelihood] = None
    basis: Optional[str] = None  # Brief explanation


@dataclass
class Scenario:
    """Scenario with probability and description."""
    name: str
    probability: float  # 0.0 - 1.0
    likelihood: Likelihood
    description: str
    indicators: List[str] = field(default_factory=list)


@dataclass
class Indicator:
    """Indicator to monitor."""
    indicator: str
    current_status: str
    threshold: str
    significance: str  # "Critical", "Important", "Monitoring"


@dataclass
class InformationGap:
    """Known information gap."""
    gap: str
    impact: str  # How it affects analysis
    collection_requirements: str


@dataclass
class IntelligenceReport:
    """Complete intelligence report in professional format."""

    # Header
    classification: Classification
    report_type: ReportType
    title: str
    date: datetime
    originator: str = "GeoBotv1 Analysis System"
    dissemination: str = "NOFORN"  # NO FOREIGN NATIONALS

    # Executive Summary (BLUF)
    executive_summary: str = ""

    # Key Judgments
    key_judgments: List[KeyJudgment] = field(default_factory=list)

    # Analysis Section
    situation_overview: str = ""
    detailed_analysis: str = ""

    # Outlook
    outlook: str = ""
    scenarios: List[Scenario] = field(default_factory=list)
    most_likely_scenario: Optional[str] = None

    # Indicators and Warnings
    indicators: List[Indicator] = field(default_factory=list)

    # Information Gaps
    information_gaps: List[InformationGap] = field(default_factory=list)

    # Sourcing and Methodology
    sources: List[str] = field(default_factory=list)
    analytic_techniques: List[str] = field(default_factory=list)
    models_used: List[str] = field(default_factory=list)

    # Confidence Assessment
    overall_confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    confidence_rationale: str = ""

    # Additional Data
    risk_score: Optional[float] = None
    time_horizon: Optional[str] = None
    geographic_focus: List[str] = field(default_factory=list)

    # Metadata
    prepared_by: str = "GeoBotv1 Automated Analysis"
    reviewed_by: Optional[str] = None

    def format_text(self, width: int = 80) -> str:
        """
        Format report as text (suitable for terminal/email).

        Args:
            width: Line width for formatting

        Returns:
            Formatted text report
        """
        lines = []
        sep = "=" * width
        line = "-" * width

        # Classification Header
        lines.append(sep)
        lines.append(f"{self.classification.value}//NOFORN".center(width))
        lines.append(sep)
        lines.append("")

        # Title and Metadata
        lines.append(f"{self.report_type.value}".center(width))
        lines.append("")
        lines.append(self.title.center(width))
        lines.append("")
        lines.append(f"Date: {self.date.strftime('%d %B %Y')}".center(width))
        lines.append(f"Originator: {self.originator}".center(width))
        lines.append("")
        lines.append(line)
        lines.append("")

        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append(line)
        lines.append("")
        lines.append(self._wrap_text(self.executive_summary, width))
        lines.append("")

        # Key Judgments
        if self.key_judgments:
            lines.append(line)
            lines.append("KEY JUDGMENTS")
            lines.append(line)
            lines.append("")
            for i, kj in enumerate(self.key_judgments, 1):
                confidence_marker = self._confidence_marker(kj.confidence)
                likelihood_str = f" ({kj.likelihood.value})" if kj.likelihood else ""
                lines.append(f"{i}. {confidence_marker} {kj.judgment}{likelihood_str}")
                if kj.basis:
                    lines.append(f"   Basis: {self._wrap_text(kj.basis, width - 10, indent=10)}")
                lines.append("")

        # Situation Overview
        if self.situation_overview:
            lines.append(line)
            lines.append("SITUATION OVERVIEW")
            lines.append(line)
            lines.append("")
            lines.append(self._wrap_text(self.situation_overview, width))
            lines.append("")

        # Detailed Analysis
        if self.detailed_analysis:
            lines.append(line)
            lines.append("DETAILED ANALYSIS")
            lines.append(line)
            lines.append("")
            lines.append(self._wrap_text(self.detailed_analysis, width))
            lines.append("")

        # Outlook and Scenarios
        if self.outlook or self.scenarios:
            lines.append(line)
            lines.append("OUTLOOK")
            lines.append(line)
            lines.append("")
            if self.outlook:
                lines.append(self._wrap_text(self.outlook, width))
                lines.append("")

            if self.scenarios:
                lines.append("Scenarios:")
                lines.append("")
                for scenario in self.scenarios:
                    lines.append(f"• {scenario.name} ({scenario.likelihood.value}, {scenario.probability:.0%})")
                    lines.append(f"  {self._wrap_text(scenario.description, width - 2, indent=2)}")
                    if scenario.indicators:
                        lines.append("  Key Indicators:")
                        for ind in scenario.indicators:
                            lines.append(f"    - {ind}")
                    lines.append("")

                if self.most_likely_scenario:
                    lines.append(f"Most Likely: {self.most_likely_scenario}")
                    lines.append("")

        # Indicators and Warnings
        if self.indicators:
            lines.append(line)
            lines.append("INDICATORS AND WARNINGS")
            lines.append(line)
            lines.append("")
            for ind in self.indicators:
                lines.append(f"• {ind.indicator} [{ind.significance}]")
                lines.append(f"  Current: {ind.current_status}")
                lines.append(f"  Threshold: {ind.threshold}")
                lines.append("")

        # Information Gaps
        if self.information_gaps:
            lines.append(line)
            lines.append("INFORMATION GAPS")
            lines.append(line)
            lines.append("")
            for gap in self.information_gaps:
                lines.append(f"• {gap.gap}")
                lines.append(f"  Impact: {gap.impact}")
                lines.append(f"  Collection Requirements: {gap.collection_requirements}")
                lines.append("")

        # Sourcing and Methodology
        lines.append(line)
        lines.append("SOURCING AND METHODOLOGY")
        lines.append(line)
        lines.append("")

        if self.models_used:
            lines.append("Mathematical Models:")
            for model in self.models_used:
                lines.append(f"  • {model}")
            lines.append("")

        if self.analytic_techniques:
            lines.append("Analytic Techniques:")
            for technique in self.analytic_techniques:
                lines.append(f"  • {technique}")
            lines.append("")

        if self.sources:
            lines.append("Sources:")
            for source in self.sources:
                lines.append(f"  • {source}")
            lines.append("")

        # Confidence Assessment
        lines.append("Overall Confidence:")
        lines.append(f"  {self.overall_confidence.value}")
        if self.confidence_rationale:
            lines.append(f"  Rationale: {self._wrap_text(self.confidence_rationale, width - 13, indent=13)}")
        lines.append("")

        # Footer
        lines.append(line)
        lines.append(f"Prepared by: {self.prepared_by}")
        if self.reviewed_by:
            lines.append(f"Reviewed by: {self.reviewed_by}")
        lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("")
        lines.append(sep)
        lines.append(f"{self.classification.value}//NOFORN".center(width))
        lines.append(sep)

        return '\n'.join(lines)

    def format_markdown(self) -> str:
        """Format report as Markdown."""
        lines = []

        # Classification banner
        lines.append(f"**{self.classification.value}//NOFORN**")
        lines.append("")

        # Title
        lines.append(f"# {self.report_type.value}")
        lines.append(f"## {self.title}")
        lines.append("")
        lines.append(f"**Date:** {self.date.strftime('%d %B %Y')}")
        lines.append(f"**Originator:** {self.originator}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(self.executive_summary)
        lines.append("")

        # Key Judgments
        if self.key_judgments:
            lines.append("## Key Judgments")
            lines.append("")
            for kj in self.key_judgments:
                confidence_badge = self._confidence_badge_md(kj.confidence)
                likelihood_str = f" *({kj.likelihood.value})*" if kj.likelihood else ""
                lines.append(f"- {confidence_badge} {kj.judgment}{likelihood_str}")
                if kj.basis:
                    lines.append(f"  - *Basis: {kj.basis}*")
            lines.append("")

        # Situation Overview
        if self.situation_overview:
            lines.append("## Situation Overview")
            lines.append("")
            lines.append(self.situation_overview)
            lines.append("")

        # Detailed Analysis
        if self.detailed_analysis:
            lines.append("## Detailed Analysis")
            lines.append("")
            lines.append(self.detailed_analysis)
            lines.append("")

        # Outlook
        if self.outlook or self.scenarios:
            lines.append("## Outlook")
            lines.append("")
            if self.outlook:
                lines.append(self.outlook)
                lines.append("")

            if self.scenarios:
                lines.append("### Scenarios")
                lines.append("")
                for scenario in self.scenarios:
                    lines.append(f"#### {scenario.name}")
                    lines.append(f"**Likelihood:** {scenario.likelihood.value} ({scenario.probability:.0%})")
                    lines.append("")
                    lines.append(scenario.description)
                    lines.append("")
                    if scenario.indicators:
                        lines.append("**Key Indicators:**")
                        for ind in scenario.indicators:
                            lines.append(f"- {ind}")
                        lines.append("")

        # Indicators
        if self.indicators:
            lines.append("## Indicators and Warnings")
            lines.append("")
            lines.append("| Indicator | Current Status | Threshold | Significance |")
            lines.append("|-----------|----------------|-----------|--------------|")
            for ind in self.indicators:
                lines.append(f"| {ind.indicator} | {ind.current_status} | {ind.threshold} | {ind.significance} |")
            lines.append("")

        # Information Gaps
        if self.information_gaps:
            lines.append("## Information Gaps")
            lines.append("")
            for gap in self.information_gaps:
                lines.append(f"### {gap.gap}")
                lines.append(f"- **Impact:** {gap.impact}")
                lines.append(f"- **Collection Requirements:** {gap.collection_requirements}")
                lines.append("")

        # Methodology
        lines.append("## Sourcing and Methodology")
        lines.append("")

        if self.models_used:
            lines.append("**Mathematical Models:**")
            for model in self.models_used:
                lines.append(f"- {model}")
            lines.append("")

        if self.analytic_techniques:
            lines.append("**Analytic Techniques:**")
            for technique in self.analytic_techniques:
                lines.append(f"- {technique}")
            lines.append("")

        lines.append("**Overall Confidence:** " + self.overall_confidence.value)
        if self.confidence_rationale:
            lines.append(f"- *{self.confidence_rationale}*")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Prepared by: {self.prepared_by}*")
        if self.reviewed_by:
            lines.append(f"*Reviewed by: {self.reviewed_by}*")
        lines.append(f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*")
        lines.append("")
        lines.append(f"**{self.classification.value}//NOFORN**")

        return '\n'.join(lines)

    def format_json(self) -> Dict[str, Any]:
        """Format report as JSON (machine-readable)."""
        return {
            "classification": self.classification.value,
            "report_type": self.report_type.value,
            "title": self.title,
            "date": self.date.isoformat(),
            "originator": self.originator,
            "executive_summary": self.executive_summary,
            "key_judgments": [
                {
                    "judgment": kj.judgment,
                    "confidence": kj.confidence.value,
                    "likelihood": kj.likelihood.value if kj.likelihood else None,
                    "basis": kj.basis
                }
                for kj in self.key_judgments
            ],
            "situation_overview": self.situation_overview,
            "detailed_analysis": self.detailed_analysis,
            "outlook": self.outlook,
            "scenarios": [
                {
                    "name": s.name,
                    "probability": s.probability,
                    "likelihood": s.likelihood.value,
                    "description": s.description,
                    "indicators": s.indicators
                }
                for s in self.scenarios
            ],
            "most_likely_scenario": self.most_likely_scenario,
            "indicators": [
                {
                    "indicator": i.indicator,
                    "current_status": i.current_status,
                    "threshold": i.threshold,
                    "significance": i.significance
                }
                for i in self.indicators
            ],
            "information_gaps": [
                {
                    "gap": g.gap,
                    "impact": g.impact,
                    "collection_requirements": g.collection_requirements
                }
                for g in self.information_gaps
            ],
            "sources": self.sources,
            "analytic_techniques": self.analytic_techniques,
            "models_used": self.models_used,
            "overall_confidence": self.overall_confidence.value,
            "confidence_rationale": self.confidence_rationale,
            "risk_score": self.risk_score,
            "time_horizon": self.time_horizon,
            "geographic_focus": self.geographic_focus,
            "prepared_by": self.prepared_by,
            "reviewed_by": self.reviewed_by
        }

    def _wrap_text(self, text: str, width: int, indent: int = 0) -> str:
        """Wrap text to specified width with optional indent."""
        import textwrap
        wrapper = textwrap.TextWrapper(width=width, initial_indent=' ' * indent, subsequent_indent=' ' * indent)
        return '\n'.join(wrapper.wrap(text))

    def _confidence_marker(self, confidence: ConfidenceLevel) -> str:
        """Get confidence marker for text output."""
        markers = {
            ConfidenceLevel.HIGH: "[HIGH]",
            ConfidenceLevel.MODERATE: "[MOD] ",
            ConfidenceLevel.LOW: "[LOW] "
        }
        return markers.get(confidence, "[---]")

    def _confidence_badge_md(self, confidence: ConfidenceLevel) -> str:
        """Get confidence badge for markdown."""
        badges = {
            ConfidenceLevel.HIGH: "`HIGH`",
            ConfidenceLevel.MODERATE: "`MODERATE`",
            ConfidenceLevel.LOW: "`LOW`"
        }
        return badges.get(confidence, "`---`")


class ReportBuilder:
    """
    Builder for creating intelligence reports from GeoBotv1 analysis results.

    Example:
        >>> from geobot.interface import AnalystAgent, ReportBuilder
        >>>
        >>> # Run analysis
        >>> agent = AnalystAgent()
        >>> result = agent.analyze("Iran nuclear risk next 30 days")
        >>>
        >>> # Create professional report
        >>> builder = ReportBuilder()
        >>> report = builder.from_analysis_result(result)
        >>>
        >>> # Format and output
        >>> print(report.format_text())
        >>> with open("report.md", "w") as f:
        ...     f.write(report.format_markdown())
    """

    def __init__(
        self,
        classification: Classification = Classification.UNCLASSIFIED,
        originator: str = "GeoBotv1 Analysis System"
    ):
        """Initialize report builder."""
        self.classification = classification
        self.originator = originator

    def from_analysis_result(
        self,
        result: Any,  # AnalysisResult from analyst_agent
        report_type: ReportType = ReportType.ASSESSMENT
    ) -> IntelligenceReport:
        """
        Create intelligence report from AnalysisResult.

        Args:
            result: AnalysisResult from AnalystAgent
            report_type: Type of report to generate

        Returns:
            Formatted IntelligenceReport
        """
        report = IntelligenceReport(
            classification=self.classification,
            report_type=report_type,
            title=self._generate_title(result),
            date=result.timestamp,
            originator=self.originator
        )

        # Executive Summary from narrative
        report.executive_summary = self._extract_executive_summary(result.narrative_answer)

        # Key Judgments
        report.key_judgments = self._generate_key_judgments(result)

        # Analysis sections
        report.situation_overview = self._generate_situation_overview(result)
        report.detailed_analysis = result.narrative_answer

        # Scenarios from structured analysis
        report.scenarios = self._generate_scenarios(result.structured_analysis)
        if report.scenarios:
            report.most_likely_scenario = max(report.scenarios, key=lambda s: s.probability).name

        # Outlook
        report.outlook = self._generate_outlook(result)

        # Indicators
        report.indicators = self._generate_indicators(result)

        # Information Gaps
        report.information_gaps = self._generate_info_gaps(result)

        # Methodology
        report.models_used = result.modules_used
        report.analytic_techniques = self._identify_techniques(result)

        # Confidence
        report.overall_confidence = self._map_confidence(result.confidence)
        report.confidence_rationale = self._explain_confidence(result)

        # Metadata
        report.risk_score = result.structured_analysis.get('risk_score')
        report.time_horizon = self._format_time_horizon(result.structured_analysis)
        report.geographic_focus = result.structured_analysis.get('entities', [])

        return report

    def _generate_title(self, result: Any) -> str:
        """Generate report title from analysis."""
        entities = result.structured_analysis.get('entities', [])
        analysis_type = result.structured_analysis.get('analysis_type', 'assessment')

        if entities:
            entity_str = ', '.join(entities[:2])
            return f"{entity_str.title()}: {analysis_type.replace('_', ' ').title()}"
        else:
            return f"Geopolitical {analysis_type.replace('_', ' ').title()}"

    def _extract_executive_summary(self, narrative: str) -> str:
        """Extract concise executive summary (BLUF)."""
        # Take first 2-3 sentences as executive summary
        sentences = narrative.split('. ')
        summary = '. '.join(sentences[:3])
        if not summary.endswith('.'):
            summary += '.'
        return summary

    def _generate_key_judgments(self, result: Any) -> List[KeyJudgment]:
        """Generate key judgments with confidence."""
        judgments = []

        # Primary judgment from risk score
        risk = result.structured_analysis.get('risk_score')
        if risk is not None:
            likelihood = self._risk_to_likelihood(risk)
            judgment = KeyJudgment(
                judgment=f"Risk level assessed at {risk:.0%}",
                confidence=self._map_confidence(result.confidence),
                likelihood=likelihood,
                basis="Mathematical model ensemble (VAR, Hawkes, Bayesian)"
            )
            judgments.append(judgment)

        # Judgments from scenarios
        scenarios = result.structured_analysis.get('scenarios', [])
        if scenarios:
            most_likely = max(scenarios, key=lambda s: s.get('probability', 0))
            judgment = KeyJudgment(
                judgment=most_likely.get('description', 'Primary scenario'),
                confidence=ConfidenceLevel.MODERATE,
                likelihood=self._prob_to_likelihood(most_likely.get('probability', 0.5))
            )
            judgments.append(judgment)

        return judgments

    def _generate_situation_overview(self, result: Any) -> str:
        """Generate situation overview section."""
        entities = result.structured_analysis.get('entities', [])
        analysis_type = result.structured_analysis.get('analysis_type', '')

        parts = []
        if entities:
            parts.append(f"This assessment focuses on {', '.join(entities)}.")

        parts.append(f"Analysis type: {analysis_type.replace('_', ' ')}.")

        if result.structured_analysis.get('time_horizon_days'):
            days = result.structured_analysis['time_horizon_days']
            parts.append(f"Time horizon: {days} days.")

        return ' '.join(parts)

    def _generate_scenarios(self, structured: Dict) -> List[Scenario]:
        """Generate scenario objects."""
        scenario_data = structured.get('scenarios', [])
        scenarios = []

        for s in scenario_data:
            probability = s.get('probability', 0.5)
            scenario = Scenario(
                name=s.get('name', 'Scenario'),
                probability=probability,
                likelihood=self._prob_to_likelihood(probability),
                description=s.get('description', ''),
                indicators=[]
            )
            scenarios.append(scenario)

        return scenarios

    def _generate_outlook(self, result: Any) -> str:
        """Generate outlook section."""
        time_horizon = result.structured_analysis.get('time_horizon_days', 30)
        return f"Over the next {time_horizon} days, the situation is expected to evolve based on the scenarios outlined above."

    def _generate_indicators(self, result: Any) -> List[Indicator]:
        """Generate indicators to monitor."""
        # Extract from key drivers
        indicators = []
        drivers = result.structured_analysis.get('key_drivers', [])

        for driver in drivers[:5]:  # Top 5
            indicator = Indicator(
                indicator=f"{driver.get('country', 'Entity')} dynamics",
                current_status=f"{driver.get('impact', 0):.0%} impact observed",
                threshold="Monitor for >75% impact",
                significance="Important"
            )
            indicators.append(indicator)

        return indicators

    def _generate_info_gaps(self, result: Any) -> List[InformationGap]:
        """Generate information gaps."""
        gaps = []

        # Standard gaps based on confidence
        if result.confidence < 0.7:
            gap = InformationGap(
                gap="Limited intelligence on key actor intentions",
                impact="Reduces forecast accuracy by 15-20%",
                collection_requirements="HUMINT from regional sources"
            )
            gaps.append(gap)

        return gaps

    def _identify_techniques(self, result: Any) -> List[str]:
        """Identify analytic techniques used."""
        techniques = ["Structured Analytic Techniques", "Quantitative Analysis"]

        if 'hawkes' in result.modules_used:
            techniques.append("Point Process Analysis")
        if 'var_model' in result.modules_used:
            techniques.append("Time-Series Econometrics")
        if 'do_calculus' in result.modules_used:
            techniques.append("Causal Inference")

        return techniques

    def _map_confidence(self, confidence: float) -> ConfidenceLevel:
        """Map numerical confidence to categorical."""
        if confidence >= 0.80:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.60:
            return ConfidenceLevel.MODERATE
        else:
            return ConfidenceLevel.LOW

    def _explain_confidence(self, result: Any) -> str:
        """Explain confidence assessment."""
        factors = []

        if result.confidence >= 0.80:
            factors.append("Multiple corroborating models")
            factors.append("High-quality intelligence sources")
        elif result.confidence >= 0.60:
            factors.append("Model agreement with some variance")
            factors.append("Adequate intelligence coverage")
        else:
            factors.append("Limited model agreement")
            factors.append("Intelligence gaps present")

        return "; ".join(factors) + "."

    def _risk_to_likelihood(self, risk: float) -> Likelihood:
        """Convert risk score to likelihood."""
        if risk > 0.90:
            return Likelihood.VERY_LIKELY
        elif risk > 0.70:
            return Likelihood.LIKELY
        elif risk > 0.30:
            return Likelihood.ROUGHLY_EVEN
        elif risk > 0.10:
            return Likelihood.UNLIKELY
        else:
            return Likelihood.REMOTE

    def _prob_to_likelihood(self, prob: float) -> Likelihood:
        """Convert probability to likelihood."""
        return self._risk_to_likelihood(prob)

    def _format_time_horizon(self, structured: Dict) -> Optional[str]:
        """Format time horizon string."""
        days = structured.get('time_horizon_days')
        if days:
            if days == 30:
                return "30 days (short-term)"
            elif days == 90:
                return "90 days (medium-term)"
            elif days == 365:
                return "365 days (long-term)"
            else:
                return f"{days} days"
        return None
