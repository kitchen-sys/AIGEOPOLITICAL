"""
LLM-Powered Analyst Agent for GeoBotv1

Interprets natural language questions and routes to appropriate mathematical modules.

The agent:
1. Parses user question (What is being asked?)
2. Determines which modules to call (Hawkes, VAR, OT, SCM, etc.)
3. Executes analysis
4. Formats human-readable answer + structured analysis block

Supports multiple LLM backends: Mistral, GPT-4, Claude, or local models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json


class AnalysisType(Enum):
    """Type of analysis requested."""
    FORECAST = "forecast"  # Predict future outcomes
    EXPLAIN = "explain"  # Explain past events
    COUNTERFACTUAL = "counterfactual"  # What-if scenarios
    COMPARISON = "comparison"  # Compare scenarios
    RISK_ASSESSMENT = "risk_assessment"  # Risk scoring
    CONTAGION = "contagion"  # Conflict spread analysis
    CAUSALITY = "causality"  # Causal relationships
    TREND = "trend"  # Time-series trends


class ModuleCall(Enum):
    """GeoBotv1 modules that can be called."""
    VAR = "var_model"  # Vector Autoregression
    HAWKES = "hawkes_process"  # Conflict contagion
    OPTIMAL_TRANSPORT = "optimal_transport"  # Scenario comparison
    DO_CALCULUS = "do_calculus"  # Interventions
    SYNTHETIC_CONTROL = "synthetic_control"  # Policy evaluation
    DID = "difference_in_differences"  # Regime change
    BAYESIAN = "bayesian_inference"  # Belief updating
    KALMAN = "kalman_filter"  # State estimation
    HMM = "hidden_markov"  # Regime detection
    GNN = "graph_neural_network"  # Network analysis


@dataclass
class QueryIntent:
    """Parsed intent from user query."""
    original_query: str
    analysis_type: AnalysisType
    entities: List[str]  # Countries, organizations, etc.
    time_horizon: Optional[int] = None  # Days/months to forecast
    confidence_required: bool = True
    modules_to_call: List[ModuleCall] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result from analyst agent."""
    query: str
    narrative_answer: str  # Human-readable explanation
    structured_analysis: Dict[str, Any]  # Machine-readable data
    timestamp: datetime
    confidence: float  # Overall confidence in analysis
    modules_used: List[str]
    execution_time: float  # seconds
    warnings: List[str] = field(default_factory=list)


class AnalystAgent:
    """
    LLM-powered analyst that interprets questions and routes to GeoBot modules.

    Example:
        >>> agent = AnalystAgent(llm_backend="mistral")
        >>>
        >>> # Natural language query
        >>> result = agent.analyze("What is the risk of conflict spreading "
        ...                        "from Syria to Lebanon in the next 30 days?")
        >>>
        >>> # Get narrative answer
        >>> print(result.narrative_answer)
        >>>
        >>> # Get structured data
        >>> print(f"Risk score: {result.structured_analysis['risk_score']}")
        >>> print(f"Key drivers: {result.structured_analysis['drivers']}")
    """

    def __init__(
        self,
        llm_backend: str = "mistral",
        verbosity: str = "standard",
        include_uncertainty: bool = True,
        enable_answer_logging: bool = False,
        answer_db_path: Optional[str] = None
    ):
        """
        Initialize analyst agent.

        Args:
            llm_backend: LLM to use ("mistral", "gpt-4", "claude-3", "local")
            verbosity: Response detail level ("concise", "standard", "detailed")
            include_uncertainty: Include uncertainty quantification
            enable_answer_logging: Enable automatic logging to answer database
            answer_db_path: Path to answer database (default: answer_database.db)
        """
        self.llm_backend = llm_backend
        self.verbosity = verbosity
        self.include_uncertainty = include_uncertainty
        self.enable_answer_logging = enable_answer_logging

        # In production, initialize actual LLM client here
        # self.llm_client = MistralClient(api_key=...)

        # Answer logging
        self.answer_db = None
        if enable_answer_logging:
            from .answer_database import AnswerDatabase
            self.answer_db = AnswerDatabase(answer_db_path or "answer_database.db")

    def analyze(self, query: str, session_id: str = "", analyst_id: str = "") -> AnalysisResult:
        """
        Analyze a natural language query.

        Args:
            query: User's question in natural language
            session_id: Optional session identifier for tracking
            analyst_id: Optional analyst identifier

        Returns:
            AnalysisResult with narrative + structured analysis
        """
        start_time = datetime.utcnow()

        # Step 1: Parse query intent
        intent = self._parse_query_intent(query)

        # Step 2: Route to appropriate modules
        analysis_data = self._execute_analysis(intent)

        # Step 3: Format response
        narrative = self._generate_narrative(intent, analysis_data)
        structured = self._structure_analysis(intent, analysis_data)

        # Step 4: Compute execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()

        result = AnalysisResult(
            query=query,
            narrative_answer=narrative,
            structured_analysis=structured,
            timestamp=start_time,
            confidence=analysis_data.get('confidence', 0.8),
            modules_used=[m.value for m in intent.modules_to_call],
            execution_time=execution_time,
            warnings=analysis_data.get('warnings', [])
        )

        # Log to answer database if enabled
        if self.enable_answer_logging and self.answer_db:
            try:
                self.answer_db.log_answer(
                    query=query,
                    query_intent={
                        'analysis_type': intent.analysis_type.value,
                        'entities': intent.entities,
                        'time_horizon': intent.time_horizon
                    },
                    answer_narrative=narrative,
                    answer_structured=structured,
                    model_used=self.llm_backend,
                    modules_invoked=[m.value for m in intent.modules_to_call],
                    processing_time_seconds=execution_time,
                    session_id=session_id,
                    analyst_id=analyst_id,
                    auto_rate=True  # Automatic quality rating
                )
            except Exception as e:
                # Don't fail analysis if logging fails
                print(f"Warning: Failed to log answer to database: {e}")

        return result

    def _parse_query_intent(self, query: str) -> QueryIntent:
        """
        Parse user query to determine intent.

        In production, this uses the LLM to extract:
        - Analysis type (forecast, explain, counterfactual, etc.)
        - Entities (countries, organizations)
        - Time horizon
        - Which modules to call

        For now, uses heuristics.
        """
        query_lower = query.lower()

        # Determine analysis type
        analysis_type = AnalysisType.FORECAST  # default
        if any(w in query_lower for w in ["risk", "likelihood", "probability"]):
            analysis_type = AnalysisType.RISK_ASSESSMENT
        elif any(w in query_lower for w in ["spread", "contagion", "spillover"]):
            analysis_type = AnalysisType.CONTAGION
        elif any(w in query_lower for w in ["what if", "counterfactual", "suppose"]):
            analysis_type = AnalysisType.COUNTERFACTUAL
        elif any(w in query_lower for w in ["explain", "why", "caused"]):
            analysis_type = AnalysisType.EXPLAIN
        elif any(w in query_lower for w in ["compare", "difference", "versus"]):
            analysis_type = AnalysisType.COMPARISON
        elif any(w in query_lower for w in ["trend", "trajectory", "pattern"]):
            analysis_type = AnalysisType.TREND
        elif any(w in query_lower for w in ["cause", "causal", "mechanism"]):
            analysis_type = AnalysisType.CAUSALITY

        # Extract entities (simplified - in production, use NER)
        entities = []
        countries = ["iran", "syria", "iraq", "russia", "china", "ukraine",
                    "israel", "lebanon", "turkey", "pakistan", "india"]
        for country in countries:
            if country in query_lower:
                entities.append(country.title())

        # Extract time horizon
        time_horizon = None
        if "30 days" in query_lower or "month" in query_lower:
            time_horizon = 30
        elif "90 days" in query_lower or "quarter" in query_lower:
            time_horizon = 90
        elif "year" in query_lower or "12 months" in query_lower:
            time_horizon = 365

        # Determine modules to call based on analysis type
        modules_to_call = self._select_modules(analysis_type, query_lower)

        return QueryIntent(
            original_query=query,
            analysis_type=analysis_type,
            entities=entities,
            time_horizon=time_horizon,
            modules_to_call=modules_to_call,
            parameters={}
        )

    def _select_modules(self, analysis_type: AnalysisType, query: str) -> List[ModuleCall]:
        """Determine which GeoBot modules to call."""
        modules = []

        if analysis_type == AnalysisType.CONTAGION:
            modules.append(ModuleCall.HAWKES)
            if "interdependencies" in query or "spillover" in query:
                modules.append(ModuleCall.VAR)

        elif analysis_type == AnalysisType.FORECAST:
            modules.append(ModuleCall.VAR)
            if "regime" in query or "shift" in query:
                modules.append(ModuleCall.HMM)
            if "uncertainty" in query:
                modules.append(ModuleCall.BAYESIAN)

        elif analysis_type == AnalysisType.COUNTERFACTUAL:
            modules.append(ModuleCall.DO_CALCULUS)
            if "sanctions" in query or "policy" in query:
                modules.append(ModuleCall.SYNTHETIC_CONTROL)

        elif analysis_type == AnalysisType.COMPARISON:
            modules.append(ModuleCall.OPTIMAL_TRANSPORT)

        elif analysis_type == AnalysisType.CAUSALITY:
            modules.append(ModuleCall.DO_CALCULUS)
            if "granger" in query:
                modules.append(ModuleCall.VAR)

        elif analysis_type == AnalysisType.RISK_ASSESSMENT:
            modules.append(ModuleCall.HAWKES)
            modules.append(ModuleCall.VAR)

        elif analysis_type == AnalysisType.TREND:
            modules.append(ModuleCall.KALMAN)
            modules.append(ModuleCall.HMM)

        return modules

    def _execute_analysis(self, intent: QueryIntent) -> Dict[str, Any]:
        """
        Execute analysis by calling appropriate modules.

        In production, this actually calls GeoBot modules.
        For now, returns simulated results.
        """
        results = {
            'analysis_type': intent.analysis_type.value,
            'entities': intent.entities,
            'confidence': 0.75,
            'warnings': []
        }

        # Simulate module calls
        if ModuleCall.HAWKES in intent.modules_to_call:
            results['hawkes'] = {
                'branching_ratio': 0.42,
                'baseline_intensity': 0.15,
                'contagion_risk': 0.68
            }

        if ModuleCall.VAR in intent.modules_to_call:
            results['var'] = {
                'forecast_horizon': intent.time_horizon or 30,
                'shock_propagation': {
                    intent.entities[0] if intent.entities else 'Country_A': 0.35,
                    intent.entities[1] if len(intent.entities) > 1 else 'Country_B': 0.22
                }
            }

        if ModuleCall.DO_CALCULUS in intent.modules_to_call:
            results['intervention_effect'] = {
                'estimated_effect': -0.25,
                'confidence_interval': (-0.45, -0.05)
            }

        return results

    def _generate_narrative(self, intent: QueryIntent, analysis_data: Dict) -> str:
        """
        Generate human-readable narrative answer.

        In production, uses LLM to generate natural language from structured data.
        """
        narrative_parts = []

        # Opening
        if intent.analysis_type == AnalysisType.CONTAGION:
            narrative_parts.append(
                f"Based on Hawkes process analysis of conflict contagion dynamics:"
            )
        elif intent.analysis_type == AnalysisType.FORECAST:
            narrative_parts.append(
                f"Based on Vector Autoregression (VAR) analysis:"
            )
        elif intent.analysis_type == AnalysisType.COUNTERFACTUAL:
            narrative_parts.append(
                f"Based on do-calculus intervention simulation:"
            )

        # Main finding
        if 'hawkes' in analysis_data:
            br = analysis_data['hawkes']['branching_ratio']
            risk = analysis_data['hawkes']['contagion_risk']
            narrative_parts.append(
                f"\nThe branching ratio is {br:.2f}, indicating {'subcritical (stable)' if br < 1 else 'supercritical (explosive)'} dynamics. "
                f"Estimated contagion risk over the next {intent.time_horizon or 30} days: {risk:.0%}."
            )

        if 'var' in analysis_data:
            narrative_parts.append(
                f"\nVAR model shows shock propagation across countries:"
            )
            for country, impact in analysis_data['var']['shock_propagation'].items():
                narrative_parts.append(f"  • {country}: {impact:.0%} impact")

        if 'intervention_effect' in analysis_data:
            effect = analysis_data['intervention_effect']['estimated_effect']
            ci = analysis_data['intervention_effect']['confidence_interval']
            narrative_parts.append(
                f"\nEstimated intervention effect: {effect:.2f} (95% CI: [{ci[0]:.2f}, {ci[1]:.2f}])"
            )

        # Confidence statement
        if self.include_uncertainty:
            confidence = analysis_data.get('confidence', 0.75)
            narrative_parts.append(
                f"\n\nOverall confidence in this analysis: {confidence:.0%}"
            )

        # Warnings
        if analysis_data.get('warnings'):
            narrative_parts.append("\n\n⚠️ Warnings:")
            for warning in analysis_data['warnings']:
                narrative_parts.append(f"  • {warning}")

        return '\n'.join(narrative_parts)

    def _structure_analysis(self, intent: QueryIntent, analysis_data: Dict) -> Dict[str, Any]:
        """
        Create structured analysis block (machine-readable).

        This is the "analysis block" that contains:
        - Risk scores
        - Scenarios
        - Key drivers
        - Uncertainty ranges
        """
        structured = {
            'analysis_type': intent.analysis_type.value,
            'timestamp': datetime.utcnow().isoformat(),
            'entities': intent.entities,
            'time_horizon_days': intent.time_horizon,
            'risk_score': None,
            'scenarios': [],
            'key_drivers': [],
            'uncertainty': {},
            'raw_results': analysis_data
        }

        # Compute risk score
        if 'hawkes' in analysis_data:
            structured['risk_score'] = analysis_data['hawkes']['contagion_risk']

        # Extract key drivers
        if 'var' in analysis_data:
            structured['key_drivers'] = [
                {'country': country, 'impact': impact}
                for country, impact in analysis_data['var']['shock_propagation'].items()
            ]

        # Scenarios
        if intent.analysis_type == AnalysisType.CONTAGION:
            structured['scenarios'] = [
                {'name': 'baseline', 'probability': 0.5, 'description': 'No contagion'},
                {'name': 'limited_spread', 'probability': 0.3, 'description': 'Limited regional spread'},
                {'name': 'full_contagion', 'probability': 0.2, 'description': 'Full regional contagion'}
            ]

        # Uncertainty quantification
        if self.include_uncertainty:
            structured['uncertainty'] = {
                'confidence': analysis_data.get('confidence', 0.75),
                'data_quality': 'high',
                'model_assumptions': ['stationarity', 'independence']
            }

        return structured

    def chat(self, query: str) -> str:
        """
        Simple chat interface that returns only narrative answer.

        Args:
            query: User question

        Returns:
            Narrative answer string
        """
        result = self.analyze(query)
        return result.narrative_answer

    def batch_analyze(self, queries: List[str]) -> List[AnalysisResult]:
        """
        Analyze multiple queries in batch.

        Args:
            queries: List of questions

        Returns:
            List of AnalysisResult objects
        """
        return [self.analyze(q) for q in queries]

    def explain_capabilities(self) -> str:
        """Return description of agent capabilities."""
        return """
GeoBotv1 Analyst Agent Capabilities:

📊 Analysis Types:
  • Forecast: Predict future outcomes using VAR, Hawkes, time-series
  • Risk Assessment: Quantify conflict/instability risk
  • Contagion: Model conflict spread using Hawkes processes
  • Counterfactual: Simulate what-if scenarios with do-calculus
  • Causality: Identify causal relationships and mechanisms
  • Comparison: Compare scenarios using optimal transport
  • Trend: Detect regime shifts and trajectory changes
  • Explain: Understand why past events occurred

🔧 Available Modules:
  • VAR (Vector Autoregression): Multi-country interdependencies
  • Hawkes Processes: Conflict contagion and escalation
  • Do-Calculus: Intervention simulation
  • Synthetic Control: Policy impact evaluation
  • Difference-in-Differences: Regime change analysis
  • Optimal Transport: Scenario comparison
  • Bayesian Inference: Belief updating with intelligence
  • Kalman Filter: State estimation and tracking
  • Hidden Markov Models: Regime detection
  • Graph Neural Networks: Network analysis

💬 Natural Language Interface:
  Ask questions like:
  • "What is the risk of conflict spreading from Syria to Lebanon?"
  • "How would sanctions on Iran affect regional stability?"
  • "What caused the escalation in tensions last month?"
  • "Compare the current scenario to 2019"
  • "Forecast Iran-Saudi relations over next 90 days"

📈 Output Format:
  • Narrative Answer: Human-readable explanation
  • Structured Analysis: Machine-readable data with:
    - Risk scores
    - Scenarios (baseline, optimistic, pessimistic)
    - Key drivers
    - Uncertainty quantification
    - Confidence intervals
"""
