"""
GeoBot CLI - Command Line Interface for Geopolitical Analysis

Provides command-line access to GeoBot 2.0 analytical framework,
Bayesian forecasting, causal models, and simulation capabilities.
"""

import argparse
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Try importing core functionality
try:
    from geobot.analysis.engine import AnalyticalEngine
    from geobot.analysis.formatter import AnalysisFormatter
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False

try:
    from geobot.bayes.forecasting import BayesianForecaster, PriorType, GeopoliticalPrior, EvidenceType, EvidenceUpdate
    import numpy as np
    HAS_BAYES = True
except ImportError:
    HAS_BAYES = False

try:
    from geobot.inference.bayesian_engine import BeliefUpdater
    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False

try:
    from geobot.causal.structural_model import (
        StructuralCausalModel,
        CausalVariable,
        CausalEdge,
        Intervention,
        get_predefined_scm
    )
    HAS_CAUSAL = True
except ImportError:
    HAS_CAUSAL = False

try:
    from geobot.monitoring.ticker import GeopoliticalTicker, start_ticker
    from geobot.data_ingestion.rss_scraper import RSSFeedScraper
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    if not HAS_ANALYSIS:
        missing.append("analysis (GeoBot 2.0 framework)")
    if not HAS_BAYES:
        missing.append("bayes (Bayesian forecasting)")
    if not HAS_INFERENCE:
        missing.append("inference (belief updating)")
    if not HAS_CAUSAL:
        missing.append("causal (structural causal models)")
    if not HAS_MONITORING:
        missing.append("monitoring (RSS ticker) - install: pip install feedparser requests")

    if missing:
        print("Warning: Some modules are not available:")
        for m in missing:
            print(f"  - {m}")
        print()


def cmd_analyze(args):
    """Run GeoBot 2.0 analytical framework."""
    if not HAS_ANALYSIS:
        print("Error: GeoBot 2.0 analysis module not available")
        return 1

    print("=" * 80)
    print("GeoBot 2.0 Analytical Framework")
    print("=" * 80)
    print()

    # Create engine
    engine = AnalyticalEngine()

    # Build context from scenario
    context = {
        'scenario': args.scenario,
        'region': getattr(args, 'region', 'Unknown'),
        'timeframe': getattr(args, 'timeframe', 'current'),
    }

    # Add any additional context from JSON file
    if args.context_file:
        try:
            with open(args.context_file, 'r') as f:
                additional_context = json.load(f)
                context.update(additional_context)
        except Exception as e:
            print(f"Warning: Could not load context file: {e}")

    # Run analysis
    query = f"Analyze {args.scenario} scenario"
    if args.query:
        query = args.query

    print(f"Query: {query}")
    print()

    result = engine.analyze(query, context)
    print(result)

    return 0


def cmd_forecast(args):
    """Run Bayesian forecasting."""
    if not HAS_BAYES:
        print("Error: Bayesian forecasting module not available")
        print("Install with: pip install numpy scipy")
        return 1

    print("=" * 80)
    print(f"Bayesian Forecasting: {args.scenario}")
    print("=" * 80)
    print()

    # Predefined scenarios
    scenarios = {
        'taiwan': {
            'parameter': 'invasion_probability_12mo',
            'prior_mean': 0.10,
            'prior_alpha': 2.0,
            'prior_beta': 18.0,
            'description': 'PRC invasion of Taiwan within 12 months'
        },
        'ukraine': {
            'parameter': 'escalation_probability',
            'prior_mean': 0.25,
            'prior_alpha': 5.0,
            'prior_beta': 15.0,
            'description': 'Major escalation in Ukraine conflict'
        },
        'iran': {
            'parameter': 'conflict_probability',
            'prior_mean': 0.15,
            'prior_alpha': 3.0,
            'prior_beta': 17.0,
            'description': 'Military conflict with Iran'
        }
    }

    scenario = scenarios.get(args.scenario.lower())
    if not scenario:
        print(f"Unknown scenario: {args.scenario}")
        print(f"Available scenarios: {', '.join(scenarios.keys())}")
        return 1

    # Create forecaster
    forecaster = BayesianForecaster()

    # Set prior
    prior = GeopoliticalPrior(
        parameter_name=scenario['parameter'],
        prior_type=PriorType.BETA,
        parameters={'alpha': scenario['prior_alpha'], 'beta': scenario['prior_beta']},
        rationale=f"Historical base rate analysis for {scenario['description']}"
    )

    print(f"Scenario: {scenario['description']}")
    print(f"Prior: Beta(α={scenario['prior_alpha']}, β={scenario['prior_beta']})")
    print(f"Prior mean: {scenario['prior_mean']:.2%}")
    print()

    # Sample from prior
    samples = forecaster.sample_prior(prior, n_samples=10000)
    print(f"Prior 95% credible interval: [{np.percentile(samples, 2.5):.2%}, {np.percentile(samples, 97.5):.2%}]")
    print()

    # If evidence file provided, update beliefs
    if args.evidence_file:
        try:
            with open(args.evidence_file, 'r') as f:
                evidence_data = json.load(f)

            print("Updating beliefs with evidence...")
            print()

            for i, ev in enumerate(evidence_data, 1):
                print(f"Evidence {i}: {ev.get('description', 'N/A')}")
                print(f"  Observation: {ev.get('observation', 'N/A')}")
                print(f"  Reliability: {ev.get('reliability', 1.0):.1%}")
                # Note: Actual belief updating requires likelihood function
                # which is complex to specify in JSON. This is a demonstration.

        except Exception as e:
            print(f"Warning: Could not process evidence file: {e}")

    print()
    print("For detailed Bayesian analysis, see examples/taiwan_situation_room.py")

    return 0


def cmd_intervene(args):
    """Run causal intervention analysis."""
    if not HAS_CAUSAL:
        print("Error: Causal analysis module not available")
        print("Install with: pip install numpy scipy")
        return 1

    print("=" * 80)
    print(f"Causal Intervention Analysis: {args.scenario}")
    print("=" * 80)
    print()

    # Get predefined SCM
    if args.scenario == 'sanctions':
        scm = get_predefined_scm('sanctions')
        print("Scenario: Economic sanctions impact on conflict risk")
        print()
        print("Causal structure:")
        print("  sanctions → economic_impact → political_pressure → conflict_risk")
        print()

        if args.intervention:
            try:
                # Parse intervention (format: variable=value)
                var, val = args.intervention.split('=')
                value = float(val)

                print(f"Intervention: do({var} = {value})")
                print()

                # Run intervention
                intervention = Intervention(variable=var, value=value)
                results = scm.intervene([intervention], n_samples=1000)

                print("Results (mean values under intervention):")
                for var_name, samples in results.items():
                    print(f"  {var_name}: {samples.mean():.3f} (std: {samples.std():.3f})")

            except Exception as e:
                print(f"Error processing intervention: {e}")
                print("Format: --intervention variable=value")
                return 1
        else:
            print("Observational mode (no intervention)")
            samples = scm.simulate(n_samples=1000)
            print("Results (mean values):")
            for var_name, var_samples in samples.items():
                print(f"  {var_name}: {var_samples.mean():.3f} (std: {var_samples.std():.3f})")

    elif args.scenario == 'escalation':
        scm = get_predefined_scm('conflict_escalation')
        print("Scenario: Conflict escalation dynamics")
        print()
        print("Causal structure:")
        print("  initial_tensions + external_pressure → military_posturing → escalation_risk")
        print()

        if args.intervention:
            print(f"Custom intervention: {args.intervention}")
            print("(Intervention logic similar to sanctions scenario)")
        else:
            print("Observational mode (no intervention)")
            samples = scm.simulate(n_samples=1000)
            print("Results (mean values):")
            for var_name, var_samples in samples.items():
                print(f"  {var_name}: {var_samples.mean():.3f} (std: {var_samples.std():.3f})")

    else:
        print(f"Unknown scenario: {args.scenario}")
        print("Available scenarios: sanctions, escalation")
        return 1

    print()
    return 0


def cmd_monitor(args):
    """Run real-time geopolitical intelligence ticker."""
    if not HAS_MONITORING:
        print("Error: Monitoring module not available")
        print("Install with: pip install feedparser requests")
        return 1

    print("=" * 80)
    print("Real-Time Geopolitical Intelligence Ticker")
    print("=" * 80)
    print()

    # Test mode - single update
    if args.test:
        print("Running single update (test mode)...")
        print()
        ticker = GeopoliticalTicker(
            update_interval_minutes=args.interval,
            output_dir=Path(args.output) if args.output else None,
            use_ai_analysis=not args.no_ai
        )
        ticker.run_once()
        return 0

    # Continuous monitoring mode
    print(f"Starting continuous monitoring...")
    print(f"Update interval: {args.interval} minutes")
    print(f"AI Analysis: {'Disabled' if args.no_ai else 'Enabled'}")
    if args.output:
        print(f"Output directory: {args.output}")
    print()
    print("Press Ctrl+C to stop")
    print()

    start_ticker(
        interval_minutes=args.interval,
        output_dir=args.output,
        use_ai=not args.no_ai
    )

    return 0


def cmd_version(args):
    """Display version information."""
    try:
        from geobot import __version__
        print(f"GeoBot v{__version__}")
    except:
        print("GeoBot (version unknown)")

    print()
    print("Modules:")
    print(f"  Analysis Framework: {'✓' if HAS_ANALYSIS else '✗'}")
    print(f"  Bayesian Forecasting: {'✓' if HAS_BAYES else '✗'}")
    print(f"  Belief Updating: {'✓' if HAS_INFERENCE else '✗'}")
    print(f"  Causal Models: {'✓' if HAS_CAUSAL else '✗'}")
    print(f"  Real-Time Monitoring: {'✓' if HAS_MONITORING else '✗'}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='geobot',
        description='GeoBot - Geopolitical Analysis and Forecasting Framework',
        epilog='For examples, see the examples/ directory in the repository'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Run GeoBot 2.0 analytical framework'
    )
    analyze_parser.add_argument(
        'scenario',
        help='Scenario to analyze'
    )
    analyze_parser.add_argument(
        '--query',
        help='Custom analysis query'
    )
    analyze_parser.add_argument(
        '--region',
        help='Geographic region'
    )
    analyze_parser.add_argument(
        '--timeframe',
        default='current',
        help='Analysis timeframe'
    )
    analyze_parser.add_argument(
        '--context-file',
        help='JSON file with additional context'
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # Forecast command
    forecast_parser = subparsers.add_parser(
        'forecast',
        help='Run Bayesian forecasting'
    )
    forecast_parser.add_argument(
        '--scenario',
        required=True,
        choices=['taiwan', 'ukraine', 'iran'],
        help='Predefined scenario to forecast'
    )
    forecast_parser.add_argument(
        '--evidence-file',
        help='JSON file with evidence for belief updating'
    )
    forecast_parser.set_defaults(func=cmd_forecast)

    # Intervene command
    intervene_parser = subparsers.add_parser(
        'intervene',
        help='Run causal intervention analysis'
    )
    intervene_parser.add_argument(
        'scenario',
        choices=['sanctions', 'escalation'],
        help='Causal scenario'
    )
    intervene_parser.add_argument(
        '--intervention',
        help='Intervention to apply (format: variable=value)'
    )
    intervene_parser.set_defaults(func=cmd_intervene)

    # Monitor command
    monitor_parser = subparsers.add_parser(
        'monitor',
        help='Run real-time geopolitical intelligence ticker'
    )
    monitor_parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Update interval in minutes (default: 30)'
    )
    monitor_parser.add_argument(
        '--output',
        help='Output directory for insights (default: ./ticker_output)'
    )
    monitor_parser.add_argument(
        '--no-ai',
        action='store_true',
        help='Disable GeoBot 2.0 AI analysis'
    )
    monitor_parser.add_argument(
        '--test',
        action='store_true',
        help='Run single update and exit (test mode)'
    )
    monitor_parser.set_defaults(func=cmd_monitor)

    # Version command
    version_parser = subparsers.add_parser(
        'version',
        help='Display version information'
    )
    version_parser.set_defaults(func=cmd_version)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Check dependencies
    check_dependencies()

    # Run command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
