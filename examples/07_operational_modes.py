"""
Example 7: GeoBotv1 Operational Modes

Demonstrates the three operational modes for GeoBotv1:

1. INTERACTIVE MODE (Chat & Ask)
   - Natural language Q&A with LLM integration
   - On-demand analysis and forecasts
   - Structured output: narrative + analysis blocks

2. WATCH MODE (Live Monitoring)
   - Autonomous monitoring of feeds
   - Automated model updates
   - Real-time alert generation
   - Chat override capability

3. REPLAY MODE (Forensic Analysis)
   - Historical backtesting
   - "How did we get here?" investigations
   - Forecast evolution tracking
   - Counterfactual what-if scenarios

This shows how GeoBotv1 transforms from a mathematical framework into
an operational intelligence system with natural language interface.
"""

import sys
sys.path.append('..')

from datetime import datetime, timedelta
import time

from geobot.interface import (
    ModeManager,
    OperationalMode,
    AnalystAgent,
    WatchDaemon,
    ReplayAnalyzer,
    AlertLevel
)


def demo_interactive_mode():
    """Demonstrate Interactive Analyst Mode."""
    print("\n" + "="*80)
    print("1. INTERACTIVE MODE - Chat & Ask with LLM Integration")
    print("="*80)

    # Create analyst agent
    agent = AnalystAgent(
        llm_backend="mistral",  # or "gpt-4", "claude-3", "local"
        verbosity="standard",
        include_uncertainty=True
    )

    print("\n📝 Natural Language Interface:")
    print("   Type questions in plain English - the LLM interprets and routes to")
    print("   appropriate mathematical modules (VAR, Hawkes, Do-Calculus, etc.)\n")

    # Example questions
    questions = [
        "What is the risk of conflict spreading from Syria to Lebanon in the next 30 days?",
        "How would sanctions on Iran affect regional stability?",
        "Compare the current Middle East situation to 2019",
        "What caused the escalation in tensions between India and Pakistan last month?",
        "Forecast Russia-Ukraine dynamics over next 90 days"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*80}")
        print(f"Question {i}: {question}")
        print(f"{'─'*80}")

        # Analyze question
        result = agent.analyze(question)

        # Show narrative answer
        print(f"\n💬 Narrative Answer:")
        print(result.narrative_answer)

        # Show structured analysis block
        print(f"\n📊 Structured Analysis Block:")
        print(f"   Analysis Type: {result.structured_analysis['analysis_type']}")
        if result.structured_analysis.get('risk_score'):
            print(f"   Risk Score: {result.structured_analysis['risk_score']:.2%}")
        if result.structured_analysis.get('entities'):
            print(f"   Entities: {', '.join(result.structured_analysis['entities'])}")
        if result.structured_analysis.get('key_drivers'):
            print(f"   Key Drivers:")
            for driver in result.structured_analysis['key_drivers']:
                print(f"     • {driver['country']}: {driver['impact']:.0%} impact")
        if result.structured_analysis.get('scenarios'):
            print(f"   Scenarios:")
            for scenario in result.structured_analysis['scenarios']:
                print(f"     • {scenario['name']}: {scenario['probability']:.0%} probability")

        # Show metadata
        print(f"\n⚙️  Execution Details:")
        print(f"   Modules Used: {', '.join(result.modules_used)}")
        print(f"   Execution Time: {result.execution_time:.2f}s")
        print(f"   Confidence: {result.confidence:.0%}")

        if i < len(questions):
            print("\n   [Next question...]\n")

    print("\n" + "="*80)
    print("✓ Interactive Mode: Analyst can ask anything in natural language!")
    print("="*80)


def demo_watch_mode():
    """Demonstrate Watch Mode (Live Monitoring)."""
    print("\n" + "="*80)
    print("2. WATCH MODE - Autonomous Live Monitoring")
    print("="*80)

    print("\n🔍 Watch Mode runs continuously in background, monitoring:")
    print("   • Intelligence feeds (RSS, APIs, documents)")
    print("   • Model performance and drift")
    print("   • Risk threshold breaches")
    print("   • Regime shifts and anomalies")
    print("   • Conflict contagion indicators\n")

    # Create watch daemon
    daemon = WatchDaemon(
        check_interval=10,  # Check every 10 seconds (for demo)
        alert_threshold=AlertLevel.MEDIUM,
        auto_update_models=True
    )

    # Configure thresholds
    print("⚙️  Configuring alert thresholds:")
    daemon.set_threshold('risk_score', 0.7, AlertLevel.MEDIUM)
    daemon.set_threshold('branching_ratio', 0.85, AlertLevel.HIGH)
    daemon.set_threshold('anomaly_score', 3.0, AlertLevel.MEDIUM)

    # Register alert handler
    def custom_alert_handler(alert):
        # In production, this could:
        # - Send email
        # - Post to Slack/Teams
        # - Trigger webhook
        # - Log to database
        pass  # Console output handled by daemon

    daemon.register_alert_handler(custom_alert_handler)

    print("\n🚀 Starting watch daemon...")
    daemon.start()

    # Let it run for a bit
    print("\n⏱️  Monitoring for 30 seconds (generating simulated checks)...")
    try:
        time.sleep(30)

        # Show status
        print("\n")
        daemon.print_status()

        # Chat override while monitoring
        print("\n💬 Chat Override (ask questions while daemon runs):")
        answer = daemon.ask("What changed in the last 24 hours?")
        print(f"   Answer: {answer[:200]}...")

    finally:
        # Stop daemon
        daemon.stop()

    # Show recent alerts
    recent_alerts = daemon.get_recent_alerts(hours=1)
    if recent_alerts:
        print(f"\n🚨 Alerts Generated During Demo:")
        for alert in recent_alerts:
            print(f"\n   [{alert.level.value.upper()}] {alert.title}")
            print(f"   {alert.message}")
            if alert.actions_recommended:
                print(f"   Recommended: {', '.join(alert.actions_recommended)}")

    print("\n" + "="*80)
    print("✓ Watch Mode: Autonomous monitoring with real-time alerts!")
    print("="*80)


def demo_replay_mode():
    """Demonstrate Replay Mode (Forensic Analysis)."""
    print("\n" + "="*80)
    print("3. REPLAY MODE - Forensic Analysis & Backtesting")
    print("="*80)

    print("\n🔄 Replay Mode answers 'How did we get here?' questions:")
    print("   • Reconstruct model state at any historical point")
    print("   • Track forecast evolution over time")
    print("   • Backtest predictions against actuals")
    print("   • Run counterfactual 'what if' scenarios\n")

    # Create replay analyzer
    replay = ReplayAnalyzer("model_history.db")

    print("📅 Historical Analysis Examples:\n")

    # 1. Get snapshot on specific date
    print("─"*80)
    print("Example 1: Reconstruct State on Specific Date")
    print("─"*80)

    target_date = datetime(2024, 3, 15)
    snapshot = replay.get_snapshot(target_date)

    if snapshot:
        print(f"\nSystem state on {target_date.date()}:")
        print(f"   Risk Scores:")
        for entity, score in snapshot.risk_scores.items():
            print(f"     • {entity}: {score:.2%}")
        print(f"   Active forecasts: {len(snapshot.forecasts)}")
        print(f"   Events on this day: {len(snapshot.events)}")
        print(f"   Model states: {list(snapshot.model_states.keys())}")

    # 2. Track forecast evolution
    print("\n" + "─"*80)
    print("Example 2: Track How Forecast Changed Over Time")
    print("─"*80)

    evolution = replay.track_forecast_evolution(
        topic="iran",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        granularity="weekly"
    )

    if evolution:
        print(f"\nIran risk forecast evolution (Jan-Jun 2024):")
        print(f"   Showing every 30 days:\n")
        for i, snapshot in enumerate(evolution[::30]):
            date = snapshot['date']
            risk = snapshot['risk_score']
            trend = "↗" if i > 0 and risk > evolution[i-1]['risk_score'] else "↘"
            print(f"   {date.strftime('%Y-%m-%d')}: {risk:.2%} {trend}")

    # 3. Backtest forecasts
    print("\n" + "─"*80)
    print("Example 3: Backtest 30-Day Forecasts")
    print("─"*80)

    backtest_results = replay.backtest_forecasts(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 5, 1),
        horizon=30
    )

    if backtest_results:
        print(f"\nBacktest Results ({len(backtest_results)} forecasts):")

        # Aggregate metrics
        all_mae = [r.metrics.get('mae') for r in backtest_results if 'mae' in r.metrics]
        if all_mae:
            print(f"   Mean Absolute Error (MAE):")
            print(f"     • Average: {sum(all_mae)/len(all_mae):.3f}")
            print(f"     • Best: {min(all_mae):.3f}")
            print(f"     • Worst: {max(all_mae):.3f}")

        # Show a few examples
        print(f"\n   Sample forecasts:")
        for i, result in enumerate(backtest_results[:3]):
            print(f"\n   Forecast {i+1}:")
            print(f"     Made on: {result.forecast_date.date()}")
            print(f"     Target: {result.target_date.date()} (30 days ahead)")
            print(f"     Errors: {result.metrics.get('mae', 'N/A'):.3f} MAE")

    # 4. Explain forecast change
    print("\n" + "─"*80)
    print("Example 4: Explain Why Forecast Changed")
    print("─"*80)

    explanation = replay.explain_forecast_change(
        topic="iran",
        date1=datetime(2024, 2, 1),
        date2=datetime(2024, 5, 1)
    )

    print(f"\nWhy did Iran forecast change (Feb → May 2024)?")
    print(f"   Risk Change: {explanation['risk_change']:+.2f} ({explanation['risk_change_pct']:+.1f}%)")
    print(f"   Events Between: {explanation['n_events']}")
    print(f"   Model Updates: {len(explanation['model_changes'])}")
    print(f"\n   Explanation: {explanation['explanation']}")

    # 5. Counterfactual analysis
    print("\n" + "─"*80)
    print("Example 5: Counterfactual 'What If' Scenario")
    print("─"*80)

    counterfactual = replay.run_counterfactual(
        intervention_date=datetime(2024, 2, 1),
        intervention={"sanctions": 1.0, "effect_multiplier": 0.7},
        actual_date=datetime(2024, 5, 1)
    )

    print(f"\nCounterfactual: What if sanctions imposed on Feb 1, 2024?")
    print(f"   Intervention: Sanctions (effect: -30% risk)")
    print(f"   Comparing outcomes on May 1, 2024:\n")

    if 'actual_outcomes' in counterfactual:
        print(f"   {'Entity':<15} {'Actual':<12} {'Counterfactual':<15} {'Difference':<12}")
        print(f"   {'-'*54}")
        for entity in counterfactual['actual_outcomes']:
            actual = counterfactual['actual_outcomes'][entity]
            cf = counterfactual['counterfactual_outcomes'].get(entity, 0)
            diff = counterfactual['differences'].get(entity, 0)
            print(f"   {entity:<15} {actual:<12.2%} {cf:<15.2%} {diff:+.2%}")

    # 6. Performance report
    print("\n" + "─"*80)
    print("Example 6: Generate Performance Report")
    print("─"*80)

    report = replay.generate_performance_report(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30)
    )

    print(report)

    print("\n" + "="*80)
    print("✓ Replay Mode: Complete forensic analysis and backtesting!")
    print("="*80)


def demo_mode_manager():
    """Demonstrate mode management and transitions."""
    print("\n" + "="*80)
    print("4. MODE MANAGER - Switching Between Modes")
    print("="*80)

    # Create mode manager
    manager = ModeManager()

    print("\n📋 Available Operational Modes:\n")

    for mode in OperationalMode:
        if mode != OperationalMode.IDLE:
            print(manager.get_mode_description(mode))
            print()

    # Transition through modes
    print("🔄 Mode Transition Demo:\n")

    # Start in interactive mode
    manager.set_mode(
        OperationalMode.INTERACTIVE,
        settings={
            "llm_model": "mistral",
            "verbosity": "detailed",
            "output_format": "both"
        },
        reason="User initiated session"
    )

    # Switch to watch mode
    time.sleep(1)
    manager.set_mode(
        OperationalMode.WATCH,
        settings={
            "check_interval": 300,
            "alert_threshold": "medium",
            "auto_update_models": True
        },
        reason="Scheduled monitoring period"
    )

    # Switch to replay mode
    time.sleep(1)
    manager.set_mode(
        OperationalMode.REPLAY,
        settings={
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "focus": "iran_nuclear",
            "include_counterfactuals": True
        },
        reason="Performance analysis requested"
    )

    # Show status
    print()
    manager.print_status()

    # Show mode settings schema
    print("\n📝 Configuration Options per Mode:\n")
    for mode in [OperationalMode.INTERACTIVE, OperationalMode.WATCH, OperationalMode.REPLAY]:
        schema = manager.get_mode_settings_schema(mode)
        if schema:
            print(f"{mode.value.upper()} Mode Settings:")
            for setting, config in schema.items():
                print(f"  • {setting}: {config['description']}")
                if 'default' in config:
                    print(f"    Default: {config['default']}")
            print()

    print("="*80)
    print("✓ Mode Manager: Seamless transitions between operational modes!")
    print("="*80)


def demo_integration():
    """Show how all modes work together."""
    print("\n" + "="*80)
    print("5. INTEGRATION - All Modes Working Together")
    print("="*80)

    print("\n🎯 Typical Operational Workflow:\n")

    workflow_steps = [
        {
            "step": 1,
            "mode": "INTERACTIVE",
            "action": "Analyst asks: 'What is Iran nuclear risk?'",
            "system": "LLM routes to VAR + Hawkes modules",
            "output": "Narrative answer + structured risk score"
        },
        {
            "step": 2,
            "mode": "WATCH",
            "action": "System switches to monitoring mode",
            "system": "Daemon checks every 5 minutes for updates",
            "output": "Alert: 'Risk threshold exceeded'"
        },
        {
            "step": 3,
            "mode": "WATCH → INTERACTIVE",
            "action": "Alert triggers - analyst investigates",
            "system": "Chat override: 'Explain the alert'",
            "output": "LLM explains: New IAEA report changed model"
        },
        {
            "step": 4,
            "mode": "REPLAY",
            "action": "Analyst: 'How did forecast change over Q1?'",
            "system": "Replay mode reconstructs historical states",
            "output": "Evolution chart + backtest accuracy"
        },
        {
            "step": 5,
            "mode": "INTERACTIVE",
            "action": "Analyst: 'What if sanctions imposed earlier?'",
            "system": "Run counterfactual with do-calculus",
            "output": "Estimated effect: -25% risk reduction"
        },
        {
            "step": 6,
            "mode": "WATCH",
            "action": "Resume autonomous monitoring",
            "system": "Continue background checks",
            "output": "Log analysis results for next replay"
        }
    ]

    for item in workflow_steps:
        print(f"\n{'─'*80}")
        print(f"Step {item['step']}: {item['mode']}")
        print(f"{'─'*80}")
        print(f"   User Action:  {item['action']}")
        print(f"   System:       {item['system']}")
        print(f"   Output:       {item['output']}")

    print("\n" + "="*80)
    print("✓ Seamless integration across all operational modes!")
    print("="*80)


def main():
    """Run all operational mode demonstrations."""
    print("="*80)
    print("GeoBotv1 - OPERATIONAL MODES DEMONSTRATION")
    print("="*80)
    print("\nTransforming mathematical framework → operational intelligence system")
    print("\nThree modes for different use cases:")
    print("  1. INTERACTIVE: Natural language Q&A for analysts")
    print("  2. WATCH: Autonomous monitoring with alerts")
    print("  3. REPLAY: Forensic analysis and backtesting")

    # Run demonstrations
    demo_interactive_mode()
    demo_watch_mode()
    demo_replay_mode()
    demo_mode_manager()
    demo_integration()

    print("\n" + "="*80)
    print("GeoBotv1 Operational Modes - Complete!")
    print("="*80)

    print("\n🎉 Key Capabilities Demonstrated:")
    print("\n1️⃣  INTERACTIVE MODE:")
    print("   • Natural language questions → mathematical analysis")
    print("   • LLM interprets intent and routes to modules")
    print("   • Returns: Narrative answer + structured analysis block")
    print("   • Handles: forecasts, risks, causality, counterfactuals")

    print("\n2️⃣  WATCH MODE:")
    print("   • Autonomous background monitoring")
    print("   • Real-time alert generation (regime shifts, thresholds, anomalies)")
    print("   • Automatic model updates with new data")
    print("   • Chat override for manual queries")
    print("   • Notification channels: console, email, webhook")

    print("\n3️⃣  REPLAY MODE:")
    print("   • Historical state reconstruction")
    print("   • Forecast evolution tracking")
    print("   • Backtest accuracy analysis")
    print("   • Counterfactual what-if scenarios")
    print("   • Performance reports and metrics")

    print("\n🔧 Technical Architecture:")
    print("   • Mode Manager: Handles transitions and state persistence")
    print("   • Analyst Agent: LLM-powered natural language interface")
    print("   • Watch Daemon: Background monitoring with threading")
    print("   • Replay Analyzer: Historical analysis and backtesting")

    print("\n💡 Production Deployment:")
    print("   • Interactive: Analyst workstation UI")
    print("   • Watch: Server daemon (systemd, supervisor, docker)")
    print("   • Replay: Scheduled reports + on-demand investigation")

    print("\n🚀 GeoBotv1 is now a complete operational intelligence system!")
    print("   Mathematics + Natural Language = Production-Ready AI")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
