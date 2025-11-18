#!/usr/bin/env python3
"""
GeoBot Live - Interactive Real-Time Geopolitical Analysis

Ask a question and watch real-time ticker with escalation/regime change percentages.
Updates continuously with latest intelligence from RSS feeds.
"""

import time
import sys
import os
from datetime import datetime
from typing import Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from geobot.discord_bot.forecaster import ConflictForecaster
    from geobot.data_ingestion.rss_scraper import RSSFeedScraper
    from geobot.analysis.engine import AnalyticalEngine
    HAS_ALL = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    HAS_ALL = False


class LiveTicker:
    """
    Interactive live ticker with real-time probability updates.
    """

    def __init__(self):
        """Initialize live ticker."""
        if not HAS_ALL:
            raise ImportError("Required modules not available")

        self.forecaster = ConflictForecaster()
        self.scraper = RSSFeedScraper()
        self.engine = AnalyticalEngine()

        self.running = False

    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def format_percentage(self, value: float, width: int = 50) -> str:
        """
        Format probability as visual percentage bar.

        Parameters
        ----------
        value : float
            Probability (0-1)
        width : int
            Width of bar in characters

        Returns
        -------
        str
            Visual percentage bar
        """
        percentage = value * 100
        filled = int(value * width)
        bar = '█' * filled + '░' * (width - filled)

        # Color code based on risk
        if percentage >= 60:
            color = '\033[91m'  # Red
        elif percentage >= 40:
            color = '\033[93m'  # Yellow
        elif percentage >= 20:
            color = '\033[94m'  # Blue
        else:
            color = '\033[92m'  # Green

        reset = '\033[0m'

        return f"{color}{bar}{reset} {percentage:5.1f}%"

    def display_header(self, question: str, update_count: int):
        """Display header with question and update count."""
        print("=" * 100)
        print("🌍 GEOBOT LIVE - REAL-TIME GEOPOLITICAL INTELLIGENCE TICKER")
        print("=" * 100)
        print()
        print(f"📊 Scenario: {question}")
        print(f"🔄 Update #{update_count} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    def display_forecast(self, forecast):
        """Display forecast with visual percentages."""
        print("━" * 100)
        print(f"⚔️  CONFLICT: {forecast.conflict_name}")
        print("━" * 100)
        print()

        # Escalation probability
        print("📈 ESCALATION PROBABILITY")
        print(self.format_percentage(forecast.escalation_probability))
        print()

        # Regime change probability
        print("🏛️  REGIME CHANGE PROBABILITY")
        print(self.format_percentage(forecast.regime_change_probability))
        print()

        # Risk level
        risk_colors = {
            'critical': '\033[91m',  # Red
            'high': '\033[93m',      # Yellow
            'medium': '\033[94m',    # Blue
            'low': '\033[92m'        # Green
        }
        risk_color = risk_colors.get(forecast.risk_level, '')
        reset = '\033[0m'

        print(f"⚠️  OVERALL RISK: {risk_color}{forecast.risk_level.upper()}{reset}")
        print(f"🎯 CONFIDENCE: {forecast.confidence:.0%}")
        print(f"⏰ TIMEFRAME: {forecast.timeframe}")
        print()

        # Key factors
        print("📋 KEY FACTORS:")
        for factor in forecast.key_factors:
            print(f"   • {factor}")
        print()

    def display_analysis(self, analysis: str):
        """Display GeoBot 2.0 analysis."""
        print("━" * 100)
        print("💡 GEOBOT 2.0 ANALYTICAL ASSESSMENT")
        print("━" * 100)
        print()

        # Truncate if too long
        if len(analysis) > 800:
            analysis = analysis[:800] + "...\n[Truncated - full analysis available in logs]"

        print(analysis)
        print()

    def display_news_ticker(self, articles):
        """Display latest news ticker."""
        if not articles:
            return

        print("━" * 100)
        print("📰 LATEST INTELLIGENCE")
        print("━" * 100)
        print()

        for i, article in enumerate(articles[:3], 1):
            countries = article.extract_countries()
            countries_str = ", ".join(countries[:3]) if countries else "Global"

            print(f"{i}. {article.title[:90]}")
            print(f"   🌍 {countries_str} | 📰 {article.source}")
            print()

    def run(self, question: str, update_interval: int = 30):
        """
        Run live ticker for a question.

        Parameters
        ----------
        question : str
            Question or conflict to analyze
        update_interval : int
            Seconds between updates (default: 30)
        """
        self.running = True
        update_count = 0

        print("\n" + "=" * 100)
        print("🚀 Starting GeoBot Live...")
        print("=" * 100)
        print()
        print(f"Question: {question}")
        print(f"Update interval: {update_interval} seconds")
        print()
        print("Press Ctrl+C to stop")
        print()
        input("Press ENTER to start...")

        try:
            while self.running:
                update_count += 1

                # Clear screen for fresh display
                self.clear_screen()

                # Display header
                self.display_header(question, update_count)

                try:
                    # Scrape latest news
                    articles = self.scraper.scrape_all(geopolitical_only=True)

                    # Filter articles relevant to question
                    question_lower = question.lower()
                    relevant_articles = [
                        a for a in articles
                        if any(word in (a.title + a.summary).lower()
                               for word in question_lower.split())
                    ]

                    # Get forecast
                    context_text = " ".join([
                        a.title + " " + a.summary
                        for a in relevant_articles[:10]
                    ]) if relevant_articles else None

                    forecast = self.forecaster.forecast_conflict(
                        question,
                        context_text=context_text,
                        recent_articles=[a.title for a in relevant_articles[:5]]
                    )

                    # Display forecast
                    self.display_forecast(forecast)

                    # Get GeoBot 2.0 analysis
                    context = {
                        'question': question,
                        'recent_news': [
                            {'title': a.title, 'source': a.source}
                            for a in relevant_articles[:5]
                        ]
                    }

                    analysis = self.engine.analyze(question, context)
                    self.display_analysis(analysis)

                    # Display news ticker
                    self.display_news_ticker(relevant_articles if relevant_articles else articles[:3])

                except Exception as e:
                    print(f"⚠️  Error during update: {e}")
                    print()

                # Footer
                print("━" * 100)
                print(f"⏱️  Next update in {update_interval} seconds... (Press Ctrl+C to stop)")
                print("=" * 100)

                # Wait for next update
                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\n\n")
            print("=" * 100)
            print("🛑 GeoBot Live stopped by user")
            print("=" * 100)
            self.running = False


def main():
    """Main entry point for GeoBot Live."""
    print()
    print("=" * 100)
    print("🌍 GEOBOT LIVE")
    print("=" * 100)
    print()
    print("Real-time geopolitical intelligence with continuous probability updates")
    print()

    # Get question from user
    if len(sys.argv) > 1:
        # Question provided as command-line argument
        question = " ".join(sys.argv[1:])
    else:
        # Interactive prompt
        print("Enter your geopolitical question or conflict to analyze:")
        print()
        print("Examples:")
        print("  - Taiwan strait escalation")
        print("  - Ukraine conflict dynamics")
        print("  - Middle East tensions")
        print("  - China military readiness")
        print()
        question = input("Your question: ").strip()

    if not question:
        print("Error: No question provided")
        return 1

    # Get update interval
    try:
        interval_input = input("\nUpdate interval in seconds (default: 30): ").strip()
        interval = int(interval_input) if interval_input else 30
    except ValueError:
        interval = 30

    # Create and run ticker
    try:
        ticker = LiveTicker()
        ticker.run(question, update_interval=interval)
    except ImportError as e:
        print(f"\nError: Required modules not available: {e}")
        print("\nInstall dependencies:")
        print("  pip install -e .")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
