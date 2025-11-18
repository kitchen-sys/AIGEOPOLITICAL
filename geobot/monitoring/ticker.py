"""
Real-Time Geopolitical Intelligence Ticker

Continuously monitors RSS feeds and generates AI-powered insights
using GeoBot 2.0 analytical framework.

Updates every 30 minutes with new geopolitical developments.
"""

import time
import json
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import signal
import sys

try:
    from ..data_ingestion.rss_scraper import RSSFeedScraper, NewsArticle
    HAS_RSS = True
except ImportError:
    HAS_RSS = False
    RSSFeedScraper = None
    NewsArticle = None

try:
    from ..analysis.engine import AnalyticalEngine
    from ..analysis.formatter import AnalysisFormatter
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False
    AnalyticalEngine = None


@dataclass
class GeopoliticalInsight:
    """
    AI-generated insight from news analysis.

    Attributes
    ----------
    timestamp : str
        When insight was generated
    headline : str
        Key headline summarizing the insight
    articles_analyzed : int
        Number of articles analyzed
    key_developments : List[str]
        Main geopolitical developments
    risk_assessment : str
        Risk level assessment
    countries_involved : List[str]
        Countries mentioned
    analysis : str
        Full GeoBot 2.0 analysis
    """
    timestamp: str
    headline: str
    articles_analyzed: int
    key_developments: List[str]
    risk_assessment: str
    countries_involved: List[str]
    analysis: Optional[str] = None


class GeopoliticalTicker:
    """
    Real-time geopolitical intelligence ticker.

    Monitors RSS feeds every 30 minutes and generates AI insights.
    """

    def __init__(self,
                 update_interval_minutes: int = 30,
                 output_dir: Optional[Path] = None,
                 use_ai_analysis: bool = True):
        """
        Initialize ticker.

        Parameters
        ----------
        update_interval_minutes : int
            Minutes between updates (default: 30)
        output_dir : Optional[Path]
            Directory to save insights (default: ./ticker_output)
        use_ai_analysis : bool
            Whether to generate AI analysis (default: True)
        """
        if not HAS_RSS:
            raise ImportError(
                "RSS scraper not available. Install with: pip install feedparser requests"
            )

        self.update_interval = update_interval_minutes * 60  # Convert to seconds
        self.output_dir = output_dir or Path('./ticker_output')
        self.output_dir.mkdir(exist_ok=True)

        self.use_ai_analysis = use_ai_analysis and HAS_ANALYSIS
        if use_ai_analysis and not HAS_ANALYSIS:
            print("Warning: GeoBot 2.0 analysis not available. Running without AI insights.")

        self.scraper = RSSFeedScraper()
        self.engine = AnalyticalEngine() if self.use_ai_analysis else None

        self.seen_articles: Set[str] = set()  # Track seen article links
        self.running = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n\nShutting down ticker gracefully...")
        self.running = False
        sys.exit(0)

    def _analyze_articles(self, articles: List[NewsArticle]) -> GeopoliticalInsight:
        """
        Generate insight from articles using GeoBot 2.0.

        Parameters
        ----------
        articles : List[NewsArticle]
            Articles to analyze

        Returns
        -------
        GeopoliticalInsight
            Generated insight
        """
        # Extract key information
        all_countries = []
        for article in articles:
            all_countries.extend(article.extract_countries())

        country_counts = {}
        for country in all_countries:
            country_counts[country] = country_counts.get(country, 0) + 1

        top_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_countries = [c[0] for c in top_countries]

        # Extract key developments from article titles
        key_developments = [a.title for a in articles[:5]]

        # Risk assessment based on keywords
        risk_keywords = {
            'critical': ['war', 'invasion', 'nuclear', 'attack', 'crisis'],
            'high': ['conflict', 'military', 'sanctions', 'escalation', 'threat'],
            'medium': ['tension', 'diplomatic', 'security', 'defense'],
            'low': ['talks', 'agreement', 'cooperation', 'dialogue']
        }

        text = ' '.join([a.title + ' ' + a.summary for a in articles]).lower()
        risk_level = 'low'

        for level in ['critical', 'high', 'medium']:
            if any(keyword in text for keyword in risk_keywords[level]):
                risk_level = level
                break

        # Generate AI analysis if available
        analysis = None
        if self.use_ai_analysis and self.engine:
            try:
                # Build context from articles
                context = {
                    'articles': [
                        {'title': a.title, 'summary': a.summary, 'source': a.source}
                        for a in articles[:10]
                    ],
                    'countries': top_countries,
                    'timeframe': 'current',
                    'risk_level': risk_level
                }

                query = f"Analyze recent geopolitical developments involving {', '.join(top_countries[:3])}"
                analysis = self.engine.analyze(query, context)

            except Exception as e:
                print(f"Warning: AI analysis failed: {e}")
                analysis = None

        # Create headline
        if top_countries:
            headline = f"Latest developments: {', '.join(top_countries[:3])} - Risk: {risk_level.upper()}"
        else:
            headline = f"Geopolitical monitoring update - Risk: {risk_level.upper()}"

        return GeopoliticalInsight(
            timestamp=datetime.now().isoformat(),
            headline=headline,
            articles_analyzed=len(articles),
            key_developments=key_developments,
            risk_assessment=risk_level,
            countries_involved=top_countries,
            analysis=analysis
        )

    def _filter_new_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """
        Filter to only new articles not seen before.

        Parameters
        ----------
        articles : List[NewsArticle]
            All articles

        Returns
        -------
        List[NewsArticle]
            Only new articles
        """
        new_articles = []
        for article in articles:
            if article.link not in self.seen_articles:
                new_articles.append(article)
                self.seen_articles.add(article.link)

        return new_articles

    def _save_insight(self, insight: GeopoliticalInsight):
        """
        Save insight to JSON file.

        Parameters
        ----------
        insight : GeopoliticalInsight
            Insight to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"insight_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(asdict(insight), f, indent=2)

    def _display_insight(self, insight: GeopoliticalInsight, new_count: int):
        """
        Display insight to console.

        Parameters
        ----------
        insight : GeopoliticalInsight
            Insight to display
        new_count : int
            Number of new articles
        """
        print("\n" + "=" * 80)
        print(f"GEOPOLITICAL INTELLIGENCE UPDATE")
        print(f"Time: {insight.timestamp}")
        print("=" * 80)
        print()
        print(f"📰 {insight.headline}")
        print()
        print(f"New Articles: {new_count}")
        print(f"Total Analyzed: {insight.articles_analyzed}")
        print(f"Risk Level: {insight.risk_assessment.upper()}")
        print()

        if insight.countries_involved:
            print(f"Countries Involved: {', '.join(insight.countries_involved)}")
            print()

        if insight.key_developments:
            print("Key Developments:")
            for i, dev in enumerate(insight.key_developments[:5], 1):
                print(f"  {i}. {dev}")
            print()

        if insight.analysis:
            print("GeoBot 2.0 Analysis:")
            print("-" * 80)
            print(insight.analysis)
            print("-" * 80)

        print()
        print(f"Next update in {self.update_interval // 60} minutes...")
        print("=" * 80)

    def run_once(self) -> Optional[GeopoliticalInsight]:
        """
        Run a single update cycle.

        Returns
        -------
        Optional[GeopoliticalInsight]
            Generated insight, or None if no new articles
        """
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning RSS feeds...")

        # Scrape all feeds
        all_articles = self.scraper.scrape_all(geopolitical_only=True)

        # Filter to new articles
        new_articles = self._filter_new_articles(all_articles)

        if not new_articles:
            print(f"No new articles found.")
            return None

        print(f"Found {len(new_articles)} new articles")

        # Generate insight
        insight = self._analyze_articles(new_articles)

        # Save and display
        self._save_insight(insight)
        self._display_insight(insight, len(new_articles))

        return insight

    def run(self):
        """
        Run ticker continuously.

        Updates every 30 minutes (or configured interval).
        Press Ctrl+C to stop.
        """
        print("=" * 80)
        print("GEOBOT REAL-TIME INTELLIGENCE TICKER")
        print("=" * 80)
        print()
        print(f"Update interval: {self.update_interval // 60} minutes")
        print(f"AI Analysis: {'Enabled' if self.use_ai_analysis else 'Disabled'}")
        print(f"Output directory: {self.output_dir}")
        print()
        print("Press Ctrl+C to stop")
        print("=" * 80)

        self.running = True

        # Run initial update immediately
        self.run_once()

        # Continue running at intervals
        while self.running:
            try:
                time.sleep(self.update_interval)
                if self.running:  # Check again after sleep
                    self.run_once()
            except KeyboardInterrupt:
                print("\n\nStopping ticker...")
                break
            except Exception as e:
                print(f"\nError during update: {e}")
                print("Continuing...")

        print("Ticker stopped.")

    def get_latest_insight(self) -> Optional[GeopoliticalInsight]:
        """
        Get the most recent saved insight.

        Returns
        -------
        Optional[GeopoliticalInsight]
            Latest insight, or None if no insights exist
        """
        insight_files = sorted(self.output_dir.glob("insight_*.json"))

        if not insight_files:
            return None

        latest_file = insight_files[-1]
        with open(latest_file, 'r') as f:
            data = json.load(f)
            return GeopoliticalInsight(**data)


def start_ticker(interval_minutes: int = 30,
                 output_dir: Optional[str] = None,
                 use_ai: bool = True):
    """
    Start real-time geopolitical ticker.

    Parameters
    ----------
    interval_minutes : int
        Minutes between updates (default: 30)
    output_dir : Optional[str]
        Output directory for insights
    use_ai : bool
        Enable GeoBot 2.0 AI analysis
    """
    output_path = Path(output_dir) if output_dir else None
    ticker = GeopoliticalTicker(
        update_interval_minutes=interval_minutes,
        output_dir=output_path,
        use_ai_analysis=use_ai
    )
    ticker.run()


if __name__ == '__main__':
    # Test ticker with single update
    ticker = GeopoliticalTicker(update_interval_minutes=30)
    ticker.run_once()
