"""
RSS Feed Scraper for Geopolitical News

Scrapes Reuters and AP News RSS feeds for real-time geopolitical intelligence.
These sources allow RSS feed access for news aggregation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    feedparser = None

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None


@dataclass
class NewsArticle:
    """
    Represents a news article from RSS feed.

    Attributes
    ----------
    title : str
        Article title
    link : str
        URL to full article
    summary : str
        Article summary/description
    published : str
        Publication timestamp
    source : str
        News source (Reuters, AP News, etc.)
    tags : List[str]
        Topic tags/categories
    """
    title: str
    link: str
    summary: str
    published: str
    source: str
    tags: List[str]

    def is_geopolitical(self) -> bool:
        """
        Check if article is geopolitically relevant.

        Returns
        -------
        bool
            True if article contains geopolitical keywords
        """
        geopolitical_keywords = [
            'military', 'war', 'conflict', 'sanctions', 'diplomatic',
            'tension', 'crisis', 'security', 'defense', 'nuclear',
            'china', 'russia', 'iran', 'taiwan', 'ukraine', 'nato',
            'pentagon', 'armed forces', 'troops', 'missiles', 'invasion',
            'escalation', 'ceasefire', 'treaty', 'alliance', 'threat'
        ]

        text = (self.title + ' ' + self.summary).lower()
        return any(keyword in text for keyword in geopolitical_keywords)

    def extract_countries(self) -> List[str]:
        """
        Extract country mentions from article.

        Returns
        -------
        List[str]
            List of countries mentioned
        """
        countries = [
            'China', 'Russia', 'United States', 'Iran', 'North Korea',
            'Taiwan', 'Ukraine', 'Israel', 'Palestine', 'Syria',
            'Afghanistan', 'Iraq', 'Saudi Arabia', 'Turkey', 'India',
            'Pakistan', 'Japan', 'South Korea', 'NATO', 'EU'
        ]

        text = self.title + ' ' + self.summary
        mentioned = []

        for country in countries:
            if re.search(r'\b' + country + r'\b', text, re.IGNORECASE):
                mentioned.append(country)

        return mentioned


class RSSFeedScraper:
    """
    Scrapes RSS feeds from major news sources.

    Supports:
    - Reuters (multiple topic feeds)
    - AP News (top news and world news)
    """

    # Reuters RSS Feeds
    REUTERS_FEEDS = {
        'world': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
        'politics': 'https://www.reuters.com/politics',
        'world_news': 'https://www.reuters.com/world',
    }

    # AP News RSS Feeds
    AP_FEEDS = {
        'top_news': 'https://feeds.apnews.com/rss/topnews',
        'world_news': 'https://feeds.apnews.com/rss/world',
        'us_news': 'https://feeds.apnews.com/rss/us-news',
        'politics': 'https://feeds.apnews.com/rss/politics',
    }

    def __init__(self):
        """Initialize RSS scraper."""
        if not HAS_FEEDPARSER:
            raise ImportError(
                "feedparser is required for RSS scraping. "
                "Install with: pip install feedparser"
            )
        if not HAS_REQUESTS:
            raise ImportError(
                "requests is required for RSS scraping. "
                "Install with: pip install requests"
            )

    def scrape_feed(self, url: str, source: str) -> List[NewsArticle]:
        """
        Scrape a single RSS feed.

        Parameters
        ----------
        url : str
            RSS feed URL
        source : str
            News source name

        Returns
        -------
        List[NewsArticle]
            List of articles from feed
        """
        try:
            feed = feedparser.parse(url)
            articles = []

            for entry in feed.entries:
                article = NewsArticle(
                    title=entry.get('title', ''),
                    link=entry.get('link', ''),
                    summary=entry.get('summary', entry.get('description', '')),
                    published=entry.get('published', entry.get('updated', '')),
                    source=source,
                    tags=entry.get('tags', [])
                )
                articles.append(article)

            return articles

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []

    def scrape_reuters(self, categories: Optional[List[str]] = None) -> List[NewsArticle]:
        """
        Scrape Reuters RSS feeds.

        Parameters
        ----------
        categories : Optional[List[str]]
            List of categories to scrape (default: all)

        Returns
        -------
        List[NewsArticle]
            All articles from selected Reuters feeds
        """
        if categories is None:
            categories = list(self.REUTERS_FEEDS.keys())

        all_articles = []
        for category in categories:
            if category in self.REUTERS_FEEDS:
                url = self.REUTERS_FEEDS[category]
                articles = self.scrape_feed(url, f"Reuters - {category}")
                all_articles.extend(articles)

        return all_articles

    def scrape_ap_news(self, categories: Optional[List[str]] = None) -> List[NewsArticle]:
        """
        Scrape AP News RSS feeds.

        Parameters
        ----------
        categories : Optional[List[str]]
            List of categories to scrape (default: all)

        Returns
        -------
        List[NewsArticle]
            All articles from selected AP News feeds
        """
        if categories is None:
            categories = list(self.AP_FEEDS.keys())

        all_articles = []
        for category in categories:
            if category in self.AP_FEEDS:
                url = self.AP_FEEDS[category]
                articles = self.scrape_feed(url, f"AP News - {category}")
                all_articles.extend(articles)

        return all_articles

    def scrape_all(self, geopolitical_only: bool = True) -> List[NewsArticle]:
        """
        Scrape all configured RSS feeds.

        Parameters
        ----------
        geopolitical_only : bool
            If True, filter to geopolitically relevant articles only

        Returns
        -------
        List[NewsArticle]
            All articles from all feeds
        """
        all_articles = []

        # Scrape Reuters
        all_articles.extend(self.scrape_reuters())

        # Scrape AP News
        all_articles.extend(self.scrape_ap_news())

        # Filter if needed
        if geopolitical_only:
            all_articles = [a for a in all_articles if a.is_geopolitical()]

        return all_articles

    def get_summary(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """
        Generate summary statistics for scraped articles.

        Parameters
        ----------
        articles : List[NewsArticle]
            Articles to summarize

        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        if not articles:
            return {
                'total': 0,
                'by_source': {},
                'countries_mentioned': {},
                'latest': None
            }

        # Count by source
        by_source = {}
        for article in articles:
            source = article.source
            by_source[source] = by_source.get(source, 0) + 1

        # Count country mentions
        countries = {}
        for article in articles:
            for country in article.extract_countries():
                countries[country] = countries.get(country, 0) + 1

        # Sort countries by mention count
        top_countries = sorted(countries.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'total': len(articles),
            'by_source': by_source,
            'countries_mentioned': dict(top_countries),
            'latest': articles[0].published if articles else None
        }


def quick_scan() -> List[NewsArticle]:
    """
    Quick scan of all RSS feeds for geopolitical news.

    Returns
    -------
    List[NewsArticle]
        Geopolitically relevant articles
    """
    scraper = RSSFeedScraper()
    return scraper.scrape_all(geopolitical_only=True)


if __name__ == '__main__':
    # Test scraper
    print("Testing RSS Feed Scraper...")
    print("=" * 80)

    scraper = RSSFeedScraper()
    articles = scraper.scrape_all(geopolitical_only=True)

    print(f"\nFound {len(articles)} geopolitically relevant articles\n")

    summary = scraper.get_summary(articles)
    print("Summary:")
    print(f"  Total articles: {summary['total']}")
    print(f"  By source: {summary['by_source']}")
    print(f"  Top countries mentioned: {summary['countries_mentioned']}")
    print()

    print("Sample articles:")
    for i, article in enumerate(articles[:5], 1):
        print(f"\n{i}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   Countries: {', '.join(article.extract_countries())}")
        print(f"   Link: {article.link}")
