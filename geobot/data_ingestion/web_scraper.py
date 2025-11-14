"""
Web Scraping and Article Extraction Module

Comprehensive web scraping capabilities for:
- News articles
- Analysis pieces
- Intelligence reports
- Research papers
- Real-time news feeds

Supports multiple extraction methods for robustness.
"""

import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse
import re


class WebScraper:
    """
    General-purpose web scraper for geopolitical content.

    Handles various website structures and content types.
    """

    def __init__(self, user_agent: Optional[str] = None):
        """
        Initialize web scraper.

        Parameters
        ----------
        user_agent : str, optional
            Custom user agent string
        """
        self.user_agent = user_agent or 'GeoBotv1/1.0 (Geopolitical Analysis)'
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

    def fetch_url(self, url: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Fetch content from URL.

        Parameters
        ----------
        url : str
            URL to fetch
        timeout : int
            Request timeout in seconds

        Returns
        -------
        dict
            Response data including content, status, headers
        """
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            return {
                'url': url,
                'status_code': response.status_code,
                'content': response.text,
                'headers': dict(response.headers),
                'encoding': response.encoding,
                'timestamp': datetime.now().isoformat()
            }

        except requests.exceptions.RequestException as e:
            return {
                'url': url,
                'error': str(e),
                'status_code': None,
                'content': None,
                'timestamp': datetime.now().isoformat()
            }

    def parse_html(self, html_content: str) -> Dict[str, Any]:
        """
        Parse HTML content.

        Parameters
        ----------
        html_content : str
            HTML content

        Returns
        -------
        dict
            Parsed content
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract basic elements
            parsed = {
                'title': soup.title.string if soup.title else '',
                'text': soup.get_text(),
                'links': [a.get('href') for a in soup.find_all('a', href=True)],
                'images': [img.get('src') for img in soup.find_all('img', src=True)],
                'meta': {}
            }

            # Extract meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    parsed['meta'][name] = content

            return parsed

        except ImportError:
            print("Warning: BeautifulSoup not available. Install with: pip install beautifulsoup4")
            return {
                'title': '',
                'text': self._simple_html_strip(html_content),
                'links': [],
                'images': [],
                'meta': {}
            }

    def _simple_html_strip(self, html: str) -> str:
        """Simple HTML tag removal."""
        return re.sub(r'<[^>]+>', '', html)

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape and parse URL.

        Parameters
        ----------
        url : str
            URL to scrape

        Returns
        -------
        dict
            Scraped and parsed content
        """
        # Fetch
        response = self.fetch_url(url)

        if response.get('error'):
            return response

        # Parse
        parsed = self.parse_html(response['content'])

        # Combine
        return {
            'url': url,
            'domain': urlparse(url).netloc,
            'title': parsed['title'],
            'text': parsed['text'],
            'meta': parsed['meta'],
            'links': parsed['links'],
            'images': parsed['images'],
            'timestamp': response['timestamp'],
            'status_code': response['status_code']
        }


class ArticleExtractor:
    """
    Extract clean article content from web pages.

    Uses multiple extraction methods for robustness.
    """

    def __init__(self, method: str = 'auto'):
        """
        Initialize article extractor.

        Parameters
        ----------
        method : str
            Extraction method ('newspaper', 'trafilatura', 'auto')
        """
        self.method = method
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check which libraries are available."""
        self.has_newspaper = False
        self.has_trafilatura = False

        try:
            import newspaper
            self.has_newspaper = True
        except ImportError:
            pass

        try:
            import trafilatura
            self.has_trafilatura = True
        except ImportError:
            pass

    def extract_article(self, url: str) -> Dict[str, Any]:
        """
        Extract article from URL.

        Parameters
        ----------
        url : str
            Article URL

        Returns
        -------
        dict
            Extracted article data
        """
        method = self.method
        if method == 'auto':
            if self.has_newspaper:
                method = 'newspaper'
            elif self.has_trafilatura:
                method = 'trafilatura'
            else:
                method = 'basic'

        if method == 'newspaper':
            return self._extract_with_newspaper(url)
        elif method == 'trafilatura':
            return self._extract_with_trafilatura(url)
        else:
            return self._extract_basic(url)

    def _extract_with_newspaper(self, url: str) -> Dict[str, Any]:
        """Extract article using newspaper3k."""
        try:
            from newspaper import Article

            article = Article(url)
            article.download()
            article.parse()

            try:
                article.nlp()
                keywords = article.keywords
                summary = article.summary
            except:
                keywords = []
                summary = ''

            return {
                'url': url,
                'title': article.title,
                'text': article.text,
                'authors': article.authors,
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'keywords': keywords,
                'summary': summary,
                'top_image': article.top_image,
                'images': list(article.images),
                'method': 'newspaper',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'method': 'newspaper'
            }

    def _extract_with_trafilatura(self, url: str) -> Dict[str, Any]:
        """Extract article using trafilatura."""
        try:
            import trafilatura

            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded)
            metadata = trafilatura.extract_metadata(downloaded)

            return {
                'url': url,
                'title': metadata.title if metadata else '',
                'text': text or '',
                'authors': [metadata.author] if metadata and metadata.author else [],
                'publish_date': metadata.date if metadata else None,
                'description': metadata.description if metadata else '',
                'method': 'trafilatura',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'method': 'trafilatura'
            }

    def _extract_basic(self, url: str) -> Dict[str, Any]:
        """Basic extraction using BeautifulSoup."""
        scraper = WebScraper()
        content = scraper.scrape_url(url)

        return {
            'url': url,
            'title': content.get('title', ''),
            'text': content.get('text', ''),
            'meta': content.get('meta', {}),
            'method': 'basic',
            'timestamp': datetime.now().isoformat()
        }

    def batch_extract(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Extract multiple articles.

        Parameters
        ----------
        urls : list
            List of article URLs

        Returns
        -------
        list
            List of extracted articles
        """
        articles = []
        for url in urls:
            try:
                article = self.extract_article(url)
                articles.append(article)
            except Exception as e:
                print(f"Error extracting {url}: {e}")

        return articles


class NewsAggregator:
    """
    Aggregate news from multiple sources.

    Monitors multiple news sources for geopolitical content.
    """

    def __init__(self):
        """Initialize news aggregator."""
        self.extractor = ArticleExtractor()
        self.sources = []

    def add_source(self, name: str, url: str, source_type: str = 'rss') -> None:
        """
        Add news source.

        Parameters
        ----------
        name : str
            Source name
        url : str
            Source URL
        source_type : str
            Source type ('rss', 'website')
        """
        self.sources.append({
            'name': name,
            'url': url,
            'type': source_type
        })

    def fetch_news(self, keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Fetch news from all sources.

        Parameters
        ----------
        keywords : list, optional
            Filter by keywords

        Returns
        -------
        list
            List of news articles
        """
        articles = []

        for source in self.sources:
            try:
                if source['type'] == 'rss':
                    source_articles = self._fetch_rss(source['url'])
                else:
                    source_articles = self._fetch_website(source['url'])

                # Add source info
                for article in source_articles:
                    article['source'] = source['name']

                    # Filter by keywords if provided
                    if keywords:
                        text = article.get('text', '').lower()
                        if any(kw.lower() in text for kw in keywords):
                            articles.append(article)
                    else:
                        articles.append(article)

            except Exception as e:
                print(f"Error fetching from {source['name']}: {e}")

        return articles

    def _fetch_rss(self, rss_url: str) -> List[Dict[str, Any]]:
        """Fetch articles from RSS feed."""
        try:
            import feedparser

            feed = feedparser.parse(rss_url)
            articles = []

            for entry in feed.entries:
                article = {
                    'title': entry.get('title', ''),
                    'url': entry.get('link', ''),
                    'summary': entry.get('summary', ''),
                    'publish_date': entry.get('published', ''),
                    'authors': [author.get('name') for author in entry.get('authors', [])],
                }

                # Fetch full article if URL available
                if article['url']:
                    try:
                        full_article = self.extractor.extract_article(article['url'])
                        article['text'] = full_article.get('text', article['summary'])
                    except:
                        article['text'] = article['summary']

                articles.append(article)

            return articles

        except ImportError:
            print("Warning: feedparser not available. Install with: pip install feedparser")
            return []

    def _fetch_website(self, website_url: str) -> List[Dict[str, Any]]:
        """Fetch articles from website."""
        # Basic implementation - could be enhanced
        article = self.extractor.extract_article(website_url)
        return [article] if not article.get('error') else []

    def monitor_sources(
        self,
        keywords: List[str],
        callback: Optional[Any] = None,
        interval: int = 3600
    ) -> None:
        """
        Monitor sources for new articles.

        Parameters
        ----------
        keywords : list
            Keywords to monitor
        callback : callable, optional
            Function to call when new articles found
        interval : int
            Check interval in seconds
        """
        import time

        seen_urls = set()

        while True:
            articles = self.fetch_news(keywords)

            # Filter new articles
            new_articles = [a for a in articles if a['url'] not in seen_urls]

            if new_articles and callback:
                callback(new_articles)

            # Update seen URLs
            seen_urls.update(a['url'] for a in new_articles)

            # Wait
            time.sleep(interval)

    def get_trending_topics(self, articles: List[Dict[str, Any]], n_topics: int = 10) -> List[Tuple[str, int]]:
        """
        Extract trending topics from articles.

        Parameters
        ----------
        articles : list
            List of articles
        n_topics : int
            Number of top topics to return

        Returns
        -------
        list
            List of (topic, count) tuples
        """
        from collections import Counter

        words = []
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                    'to', 'for', 'of', 'with', 'by', 'from'}

        for article in articles:
            text = article.get('text', '') + ' ' + article.get('title', '')
            text_words = text.lower().split()
            words.extend([w for w in text_words if w not in stopwords and len(w) > 3])

        word_counts = Counter(words)
        return word_counts.most_common(n_topics)
