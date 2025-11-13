"""
Data ingestion modules for GeoBotv1

Support for:
- PDF document reading and processing
- Web scraping and article extraction
- News feed ingestion
"""

from .pdf_reader import PDFReader, PDFProcessor
from .web_scraper import WebScraper, ArticleExtractor, NewsAggregator

__all__ = [
    "PDFReader",
    "PDFProcessor",
    "WebScraper",
    "ArticleExtractor",
    "NewsAggregator",
]
