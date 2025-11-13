"""
Data ingestion modules for GeoBotv1

Support for:
- PDF document reading and processing
- Web scraping and article extraction
- News feed ingestion
- Structured event extraction
- Event database and temporal normalization
"""

from .pdf_reader import PDFReader, PDFProcessor
from .web_scraper import WebScraper, ArticleExtractor, NewsAggregator
from .event_extraction import (
    EventExtractor,
    GeopoliticalEvent,
    EventType,
    TemporalNormalizer,
    CausalFeatureExtractor
)
from .event_database import EventDatabase, EventStream

__all__ = [
    "PDFReader",
    "PDFProcessor",
    "WebScraper",
    "ArticleExtractor",
    "NewsAggregator",
    "EventExtractor",
    "GeopoliticalEvent",
    "EventType",
    "TemporalNormalizer",
    "CausalFeatureExtractor",
    "EventDatabase",
    "EventStream",
]
