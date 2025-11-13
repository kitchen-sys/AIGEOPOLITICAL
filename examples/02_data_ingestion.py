"""
Example 2: Data Ingestion - PDF and Web Scraping

This example demonstrates:
- PDF document reading and processing
- Web article extraction
- News aggregation
- Intelligence extraction from documents
"""

import sys
sys.path.append('..')

from geobot.data_ingestion.pdf_reader import PDFReader, PDFProcessor
from geobot.data_ingestion.web_scraper import WebScraper, ArticleExtractor, NewsAggregator


def demo_pdf_processing():
    """Demonstrate PDF processing capabilities."""
    print("\n" + "=" * 80)
    print("PDF Processing Demo")
    print("=" * 80)

    # Create PDF processor
    processor = PDFProcessor()

    print("\nPDF processing capabilities:")
    print("- Text extraction from PDFs")
    print("- Table extraction")
    print("- Metadata extraction")
    print("- Entity recognition (countries, organizations)")
    print("- Keyword extraction")
    print("- Risk assessment")
    print("\nTo use: processor.process_document('path/to/document.pdf')")

    # Example code structure
    example_code = """
    # Process a single PDF
    result = processor.process_document('intelligence_report.pdf')

    print(f"Title: {result['metadata'].get('title', 'Unknown')}")
    print(f"Pages: {result['num_pages']}")
    print(f"Keywords: {result['keywords']}")
    print(f"Risk Level: {result['intelligence']['risk_level']}")

    # Process multiple PDFs
    results = processor.batch_process('reports_directory/', '*.pdf')
    """

    print("\nExample usage:")
    print(example_code)


def demo_web_scraping():
    """Demonstrate web scraping capabilities."""
    print("\n" + "=" * 80)
    print("Web Scraping Demo")
    print("=" * 80)

    # Create article extractor
    extractor = ArticleExtractor()

    print("\nWeb scraping capabilities:")
    print("- Extract articles from URLs")
    print("- Clean HTML content")
    print("- Extract metadata (author, date, etc.)")
    print("- Multiple extraction methods (newspaper3k, trafilatura, BeautifulSoup)")

    # Example with a well-known news site (without actually fetching)
    example_url = "https://www.example.com/geopolitical-analysis"

    print(f"\nExample: Extracting article from {example_url}")
    print("(This is a demonstration - no actual web request is made)")

    example_code = """
    # Extract article
    article = extractor.extract_article(url)

    print(f"Title: {article['title']}")
    print(f"Author: {article['authors']}")
    print(f"Published: {article['publish_date']}")
    print(f"Content length: {len(article['text'])} characters")

    # Extract multiple articles
    urls = ['url1', 'url2', 'url3']
    articles = extractor.batch_extract(urls)
    """

    print("\nExample usage:")
    print(example_code)


def demo_news_aggregation():
    """Demonstrate news aggregation capabilities."""
    print("\n" + "=" * 80)
    print("News Aggregation Demo")
    print("=" * 80)

    aggregator = NewsAggregator()

    print("\nNews aggregation capabilities:")
    print("- Aggregate from multiple sources")
    print("- RSS feed support")
    print("- Keyword filtering")
    print("- Trending topic detection")
    print("- Real-time monitoring")

    # Example configuration
    print("\nExample: Setting up news aggregation")

    example_code = """
    # Add news sources
    aggregator.add_source(
        name='Reuters',
        url='https://www.reuters.com/news/world',
        source_type='rss'
    )

    aggregator.add_source(
        name='Al Jazeera',
        url='https://www.aljazeera.com/xml/rss/all.xml',
        source_type='rss'
    )

    # Fetch news with keywords
    keywords = ['sanctions', 'conflict', 'diplomacy', 'military']
    articles = aggregator.fetch_news(keywords)

    print(f"Found {len(articles)} relevant articles")

    # Get trending topics
    topics = aggregator.get_trending_topics(articles, n_topics=10)
    print("Trending topics:", topics)

    # Monitor sources continuously
    def alert_callback(new_articles):
        print(f"ALERT: {len(new_articles)} new relevant articles found")
        for article in new_articles:
            print(f"  - {article['title']}")

    # Monitor every hour
    aggregator.monitor_sources(keywords, callback=alert_callback, interval=3600)
    """

    print(example_code)


def demo_intelligence_extraction():
    """Demonstrate intelligence extraction from documents."""
    print("\n" + "=" * 80)
    print("Intelligence Extraction Demo")
    print("=" * 80)

    print("\nIntelligence extraction capabilities:")
    print("- Country and organization detection")
    print("- Conflict indicator detection")
    print("- Risk level assessment")
    print("- Document classification")
    print("- Key phrase extraction")

    example_code = """
    processor = PDFProcessor()

    # Extract intelligence from PDF
    intel = processor.extract_intelligence('report.pdf')

    print("Intelligence Summary:")
    print(f"Risk Level: {intel['intelligence']['risk_level']}")
    print(f"Countries mentioned: {intel['intelligence']['mentioned_countries']}")
    print(f"Conflict indicators: {intel['intelligence']['conflict_indicators']}")
    print(f"Key topics: {intel['intelligence']['key_topics']}")
    print(f"Document type: {intel['intelligence']['document_type']}")
    """

    print("\nExample usage:")
    print(example_code)


def main():
    print("=" * 80)
    print("GeoBotv1 - Data Ingestion Examples")
    print("=" * 80)
    print("\nThis module demonstrates the data ingestion capabilities of GeoBotv1:")
    print("1. PDF document processing")
    print("2. Web scraping and article extraction")
    print("3. News aggregation from multiple sources")
    print("4. Intelligence extraction from documents")

    demo_pdf_processing()
    demo_web_scraping()
    demo_news_aggregation()
    demo_intelligence_extraction()

    print("\n" + "=" * 80)
    print("Data Ingestion Demo Complete")
    print("=" * 80)
    print("\nNote: Install required packages for full functionality:")
    print("  pip install pypdf pdfplumber beautifulsoup4 newspaper3k trafilatura")


if __name__ == "__main__":
    main()
