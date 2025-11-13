"""
PDF Reading and Processing Module

Comprehensive PDF ingestion capabilities for geopolitical intelligence documents,
reports, briefings, and analysis.

Supports:
- Text extraction from PDFs
- Table extraction
- Metadata extraction
- Multi-format PDF handling
- Batch processing
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import re


class PDFReader:
    """
    Read and extract text from PDF documents.

    Supports multiple PDF libraries for robust extraction.
    """

    def __init__(self, method: str = 'auto'):
        """
        Initialize PDF reader.

        Parameters
        ----------
        method : str
            Extraction method ('pypdf', 'pdfplumber', 'pdfminer', 'auto')
        """
        self.method = method
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check which PDF libraries are available."""
        self.has_pypdf = False
        self.has_pdfplumber = False
        self.has_pdfminer = False

        try:
            import pypdf
            self.has_pypdf = True
        except ImportError:
            pass

        try:
            import pdfplumber
            self.has_pdfplumber = True
        except ImportError:
            pass

        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            self.has_pdfminer = True
        except ImportError:
            pass

        if not any([self.has_pypdf, self.has_pdfplumber, self.has_pdfminer]):
            print("Warning: No PDF libraries available. Please install pypdf, pdfplumber, or pdfminer.six")

    def read_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Read PDF and extract all information.

        Parameters
        ----------
        pdf_path : str
            Path to PDF file

        Returns
        -------
        dict
            Extracted information including text, metadata, pages
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        method = self.method
        if method == 'auto':
            # Choose best available method
            if self.has_pdfplumber:
                method = 'pdfplumber'
            elif self.has_pypdf:
                method = 'pypdf'
            elif self.has_pdfminer:
                method = 'pdfminer'
            else:
                raise ImportError("No PDF library available")

        if method == 'pypdf':
            return self._read_with_pypdf(pdf_path)
        elif method == 'pdfplumber':
            return self._read_with_pdfplumber(pdf_path)
        elif method == 'pdfminer':
            return self._read_with_pdfminer(pdf_path)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _read_with_pypdf(self, pdf_path: str) -> Dict[str, Any]:
        """Read PDF using pypdf."""
        import pypdf

        result = {
            'text': '',
            'pages': [],
            'metadata': {},
            'num_pages': 0
        }

        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            result['num_pages'] = len(reader.pages)

            # Extract metadata
            if reader.metadata:
                result['metadata'] = {
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'subject': reader.metadata.get('/Subject', ''),
                    'creator': reader.metadata.get('/Creator', ''),
                }

            # Extract text from each page
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                result['pages'].append({
                    'page_number': page_num + 1,
                    'text': page_text
                })
                result['text'] += page_text + '\n'

        return result

    def _read_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Read PDF using pdfplumber (best for tables)."""
        import pdfplumber

        result = {
            'text': '',
            'pages': [],
            'tables': [],
            'metadata': {},
            'num_pages': 0
        }

        with pdfplumber.open(pdf_path) as pdf:
            result['num_pages'] = len(pdf.pages)
            result['metadata'] = pdf.metadata

            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                page_tables = page.extract_tables()

                result['pages'].append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'tables': page_tables
                })

                result['text'] += page_text + '\n' if page_text else ''

                if page_tables:
                    result['tables'].extend([{
                        'page': page_num + 1,
                        'data': table
                    } for table in page_tables])

        return result

    def _read_with_pdfminer(self, pdf_path: str) -> Dict[str, Any]:
        """Read PDF using pdfminer."""
        from pdfminer.high_level import extract_text, extract_pages
        from pdfminer.layout import LTTextContainer

        result = {
            'text': '',
            'pages': [],
            'metadata': {},
            'num_pages': 0
        }

        # Extract all text
        result['text'] = extract_text(pdf_path)

        # Extract page by page
        pages = list(extract_pages(pdf_path))
        result['num_pages'] = len(pages)

        for page_num, page_layout in enumerate(pages):
            page_text = ''
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    page_text += element.get_text()

            result['pages'].append({
                'page_number': page_num + 1,
                'text': page_text
            })

        return result

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF (simple interface).

        Parameters
        ----------
        pdf_path : str
            Path to PDF

        Returns
        -------
        str
            Extracted text
        """
        result = self.read_pdf(pdf_path)
        return result['text']

    def extract_tables(self, pdf_path: str) -> List[List[List[str]]]:
        """
        Extract tables from PDF.

        Parameters
        ----------
        pdf_path : str
            Path to PDF

        Returns
        -------
        list
            List of tables
        """
        if not self.has_pdfplumber:
            print("Warning: pdfplumber required for table extraction")
            return []

        result = self._read_with_pdfplumber(pdf_path)
        return [table['data'] for table in result.get('tables', [])]


class PDFProcessor:
    """
    Process and analyze PDF documents for geopolitical intelligence.

    Provides high-level processing capabilities including:
    - Entity extraction
    - Topic extraction
    - Sentiment analysis
    - Key phrase extraction
    """

    def __init__(self, pdf_reader: Optional[PDFReader] = None):
        """
        Initialize PDF processor.

        Parameters
        ----------
        pdf_reader : PDFReader, optional
            PDF reader to use
        """
        self.reader = pdf_reader or PDFReader()

    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process PDF document and extract intelligence.

        Parameters
        ----------
        pdf_path : str
            Path to PDF

        Returns
        -------
        dict
            Processed document with analysis
        """
        # Extract content
        content = self.reader.read_pdf(pdf_path)

        # Basic processing
        processed = {
            'file_path': pdf_path,
            'file_name': Path(pdf_path).name,
            'text': content['text'],
            'num_pages': content['num_pages'],
            'metadata': content.get('metadata', {}),
            'word_count': len(content['text'].split()),
            'char_count': len(content['text']),
        }

        # Extract key information
        processed['entities'] = self._extract_entities(content['text'])
        processed['keywords'] = self._extract_keywords(content['text'])
        processed['summary'] = self._generate_summary(content['text'])

        return processed

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities (countries, organizations, people).

        Parameters
        ----------
        text : str
            Text to analyze

        Returns
        -------
        dict
            Extracted entities by type
        """
        entities = {
            'countries': [],
            'organizations': [],
            'people': [],
            'locations': []
        }

        # Simple pattern-based extraction (can be enhanced with NER)
        # Common country names
        countries = ['United States', 'China', 'Russia', 'Iran', 'North Korea',
                    'India', 'Pakistan', 'Israel', 'Saudi Arabia', 'Turkey',
                    'France', 'Germany', 'United Kingdom', 'Japan', 'South Korea']

        for country in countries:
            if country in text:
                entities['countries'].append(country)

        # Organizations (simple patterns)
        org_patterns = [r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:Organization|Agency|Ministry|Department|Council)\b']
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            entities['organizations'].extend(matches)

        return entities

    def _extract_keywords(self, text: str, n_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords from text.

        Parameters
        ----------
        text : str
            Text to analyze
        n_keywords : int
            Number of keywords to extract

        Returns
        -------
        list
            List of (keyword, score) tuples
        """
        # Simple frequency-based extraction
        words = text.lower().split()

        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                    'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                    'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                    'does', 'did', 'will', 'would', 'should', 'could', 'may',
                    'might', 'can', 'this', 'that', 'these', 'those'}

        words = [w for w in words if w not in stopwords and len(w) > 3]

        # Count frequencies
        from collections import Counter
        word_freq = Counter(words)

        # Return top keywords
        return word_freq.most_common(n_keywords)

    def _generate_summary(self, text: str, num_sentences: int = 3) -> str:
        """
        Generate simple extractive summary.

        Parameters
        ----------
        text : str
            Text to summarize
        num_sentences : int
            Number of sentences in summary

        Returns
        -------
        str
            Summary
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Take first few sentences as summary (simple approach)
        summary_sentences = sentences[:num_sentences]

        return '. '.join(summary_sentences) + '.'

    def batch_process(self, pdf_directory: str, pattern: str = '*.pdf') -> List[Dict[str, Any]]:
        """
        Process multiple PDFs in a directory.

        Parameters
        ----------
        pdf_directory : str
            Directory containing PDFs
        pattern : str
            File pattern to match

        Returns
        -------
        list
            List of processed documents
        """
        pdf_dir = Path(pdf_directory)
        pdf_files = list(pdf_dir.glob(pattern))

        results = []
        for pdf_file in pdf_files:
            try:
                processed = self.process_document(str(pdf_file))
                results.append(processed)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

        return results

    def extract_intelligence(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract geopolitical intelligence from PDF.

        Parameters
        ----------
        pdf_path : str
            Path to PDF

        Returns
        -------
        dict
            Intelligence summary
        """
        processed = self.process_document(pdf_path)

        # Analyze for geopolitical indicators
        text = processed['text'].lower()

        indicators = {
            'conflict_indicators': self._detect_conflict_indicators(text),
            'risk_level': self._assess_risk_level(text),
            'mentioned_countries': processed['entities'].get('countries', []),
            'key_topics': [kw[0] for kw in processed['keywords'][:5]],
            'document_type': self._classify_document_type(text)
        }

        return {**processed, 'intelligence': indicators}

    def _detect_conflict_indicators(self, text: str) -> List[str]:
        """Detect conflict-related keywords."""
        conflict_keywords = ['war', 'conflict', 'military', 'attack', 'invasion',
                            'sanctions', 'escalation', 'tension', 'threat', 'crisis']

        detected = [kw for kw in conflict_keywords if kw in text]
        return detected

    def _assess_risk_level(self, text: str) -> str:
        """Simple risk level assessment."""
        high_risk_terms = ['imminent', 'urgent', 'critical', 'severe', 'escalating']
        medium_risk_terms = ['concern', 'monitoring', 'potential', 'emerging']

        high_count = sum(1 for term in high_risk_terms if term in text)
        medium_count = sum(1 for term in medium_risk_terms if term in text)

        if high_count > 2:
            return 'HIGH'
        elif medium_count > 2:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _classify_document_type(self, text: str) -> str:
        """Classify document type."""
        if 'intelligence report' in text or 'classified' in text:
            return 'Intelligence Report'
        elif 'analysis' in text or 'assessment' in text:
            return 'Analysis'
        elif 'briefing' in text:
            return 'Briefing'
        else:
            return 'General Document'
