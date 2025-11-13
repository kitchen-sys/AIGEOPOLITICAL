"""
Document Indexing System for Geopolitical Intelligence PDFs

Comprehensive indexing system with:
- Full-text search with ranking
- Metadata indexing (title, author, date, classification, source)
- Vector embeddings for semantic search
- Document-to-event linking
- Source reliability scoring
- Version control
- Citation tracking
- Cross-reference management
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class DocumentMetadata:
    """Metadata for an indexed document."""
    doc_id: str  # Unique document ID (hash of content + filename)
    file_path: str
    file_name: str
    title: str
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    ingest_date: datetime = None
    classification: str = "UNCLASSIFIED"  # UNCLASSIFIED, CONFIDENTIAL, SECRET, etc.
    document_type: str = "GENERAL"  # INTELLIGENCE, ANALYSIS, BRIEFING, REPORT, etc.
    source: str = "UNKNOWN"  # Organization/agency that produced document
    source_reliability: float = 0.5  # 0.0 (unreliable) to 1.0 (highly reliable)
    language: str = "en"
    num_pages: int = 0
    word_count: int = 0
    file_size: int = 0  # bytes
    file_hash: str = ""  # SHA-256 hash of file
    version: int = 1
    parent_doc_id: Optional[str] = None  # If this is an updated version
    tags: List[str] = None
    summary: str = ""
    keywords: List[str] = None
    mentioned_entities: Dict[str, List[str]] = None  # {entity_type: [entities]}
    confidence_score: float = 1.0  # Overall confidence in extraction quality

    def __post_init__(self):
        if self.ingest_date is None:
            self.ingest_date = datetime.utcnow()
        if self.tags is None:
            self.tags = []
        if self.keywords is None:
            self.keywords = []
        if self.mentioned_entities is None:
            self.mentioned_entities = {}


@dataclass
class DocumentChunk:
    """Chunk of document text for granular indexing."""
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_index: int  # Index within page
    text: str
    embedding: Optional[np.ndarray] = None  # Vector embedding for semantic search
    start_char: int = 0
    end_char: int = 0


class DocumentIndex:
    """
    Comprehensive document indexing system with full-text and semantic search.

    Example:
        >>> index = DocumentIndex("documents.db")
        >>>
        >>> # Index a PDF
        >>> metadata = index.index_pdf(
        ...     pdf_path="intelligence_report.pdf",
        ...     title="Iran Nuclear Program Assessment",
        ...     author="DIA",
        ...     classification="SECRET",
        ...     source="Defense Intelligence Agency",
        ...     source_reliability=0.95
        ... )
        >>>
        >>> # Full-text search
        >>> results = index.search_documents("nuclear enrichment facilities", limit=10)
        >>>
        >>> # Semantic search (if embeddings available)
        >>> results = index.semantic_search("threats to regional stability")
        >>>
        >>> # Link document to events
        >>> index.link_document_to_event(metadata.doc_id, event_id="evt_123")
        >>>
        >>> # Find related documents
        >>> related = index.find_related_documents(metadata.doc_id)
    """

    def __init__(self, db_path: str = "document_index.db"):
        """
        Initialize document index.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_schema()

    def _connect(self):
        """Connect to database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        # Enable full-text search
        self.conn.execute("PRAGMA journal_mode=WAL")

    def _create_schema(self):
        """Create database schema with full-text search support."""
        cursor = self.conn.cursor()

        # Documents table with metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                title TEXT NOT NULL,
                author TEXT,
                publish_date TEXT,
                ingest_date TEXT NOT NULL,
                classification TEXT DEFAULT 'UNCLASSIFIED',
                document_type TEXT DEFAULT 'GENERAL',
                source TEXT DEFAULT 'UNKNOWN',
                source_reliability REAL DEFAULT 0.5,
                language TEXT DEFAULT 'en',
                num_pages INTEGER,
                word_count INTEGER,
                file_size INTEGER,
                file_hash TEXT UNIQUE,
                version INTEGER DEFAULT 1,
                parent_doc_id TEXT,
                tags TEXT,
                summary TEXT,
                keywords TEXT,
                mentioned_entities TEXT,
                confidence_score REAL DEFAULT 1.0,
                FOREIGN KEY (parent_doc_id) REFERENCES documents(doc_id)
            )
        ''')

        # Full document text for full-text search
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
                doc_id UNINDEXED,
                title,
                text,
                summary,
                keywords,
                content=documents,
                content_rowid=rowid
            )
        ''')

        # Document chunks (for granular search and embeddings)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page_number INTEGER,
                chunk_index INTEGER,
                text TEXT NOT NULL,
                embedding BLOB,
                start_char INTEGER,
                end_char INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        ''')

        # Document-to-Event links
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_events (
                doc_id TEXT,
                event_id TEXT,
                relevance_score REAL DEFAULT 1.0,
                extracted_by TEXT DEFAULT 'manual',
                PRIMARY KEY (doc_id, event_id),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        ''')

        # Document citations (which documents cite which)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_citations (
                citing_doc_id TEXT,
                cited_doc_id TEXT,
                citation_context TEXT,
                page_number INTEGER,
                PRIMARY KEY (citing_doc_id, cited_doc_id),
                FOREIGN KEY (citing_doc_id) REFERENCES documents(doc_id),
                FOREIGN KEY (cited_doc_id) REFERENCES documents(doc_id)
            )
        ''')

        # Document access log (track usage)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_access_log (
                doc_id TEXT,
                access_time TEXT,
                access_type TEXT,
                user_id TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        ''')

        # Create indices for fast queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_title ON documents(title)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_author ON documents(author)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_publish_date ON documents(publish_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_ingest_date ON documents(ingest_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(document_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_source ON documents(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_classification ON documents(classification)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(file_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_doc ON document_chunks(doc_id)')

        self.conn.commit()

    def index_pdf(
        self,
        pdf_path: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        publish_date: Optional[datetime] = None,
        classification: str = "UNCLASSIFIED",
        document_type: str = "GENERAL",
        source: str = "UNKNOWN",
        source_reliability: float = 0.5,
        tags: Optional[List[str]] = None,
        extract_embeddings: bool = False
    ) -> DocumentMetadata:
        """
        Index a PDF document with full metadata extraction.

        Args:
            pdf_path: Path to PDF file
            title: Document title (extracted if not provided)
            author: Document author
            publish_date: Publication date
            classification: Security classification
            document_type: Type of document
            source: Source organization
            source_reliability: Reliability score (0-1)
            tags: Custom tags
            extract_embeddings: Whether to compute vector embeddings

        Returns:
            DocumentMetadata object
        """
        from .pdf_reader import PDFReader, PDFProcessor

        # Read PDF
        reader = PDFReader()
        processor = PDFProcessor(reader)

        pdf_data = reader.read_pdf(pdf_path)
        processed = processor.process_document(pdf_path)

        # Compute file hash for deduplication
        file_hash = self._compute_file_hash(pdf_path)

        # Check if already indexed
        existing = self._get_by_hash(file_hash)
        if existing:
            print(f"Document already indexed: {existing['doc_id']} (version {existing['version']})")
            return self._row_to_metadata(existing)

        # Generate document ID
        doc_id = self._generate_doc_id(pdf_path, file_hash)

        # Extract metadata
        pdf_metadata = pdf_data.get('metadata', {})
        if title is None:
            title = pdf_metadata.get('title') or pdf_metadata.get('/Title') or Path(pdf_path).stem

        if author is None:
            author = pdf_metadata.get('author') or pdf_metadata.get('/Author')

        # Create metadata object
        metadata = DocumentMetadata(
            doc_id=doc_id,
            file_path=str(Path(pdf_path).absolute()),
            file_name=Path(pdf_path).name,
            title=title,
            author=author,
            publish_date=publish_date,
            ingest_date=datetime.utcnow(),
            classification=classification,
            document_type=document_type,
            source=source,
            source_reliability=source_reliability,
            num_pages=pdf_data.get('num_pages', 0),
            word_count=processed.get('word_count', 0),
            file_size=Path(pdf_path).stat().st_size,
            file_hash=file_hash,
            tags=tags or [],
            summary=processed.get('summary', ''),
            keywords=[kw[0] for kw in processed.get('keywords', [])[:20]],
            mentioned_entities=processed.get('entities', {}),
            confidence_score=1.0
        )

        # Insert into database
        self._insert_document(metadata)

        # Insert full text for FTS
        full_text = pdf_data.get('text', '')
        self._insert_fts(doc_id, title, full_text, metadata.summary, metadata.keywords)

        # Index chunks for granular search
        self._index_chunks(doc_id, pdf_data.get('pages', []), extract_embeddings)

        # Log access
        self._log_access(doc_id, "INDEX", "system")

        return metadata

    def search_documents(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        classifications: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        min_reliability: Optional[float] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across documents.

        Args:
            query: Search query
            document_types: Filter by document types
            sources: Filter by sources
            classifications: Filter by classifications
            date_range: Filter by publish date (start, end)
            min_reliability: Minimum source reliability
            limit: Maximum results

        Returns:
            List of matching documents with relevance scores
        """
        cursor = self.conn.cursor()

        # Build FTS query
        fts_query = f'''
            SELECT
                d.*,
                fts.rank as relevance_score
            FROM document_fts fts
            JOIN documents d ON d.doc_id = fts.doc_id
            WHERE document_fts MATCH ?
        '''

        conditions = []
        params = [query]

        # Add filters
        if document_types:
            placeholders = ','.join('?' * len(document_types))
            conditions.append(f"d.document_type IN ({placeholders})")
            params.extend(document_types)

        if sources:
            placeholders = ','.join('?' * len(sources))
            conditions.append(f"d.source IN ({placeholders})")
            params.extend(sources)

        if classifications:
            placeholders = ','.join('?' * len(classifications))
            conditions.append(f"d.classification IN ({placeholders})")
            params.extend(classifications)

        if date_range:
            start, end = date_range
            conditions.append("d.publish_date BETWEEN ? AND ?")
            params.extend([start.isoformat(), end.isoformat()])

        if min_reliability is not None:
            conditions.append("d.source_reliability >= ?")
            params.append(min_reliability)

        if conditions:
            fts_query += " AND " + " AND ".join(conditions)

        fts_query += " ORDER BY relevance_score DESC LIMIT ?"
        params.append(limit)

        cursor.execute(fts_query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            doc_dict = dict(row)
            # Parse JSON fields
            doc_dict['tags'] = json.loads(doc_dict.get('tags', '[]'))
            doc_dict['keywords'] = json.loads(doc_dict.get('keywords', '[]'))
            doc_dict['mentioned_entities'] = json.loads(doc_dict.get('mentioned_entities', '{}'))
            results.append(doc_dict)

        # Log searches
        for result in results:
            self._log_access(result['doc_id'], "SEARCH", "system")

        return results

    def semantic_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using vector embeddings.

        Args:
            query_embedding: Query vector (must match indexed embeddings)
            top_k: Number of results
            threshold: Minimum cosine similarity

        Returns:
            List of matching chunks with similarity scores
        """
        cursor = self.conn.cursor()

        # Get all chunks with embeddings
        cursor.execute('''
            SELECT chunk_id, doc_id, page_number, text, embedding
            FROM document_chunks
            WHERE embedding IS NOT NULL
        ''')

        results = []
        for row in cursor.fetchall():
            # Deserialize embedding
            embedding_bytes = row['embedding']
            chunk_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

            # Compute cosine similarity
            similarity = self._cosine_similarity(query_embedding, chunk_embedding)

            if similarity >= threshold:
                results.append({
                    'chunk_id': row['chunk_id'],
                    'doc_id': row['doc_id'],
                    'page_number': row['page_number'],
                    'text': row['text'],
                    'similarity': float(similarity)
                })

        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results[:top_k]

    def link_document_to_event(
        self,
        doc_id: str,
        event_id: str,
        relevance_score: float = 1.0,
        extracted_by: str = "manual"
    ) -> None:
        """
        Link document to a geopolitical event.

        Args:
            doc_id: Document ID
            event_id: Event ID
            relevance_score: How relevant (0-1)
            extracted_by: Extraction method
        """
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO document_events
            (doc_id, event_id, relevance_score, extracted_by)
            VALUES (?, ?, ?, ?)
        ''', (doc_id, event_id, relevance_score, extracted_by))

        self.conn.commit()

    def get_events_for_document(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all events linked to a document."""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT event_id, relevance_score, extracted_by
            FROM document_events
            WHERE doc_id = ?
            ORDER BY relevance_score DESC
        ''', (doc_id,))

        return [dict(row) for row in cursor.fetchall()]

    def get_documents_for_event(self, event_id: str) -> List[Dict[str, Any]]:
        """Get all documents linked to an event."""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT d.*, de.relevance_score
            FROM documents d
            JOIN document_events de ON d.doc_id = de.doc_id
            WHERE de.event_id = ?
            ORDER BY de.relevance_score DESC
        ''', (event_id,))

        results = []
        for row in cursor.fetchall():
            doc_dict = dict(row)
            doc_dict['tags'] = json.loads(doc_dict.get('tags', '[]'))
            doc_dict['keywords'] = json.loads(doc_dict.get('keywords', '[]'))
            doc_dict['mentioned_entities'] = json.loads(doc_dict.get('mentioned_entities', '{}'))
            results.append(doc_dict)

        return results

    def add_citation(
        self,
        citing_doc_id: str,
        cited_doc_id: str,
        citation_context: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> None:
        """
        Record that one document cites another.

        Args:
            citing_doc_id: Document that does the citing
            cited_doc_id: Document being cited
            citation_context: Surrounding text of citation
            page_number: Page where citation appears
        """
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO document_citations
            (citing_doc_id, cited_doc_id, citation_context, page_number)
            VALUES (?, ?, ?, ?)
        ''', (citing_doc_id, cited_doc_id, citation_context, page_number))

        self.conn.commit()

    def find_related_documents(
        self,
        doc_id: str,
        method: str = "citations",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find documents related to a given document.

        Args:
            doc_id: Source document ID
            method: Relationship method ("citations", "events", "entities", "keywords")
            limit: Maximum results

        Returns:
            List of related documents
        """
        if method == "citations":
            return self._related_by_citations(doc_id, limit)
        elif method == "events":
            return self._related_by_events(doc_id, limit)
        elif method == "entities":
            return self._related_by_entities(doc_id, limit)
        elif method == "keywords":
            return self._related_by_keywords(doc_id, limit)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents."""
        cursor = self.conn.cursor()

        stats = {}

        # Total documents
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        stats['total_documents'] = cursor.fetchone()['count']

        # By document type
        cursor.execute('''
            SELECT document_type, COUNT(*) as count
            FROM documents
            GROUP BY document_type
        ''')
        stats['by_type'] = {row['document_type']: row['count'] for row in cursor.fetchall()}

        # By classification
        cursor.execute('''
            SELECT classification, COUNT(*) as count
            FROM documents
            GROUP BY classification
        ''')
        stats['by_classification'] = {row['classification']: row['count'] for row in cursor.fetchall()}

        # By source
        cursor.execute('''
            SELECT source, COUNT(*) as count, AVG(source_reliability) as avg_reliability
            FROM documents
            GROUP BY source
            ORDER BY count DESC
            LIMIT 10
        ''')
        stats['top_sources'] = [dict(row) for row in cursor.fetchall()]

        # Total storage
        cursor.execute("SELECT SUM(file_size) as total_size FROM documents")
        stats['total_file_size'] = cursor.fetchone()['total_size'] or 0

        # Average reliability
        cursor.execute("SELECT AVG(source_reliability) as avg_reliability FROM documents")
        stats['avg_source_reliability'] = cursor.fetchone()['avg_reliability']

        return stats

    # Helper methods

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _generate_doc_id(self, file_path: str, file_hash: str) -> str:
        """Generate unique document ID."""
        # Use file hash + filename for uniqueness
        combined = f"{file_hash}_{Path(file_path).stem}"
        return f"doc_{hashlib.md5(combined.encode()).hexdigest()[:12]}"

    def _get_by_hash(self, file_hash: str) -> Optional[sqlite3.Row]:
        """Check if document with hash already exists."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
        return cursor.fetchone()

    def _row_to_metadata(self, row: sqlite3.Row) -> DocumentMetadata:
        """Convert database row to DocumentMetadata."""
        return DocumentMetadata(
            doc_id=row['doc_id'],
            file_path=row['file_path'],
            file_name=row['file_name'],
            title=row['title'],
            author=row['author'],
            publish_date=datetime.fromisoformat(row['publish_date']) if row['publish_date'] else None,
            ingest_date=datetime.fromisoformat(row['ingest_date']),
            classification=row['classification'],
            document_type=row['document_type'],
            source=row['source'],
            source_reliability=row['source_reliability'],
            language=row['language'],
            num_pages=row['num_pages'],
            word_count=row['word_count'],
            file_size=row['file_size'],
            file_hash=row['file_hash'],
            version=row['version'],
            parent_doc_id=row['parent_doc_id'],
            tags=json.loads(row['tags']),
            summary=row['summary'],
            keywords=json.loads(row['keywords']),
            mentioned_entities=json.loads(row['mentioned_entities']),
            confidence_score=row['confidence_score']
        )

    def _insert_document(self, metadata: DocumentMetadata) -> None:
        """Insert document metadata."""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO documents (
                doc_id, file_path, file_name, title, author, publish_date, ingest_date,
                classification, document_type, source, source_reliability, language,
                num_pages, word_count, file_size, file_hash, version, parent_doc_id,
                tags, summary, keywords, mentioned_entities, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metadata.doc_id,
            metadata.file_path,
            metadata.file_name,
            metadata.title,
            metadata.author,
            metadata.publish_date.isoformat() if metadata.publish_date else None,
            metadata.ingest_date.isoformat(),
            metadata.classification,
            metadata.document_type,
            metadata.source,
            metadata.source_reliability,
            metadata.language,
            metadata.num_pages,
            metadata.word_count,
            metadata.file_size,
            metadata.file_hash,
            metadata.version,
            metadata.parent_doc_id,
            json.dumps(metadata.tags),
            metadata.summary,
            json.dumps(metadata.keywords),
            json.dumps(metadata.mentioned_entities),
            metadata.confidence_score
        ))

        self.conn.commit()

    def _insert_fts(
        self,
        doc_id: str,
        title: str,
        text: str,
        summary: str,
        keywords: List[str]
    ) -> None:
        """Insert into full-text search index."""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO document_fts (doc_id, title, text, summary, keywords)
            VALUES (?, ?, ?, ?, ?)
        ''', (doc_id, title, text, summary, ' '.join(keywords)))

        self.conn.commit()

    def _index_chunks(
        self,
        doc_id: str,
        pages: List[Dict],
        extract_embeddings: bool = False
    ) -> None:
        """Index document in chunks for granular search."""
        cursor = self.conn.cursor()

        for page in pages:
            page_num = page.get('page_number', 0)
            page_text = page.get('text', '')

            if not page_text:
                continue

            # Split into chunks (e.g., every 500 characters)
            chunk_size = 500
            for i, start in enumerate(range(0, len(page_text), chunk_size)):
                chunk_text = page_text[start:start + chunk_size]
                chunk_id = f"{doc_id}_p{page_num}_c{i}"

                embedding_bytes = None
                if extract_embeddings and chunk_text.strip():
                    # Placeholder for embedding extraction
                    # In production, use sentence-transformers or similar
                    # embedding = model.encode(chunk_text)
                    # embedding_bytes = embedding.astype(np.float32).tobytes()
                    pass

                cursor.execute('''
                    INSERT INTO document_chunks
                    (chunk_id, doc_id, page_number, chunk_index, text, embedding, start_char, end_char)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chunk_id,
                    doc_id,
                    page_num,
                    i,
                    chunk_text,
                    embedding_bytes,
                    start,
                    start + len(chunk_text)
                ))

        self.conn.commit()

    def _log_access(self, doc_id: str, access_type: str, user_id: str) -> None:
        """Log document access."""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO document_access_log (doc_id, access_time, access_type, user_id)
            VALUES (?, ?, ?, ?)
        ''', (doc_id, datetime.utcnow().isoformat(), access_type, user_id))

        self.conn.commit()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _related_by_citations(self, doc_id: str, limit: int) -> List[Dict]:
        """Find documents related by citations."""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT d.*, 'cited_by' as relationship
            FROM documents d
            JOIN document_citations dc ON d.doc_id = dc.citing_doc_id
            WHERE dc.cited_doc_id = ?
            UNION
            SELECT d.*, 'cites' as relationship
            FROM documents d
            JOIN document_citations dc ON d.doc_id = dc.cited_doc_id
            WHERE dc.citing_doc_id = ?
            LIMIT ?
        ''', (doc_id, doc_id, limit))

        return [dict(row) for row in cursor.fetchall()]

    def _related_by_events(self, doc_id: str, limit: int) -> List[Dict]:
        """Find documents related by shared events."""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT d.*, COUNT(*) as shared_events
            FROM documents d
            JOIN document_events de1 ON d.doc_id = de1.doc_id
            JOIN document_events de2 ON de1.event_id = de2.event_id
            WHERE de2.doc_id = ? AND d.doc_id != ?
            GROUP BY d.doc_id
            ORDER BY shared_events DESC
            LIMIT ?
        ''', (doc_id, doc_id, limit))

        return [dict(row) for row in cursor.fetchall()]

    def _related_by_entities(self, doc_id: str, limit: int) -> List[Dict]:
        """Find documents related by shared entities."""
        # Get source document entities
        cursor = self.conn.cursor()
        cursor.execute("SELECT mentioned_entities FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        if not row:
            return []

        source_entities = json.loads(row['mentioned_entities'])
        source_countries = set(source_entities.get('countries', []))

        if not source_countries:
            return []

        # Find documents with overlapping entities
        cursor.execute("SELECT doc_id, mentioned_entities FROM documents WHERE doc_id != ?", (doc_id,))

        matches = []
        for row in cursor.fetchall():
            entities = json.loads(row['mentioned_entities'])
            countries = set(entities.get('countries', []))
            overlap = len(source_countries & countries)
            if overlap > 0:
                matches.append((row['doc_id'], overlap))

        # Sort by overlap
        matches.sort(key=lambda x: x[1], reverse=True)

        # Get full document info
        results = []
        for doc_id, overlap in matches[:limit]:
            cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            doc_dict = dict(cursor.fetchone())
            doc_dict['entity_overlap'] = overlap
            results.append(doc_dict)

        return results

    def _related_by_keywords(self, doc_id: str, limit: int) -> List[Dict]:
        """Find documents related by shared keywords."""
        cursor = self.conn.cursor()

        # Get source keywords
        cursor.execute("SELECT keywords FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        if not row:
            return []

        source_keywords = set(json.loads(row['keywords']))

        if not source_keywords:
            return []

        # Find documents with overlapping keywords
        cursor.execute("SELECT doc_id, title, keywords FROM documents WHERE doc_id != ?", (doc_id,))

        matches = []
        for row in cursor.fetchall():
            keywords = set(json.loads(row['keywords']))
            overlap = len(source_keywords & keywords)
            if overlap > 0:
                matches.append((row['doc_id'], overlap))

        matches.sort(key=lambda x: x[1], reverse=True)

        # Get full document info
        results = []
        for doc_id, overlap in matches[:limit]:
            cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            doc_dict = dict(cursor.fetchone())
            doc_dict['keyword_overlap'] = overlap
            results.append(doc_dict)

        return results

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
