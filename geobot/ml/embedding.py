"""
Geopolitical embeddings for text and entities.
"""

import numpy as np
from typing import List, Dict, Optional


class GeopoliticalEmbedding:
    """
    Create embeddings for geopolitical entities and text.

    Transforms text into risk vectors using NLP models.
    """

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize embedding model.

        Parameters
        ----------
        model_name : str
            Name of the embedding model
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            print("sentence-transformers not installed. Embeddings will not be available.")
            self.model = None

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into vectors.

        Parameters
        ----------
        texts : list
            List of texts to encode

        Returns
        -------
        np.ndarray
            Embeddings
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        return self.model.encode(texts)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.

        Parameters
        ----------
        text1 : str
            First text
        text2 : str
            Second text

        Returns
        -------
        float
            Cosine similarity
        """
        embeddings = self.encode_text([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / \
                    (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return float(similarity)
