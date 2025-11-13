"""
Settings and configuration for GeoBotv1
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class Settings:
    """
    Global settings for GeoBotv1.
    """
    # Simulation settings
    default_n_simulations: int = 1000
    default_time_horizon: int = 100
    random_seed: Optional[int] = None

    # Data ingestion settings
    pdf_extraction_method: str = 'auto'
    web_scraping_timeout: int = 30
    article_extraction_method: str = 'auto'

    # ML settings
    risk_scoring_method: str = 'gradient_boosting'
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'

    # Bayesian inference settings
    bayesian_method: str = 'grid'
    n_mcmc_samples: int = 10000

    # Causal inference settings
    causal_discovery_method: str = 'pc'
    causal_discovery_alpha: float = 0.05

    # Data directories
    data_dir: str = 'data'
    cache_dir: str = '.cache'
    output_dir: str = 'output'

    # Logging
    log_level: str = 'INFO'
    log_file: Optional[str] = None

    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'simulation': {
                'default_n_simulations': self.default_n_simulations,
                'default_time_horizon': self.default_time_horizon,
                'random_seed': self.random_seed
            },
            'data_ingestion': {
                'pdf_extraction_method': self.pdf_extraction_method,
                'web_scraping_timeout': self.web_scraping_timeout,
                'article_extraction_method': self.article_extraction_method
            },
            'ml': {
                'risk_scoring_method': self.risk_scoring_method,
                'embedding_model': self.embedding_model
            },
            'bayesian': {
                'method': self.bayesian_method,
                'n_mcmc_samples': self.n_mcmc_samples
            },
            'causal': {
                'discovery_method': self.causal_discovery_method,
                'discovery_alpha': self.causal_discovery_alpha
            },
            'directories': {
                'data_dir': self.data_dir,
                'cache_dir': self.cache_dir,
                'output_dir': self.output_dir
            },
            'logging': {
                'log_level': self.log_level,
                'log_file': self.log_file
            },
            'custom': self.custom
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Settings':
        """Load settings from dictionary."""
        settings = cls()

        if 'simulation' in data:
            settings.default_n_simulations = data['simulation'].get('default_n_simulations', 1000)
            settings.default_time_horizon = data['simulation'].get('default_time_horizon', 100)
            settings.random_seed = data['simulation'].get('random_seed')

        if 'data_ingestion' in data:
            settings.pdf_extraction_method = data['data_ingestion'].get('pdf_extraction_method', 'auto')
            settings.web_scraping_timeout = data['data_ingestion'].get('web_scraping_timeout', 30)
            settings.article_extraction_method = data['data_ingestion'].get('article_extraction_method', 'auto')

        if 'ml' in data:
            settings.risk_scoring_method = data['ml'].get('risk_scoring_method', 'gradient_boosting')
            settings.embedding_model = data['ml'].get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')

        if 'bayesian' in data:
            settings.bayesian_method = data['bayesian'].get('method', 'grid')
            settings.n_mcmc_samples = data['bayesian'].get('n_mcmc_samples', 10000)

        if 'causal' in data:
            settings.causal_discovery_method = data['causal'].get('discovery_method', 'pc')
            settings.causal_discovery_alpha = data['causal'].get('discovery_alpha', 0.05)

        if 'directories' in data:
            settings.data_dir = data['directories'].get('data_dir', 'data')
            settings.cache_dir = data['directories'].get('cache_dir', '.cache')
            settings.output_dir = data['directories'].get('output_dir', 'output')

        if 'logging' in data:
            settings.log_level = data['logging'].get('log_level', 'INFO')
            settings.log_file = data['logging'].get('log_file')

        if 'custom' in data:
            settings.custom = data['custom']

        return settings

    def save(self, path: str) -> None:
        """Save settings to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> 'Settings':
        """Load settings from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


# Global settings instance
_global_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get global settings instance.

    Returns
    -------
    Settings
        Global settings
    """
    global _global_settings
    if _global_settings is None:
        _global_settings = Settings()
    return _global_settings


def update_settings(settings: Settings) -> None:
    """
    Update global settings.

    Parameters
    ----------
    settings : Settings
        New settings
    """
    global _global_settings
    _global_settings = settings


def load_settings_from_file(path: str) -> None:
    """
    Load settings from file and update global settings.

    Parameters
    ----------
    path : str
        Path to settings file
    """
    settings = Settings.load(path)
    update_settings(settings)
