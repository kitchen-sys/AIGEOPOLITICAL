# GeoBotv1 Examples

This directory contains example scripts demonstrating the capabilities of GeoBotv1.

## Examples Overview

### 01_basic_usage.py
Basic introduction to GeoBotv1 core components:
- Creating geopolitical scenarios
- Building causal graphs
- Running Monte Carlo simulations
- Bayesian belief updating
- Uncertainty quantification

**Run it:**
```bash
python 01_basic_usage.py
```

### 02_data_ingestion.py
Demonstrates data ingestion capabilities:
- PDF document processing
- Web scraping and article extraction
- News aggregation from multiple sources
- Intelligence extraction from documents
- Entity and keyword extraction

**Run it:**
```bash
python 02_data_ingestion.py
```

**Note:** For full functionality, install optional dependencies:
```bash
pip install pypdf pdfplumber beautifulsoup4 newspaper3k trafilatura feedparser
```

### 03_intervention_simulation.py
Advanced intervention and counterfactual analysis:
- Policy intervention simulation
- Comparing multiple policy options
- Finding optimal interventions
- Counterfactual reasoning ("what if" scenarios)
- Causal effect estimation

**Run it:**
```bash
python 03_intervention_simulation.py
```

## Additional Resources

### Creating Custom Scenarios
```python
from geobot.core.scenario import Scenario
import numpy as np

scenario = Scenario(
    name="custom_scenario",
    features={
        'tension': np.array([0.7]),
        'stability': np.array([0.4]),
    },
    probability=1.0
)
```

### Building Causal Models
```python
from geobot.models.causal_graph import CausalGraph

graph = CausalGraph(name="my_model")
graph.add_node('cause')
graph.add_node('effect')
graph.add_edge('cause', 'effect', strength=0.8)
```

### Monte Carlo Simulation
```python
from geobot.simulation.monte_carlo import MonteCarloEngine, SimulationConfig

config = SimulationConfig(n_simulations=1000, time_horizon=100)
engine = MonteCarloEngine(config)
```

### Web Scraping
```python
from geobot.data_ingestion.web_scraper import ArticleExtractor

extractor = ArticleExtractor()
article = extractor.extract_article('https://example.com/article')
print(article['title'])
print(article['text'])
```

### PDF Processing
```python
from geobot.data_ingestion.pdf_reader import PDFProcessor

processor = PDFProcessor()
result = processor.extract_intelligence('report.pdf')
print(f"Risk Level: {result['intelligence']['risk_level']}")
```

## Need Help?

- Check the main README.md in the project root
- Review the module documentation in each package
- Examine the source code for detailed implementation

## Contributing

Have an interesting use case? Create a new example script and submit a PR!
