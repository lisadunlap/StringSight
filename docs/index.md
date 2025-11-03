# Welcome to StringSight

**Extract, cluster, and analyze behavioral properties from Large Language Models**

StringSight helps you understand how different generative models behave by automatically extracting behavioral properties from their responses, grouping similar behaviors together, and quantifying how important these behaviors are.

## What is StringSight?

StringSight is a comprehensive analysis framework for evaluating and comparing Large Language Model (LLM) responses. Instead of just measuring accuracy or overall quality scores, StringSight:

1. **Extracts behavioral properties** - Uses LLMs to identify specific behavioral traits in model responses (e.g., "provides step-by-step reasoning", "uses technical jargon", "includes creative examples")

2. **Clusters similar behaviors** - Groups related properties together to identify common patterns (e.g., "Reasoning Transparency", "Communication Style")

3. **Quantifies importance** - Calculates statistical metrics to show which models excel at which behaviors and by how much

4. **Provides insights** - Surfaces the *why* behind model performance differences, not just the *what*

## Key Features

- **🔍 Automatic property extraction** - LLM-powered analysis identifies behavioral patterns without manual coding
- **📊 Clustering** - Groups similar behaviors into meaningful clusters
- **📈 Statistical analysis** - Computes significance testing, confidence intervals, and quality scores
- **Multiple analysis modes** - Single-model analysis or side-by-side comparisons (Arena-style)
- **🏷️ Fixed-taxonomy labeling** - LLM-as-judge with predefined behavioral axes
- **💰 Cost tracking** - Built-in monitoring of LLM API costs
- **📱 Interactive visualizations** - React frontend for exploring results
- **🔧 Flexible pipeline** - Modular architecture supports custom extractors, clusterers, and metrics

## Quick Example

```python
import pandas as pd
from stringsight import explain

# Your data with model responses
df = pd.DataFrame({
    "prompt": ["What is machine learning?", "Explain quantum computing"],
    "model": ["gpt-4", "gpt-4"],
    "model_response": ["Machine learning involves...", "Quantum computing uses..."],
    "score": [{"accuracy": 1, "helpfulness": 4.2}, {"accuracy": 0, "helpfulness": 3.8}]
})

# Extract and cluster behavioral properties
clustered_df, model_stats = explain(
    df,
    method="single_model",
    output_dir="results/"
)

# View results using the React frontend or other visualization tools
```

## Use Cases

### 🏆 Model Evaluation & Comparison
Compare multiple models to understand their behavioral strengths and weaknesses. Identify which models excel at specific tasks (reasoning, creativity, factual accuracy, etc.).

### 🔬 Research & Analysis
Analyze how model behavior changes across:

- Different prompting strategies
- Model versions/checkpoints
- Fine-tuning approaches
- Temperature settings

### 📊 Arena Analysis
Process head-to-head battle data (like Chatbot Arena) to understand *why* users prefer one model over another, not just win rates.

### Task-Specific Evaluation
Focus on behaviors relevant to your domain:

- Call center responses (empathy, clarity, policy adherence)
- Code generation (correctness, efficiency, edge case handling)
- Creative writing (originality, coherence, style)

## How It Works

StringSight uses a 4-stage pipeline:

```
Data Input → Property Extraction → Post-processing → Clustering → Metrics & Analysis
```

1. **Property Extraction** - An LLM analyzes each response and extracts behavioral properties
2. **Post-processing** - Parse and validate extracted properties into structured data
3. **Clustering** - Group similar properties using embeddings and HDBSCAN
4. **Metrics & Analysis** - Calculate per-model statistics, quality scores, and significance tests

## Installation

```bash
# Create conda environment
conda create -n stringsight python=3.11
conda activate stringsight

# Install StringSight
pip install -e ".[full]"

# Set API key
export OPENAI_API_KEY="your-api-key-here"
```

See the [Installation Guide](getting-started/installation.md) for detailed setup instructions.

## Next Steps

- **[Quick Start](getting-started/quick-start.md)** - Get up and running in 5 minutes
- **[User Guide](user-guide/basic-usage.md)** - Learn how to use StringSight effectively
  
- **[Advanced Usage](advanced/custom-pipelines.md)** - Custom pipelines and performance tuning

## Support

- **Documentation**: You're reading it!
- **Issues**: [GitHub Issues](https://github.com/lisabdunlap/stringsight/issues)
- **Source Code**: [GitHub Repository](https://github.com/lisabdunlap/stringsight)

## License

StringSight is released under the MIT License.
