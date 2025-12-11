# NNB-News-Rewriter

A news optimizing and rewriting tool for the New News Broadcasting (NNB) system, designed to enhance news articles generated from summaries through iterative refinement and quality evaluation.

## Overview

This project provides a comprehensive pipeline for generating, evaluating, and improving news articles based on given summaries. It leverages natural language processing techniques and large language models (LLMs) to optimize news content, ensuring it meets journalistic standards in terms of clarity, coherence, and factual accuracy.

## Key Features

- **News Generation**: Generate initial news articles from summaries using state-of-the-art LLMs.
- **Iterative Rewriting**: Refine news content through iterative optimization based on evaluation feedback.
- **Quality Evaluation**: Assess news quality using multiple metrics (BERTScore, SMS, GPTScore, G-Eval).
- **Similarity Matching**: Identify similar news articles to use as reference for improvement.
- **Visualization**: Generate visual reports of evaluation metrics to track performance across iterations.

## File Structure

```plaintext
NNB-News-Rewriter/
├── Rewriter.py
├── utils/
│   ├── __init__.py
│   ├── prompt.py
│   ├── evaluation.py
│   └── visualization.py
├── dataset/
│   └── cnn_dailymail.json
├── result/
│   ├── cnn_dailymail_updated.json
│   └── evaluation_result.json
├── local_models/
├── requirements.txt
└── README.md
```

- `Rewriter.py`: Core module for iterative news rewriting, integrating LLM calls and feedback loops.
- `utils/evaluation.py`: Implements various evaluation metrics to assess news article quality.
- `utils/prompt.py`: Handles prompt generation for LLM interactions, including evaluation criteria and rewriting guidelines.
- `utils/visualization.py`: Generates visualizations of evaluation metrics to monitor performance.
- `dataset/cnn_dailymail.json`: Example dataset (CNN/DailyMail) used for training and testing.

## Dependencies

Install dependencies via pip:
```bash
pip install -r requirements.txt
```

## Configuration

Set environment variables for API access in python files Rewriter.py, evaluation.py, dataset.py:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="your-api-base-url"
```

## Preparing Models

You can download these models from the [Hugging Face Hub](https://huggingface.co/) or [ModelScope](https://modelscope.cn/home) and place them in the corresponding directories, or use the `sentence-transformers` library to cache them locally:

```python
from sentence_transformers import SentenceTransformer

# Download and save models to local_models directory
models = [
    "bert-base-uncased",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2"
]

for model_name in models:
    model = SentenceTransformer(model_name)
    model.save(f"local_models/{model_name}")
```

## Usage
### 1. Dataset Preparation

We provide a 1000 news articles dataset cnn_dailymail.json. You can utilize it directly.
You can obtain news from [huggingface](https://huggingface.co/datasets/abisee/cnn_dailymail) for more news articles.

### 2. Run Iterative Rewriting
Run the Rewriter function in Rewriter.py:
```python
python Rewriter.py
```
The Rewriter functions iteratively improve news articles using similar news references, quality feedback, and LLMs, enhancing quality while staying true to the original summary.

## Notes
- The system uses both local models (e.g., all-MiniLM-L6-v2 for similarity) and external LLM APIs for generation/evaluation.
- Evaluation metrics include both automated scores (BERTScore, SMS) and LLM-based assessments (GPTScore, G-Eval).
- Iterative refinement leverages similar news articles as references to improve content quality.

## Contributors
- [lizhizhongpingguo](https://github.com/lizhizhongpingguo) - Write the top5_similar function for prompt.py to extracting similar news IDs
- [Yichan521](https://github.com/Yichan521) - Optimize prompt logic for prompt.py to achieve a better rewriting effect
