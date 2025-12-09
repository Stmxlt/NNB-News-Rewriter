# NNB-News-Rewriter

A news optimizing and rewriting tool for the New News Broadcasting (NNB) system, designed to enhance news articles generated from summaries through iterative refinement and quality evaluation.

## Overview

This project provides a comprehensive pipeline for generating, evaluating, and improving news articles based on given summaries. It leverages natural language processing techniques and large language models (LLMs) to optimize news content, ensuring it meets journalistic standards in terms of clarity, coherence, and factual accuracy.

## Key Features

- **News Generation**: Generate initial news articles from summaries using state-of-the-art LLMs.
- **Iterative Rewriting**: Refine news content through iterative optimization based on evaluation feedback.
- **Quality Evaluation**: Assess news quality using multiple metrics (BLEU, METEOR, ROUGE-L, BERTScore, G-Eval).
- **Similarity Matching**: Identify similar news articles to use as reference for improvement.
- **Visualization**: Generate visual reports of evaluation metrics to track performance across iterations.

## File Structure

- `prompt.py`: Handles prompt generation for LLM interactions, including evaluation criteria and rewriting guidelines.
- `evaluation.py`: Implements various evaluation metrics to assess news article quality.
- `Rewriter.py`: Core module for iterative news rewriting, integrating LLM calls and feedback loops.
- `dataset/dataset.py`: Processes datasets to generate initial machine-written news from summaries.
- `visualization.py`: Generates visualizations of evaluation metrics to monitor performance.
- `dataset/cnn_dailymail.json`: Example dataset (CNN/DailyMail) used for training and testing.

## Dependencies

- Python 3.8+
- `tiktoken` for tokenization
- `sentence-transformers` for similarity calculations
- `scikit-learn` for cosine similarity
- `transformers` and `torch` for BERTScore evaluation
- `rouge-score` and `sacrebleu` for n-gram based metrics
- `openai` for LLM API interactions
- `matplotlib` for visualization
- `nltk` for METEOR score calculation

Install dependencies via pip:
```bash
pip install tiktoken sentence-transformers scikit-learn transformers torch rouge-score sacrebleu openai matplotlib nltk
```
