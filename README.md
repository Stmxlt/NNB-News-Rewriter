# NNB-News-Rewriter

<p align="center">
<img src="icon.png" width="200"/>
</p>

This project is a news optimizing and rewriting tool for the New News Broadcasting (NNB) system, designed to enhance news articles generated from summaries through iterative refinement and quality evaluation.

## Overview

This project provides an end-to-end pipeline for generating, evaluating, and improving news articles based on given summaries. It uses:
- Local embedding models for similarity & SMS evaluation
- LLM APIs for evaluation feedback (critique) and iterative rewriting (generation)
- Multi-metric evaluation and visualization across iterations
- Per-news metric tracking to enable iteration-aware, sample-specific improvement guidance

## Key Features

- Similarity Matching (Top-5): Retrieve 5 most similar summaries as exemplars using sentence-transformers embeddings.
- Iterative Rewriting: Improve machine news in multiple rounds using exemplar style + metric-driven guidance.
- Quality Evaluation: Evaluate with BERTScore, SMS (Sentence Movers Similarity), GPTScore, and G-Eval dimensions.
- Per-News Tracking: Store per-sample metrics by news_id across iterations for targeted improvement suggestions.
- One-shot News Generation: Generate a complete news article directly from a user-provided summary using high-quality exemplar news.

## File Structure

```plaintext
NNB-News-Rewriter/
├── Rewriter.py
├── Generator.py
├── utils/
│   ├── __init__.py
│   ├── prompt.py
│   ├── evaluation.py
│   ├── per_news_evaluation.py
│   └── visualization.py
├── dataset/
│   ├── cnn_dailymail_debug.json
│   ├── rewrited_cnn_dailymail.json
│   └── dataset.py
├── result/
│   ├── evaluation_result.json
│   ├── per_news_metrics.json
│   ├── metrics_visualization.png
│   └── news/
├── local_models/
├── user_news_abstract.txt
├── requirements.txt
└── README.md
```

## Dependencies

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

## Configuration

### Required Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="your-api-link"
```

### Optional Environment Variables

```bash
export EVAL_MODEL="deepseek-ai/DeepSeek-V3.2-Exp"
export BERTSCORE_BATCH_SIZE=16
```

## Preparing Local Models

You can download these models from the [Hugging Face Hub](https://huggingface.co/) or [ModelScope](https://modelscope.cn/home) and place them in the corresponding directories, or use the `sentence-transformers` library to cache them locally:

For Hugging Face:
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
For ModelScope:
```bash
pip install modelscope
modelscope download --model google-bert/bert-base-uncased
modelscope download --model sentence-transformers/all-MiniLM-L6-v2
modelscope download --model sentence-transformers/all-mpnet-base-v2
```

## Dataset Preparation

We provide a 1000 news articles dataset cnn_dailymail.json. You can utilize it directly.

You can obtain news from [Hugging Face Hub](https://huggingface.co/datasets/abisee/cnn_dailymail) for more news articles. After convert parquet files to json files, you need to run dataset/dataset.py to add machine_news content.

All dataset files share the same schema:

```json
{
  "id": "...",
  "abstract": "...",
  "human_news": "...",
  "machine_news": "...",
  "evaluation": "",
  "gpt_news": "",
  "pre_gpt_news": "",
  "similar": []
}
```

- `cnn_dailymail.json`  
  Raw baseline dataset. Before each run, work-related fields are cleared.

## Generator

`Generator.py` provides a one-shot news generation function. Given a single user summary `user_news_abstract.txt`, it retrieves similar summaries from the working dataset, uses their high-quality news as exemplars, and generates a complete news article strictly grounded in the input summary.

## Usage

Run iterative rewriting:

```python
python Rewriter.py
```

Run one-shot news generation:

```python
python Generator.py
```

## Outputs

- `result/evaluation_result.json`  
  Average metrics per iteration.

- `result/per_news_metrics.json`  
  Per-news metrics across iterations.

- `result/metrics_visualization.png`  
  Metric trend plots.

- `result/news/news_{id}.txt`  
  Exported cleaned or generated news text files.

- `result/user_generated_news.txt`  
  Exported generated news text files based on `user_news_abstract.txt`.

## Notes

- The summary is the ONLY factual source.
- No new entities, numbers, dates, or quotes are allowed.
- Iterative rewriting and one-shot generation share the same dataset schema.
- Training, rewriting, and generation always operate on `rewrited_cnn_dailymail.json`.

## Contributors
- [lizhizhongpingguo](https://github.com/lizhizhongpingguo) - Write the top5_similar function for prompt.py to extracting similar news IDs
- [Yichan521](https://github.com/Yichan521) - Optimize prompt logic for prompt.py to achieve a better rewriting effect
