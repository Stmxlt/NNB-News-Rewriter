"""
@filename:evaluation.py
@author:Stmxlt
@time:2025-10-20
"""

import os
import re
import json
import numpy as np
import torch
import traceback
import time
from transformers import AutoTokenizer, AutoModel
from bert_score import score as bertscore_score
from typing import List, Tuple, Optional, Dict, Any
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import ot
from requests.exceptions import ConnectionError as RequestsConnectionError

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Evaluation Module] Initialization completed, using device: {device}")

LOCAL_BERT_PATH = "./local_models/bert-base-uncased"
LOCAL_SMS_MODEL_PATH = "./local_models/all-mpnet-base-v2"
FALLBACK_BERT_MODEL = "microsoft/deberta-large-mnli"
BERTSCORE_BATCH_SIZE = int(os.getenv("BERTSCORE_BATCH_SIZE", "16"))
PARALLEL_WORKERS = 8

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
    base_url=os.getenv("OPENAI_API_BASE", "your-api-link"),
    timeout=30
)

def _safe_strip(x: Any) -> str:
    """
    Safely process a value to ensure it's a non-empty string.
    Args:
        x: Any input value to be processed
    Returns:
        str: Stripped string, empty string if input is None or non-string convertible
    """
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()

_num_regex = re.compile(r"(\d+(?:\.\d+)?)")

def _to_float01_maybe(s: str) -> float:
    """
    Extract a number from text and normalize it to the range [0, 1].
    Args:
        s: Input string containing a number
    Returns:
        float: Normalized number between 0 and 1
    """
    if not s:
        return 0.0
    m = _num_regex.search(str(s))
    if not m:
        return 0.0
    val = float(m.group(1))
    if val > 1:
        val = min(val / 100.0, 1.0)
    return max(0.0, min(1.0, val))

def create_dataset(json_path: str) -> list:
    """
    Load and validate a dataset from a JSON file.
    Args:
        json_path: Path to the JSON file containing the dataset
    Returns:
        list: List of valid dataset items with 'id', 'human_news', and 'abstract' fields
    """
    try:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File does not exist: {json_path}")
        if os.path.getsize(json_path) == 0:
            raise ValueError(f"File is empty: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_dataset = json.load(f)

        valid_dataset = []
        for item in tqdm(raw_dataset, desc="[Data Loading] Processing dataset"):
            human_news = _safe_strip(item.get('human_news', ''))
            abstract = _safe_strip(item.get('abstract', ''))
            if ('id' in item) and human_news and abstract:
                new_item = dict(item)
                new_item['human_news'] = human_news
                new_item['abstract'] = abstract
                valid_dataset.append(new_item)

        print(f"[Data Loading] Successfully loaded dataset: {json_path}, "
              f"original entries: {len(raw_dataset)}, valid entries: {len(valid_dataset)}")
        return valid_dataset
    except Exception as e:
        print(f"[Data Loading] Failed: {str(e)}")
        return []

def preprocess_english_text(text: str) -> str:
    """
    Preprocess English text by removing extra whitespace.
    Args:
        text: Input English text to be preprocessed
    Returns:
        str: Cleaned text with normalized whitespace
    """
    text = _safe_strip(text)
    if not text:
        return ""
    cleaned_text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text

SMS_MODEL = None

def compute_sentence_movers_similarity(candidate: str, reference: str) -> float:
    """
    Compute Sentence Mover's Similarity between two texts.
    Args:
        candidate: Text to be evaluated
        reference: Reference text for comparison
    Returns:
        float: Similarity score between 0 and 1
    """
    global SMS_MODEL
    if SMS_MODEL is None:
        try:
            if not os.path.isdir(LOCAL_SMS_MODEL_PATH):
                raise FileNotFoundError(f"SMS model not found at {LOCAL_SMS_MODEL_PATH}")
            SMS_MODEL = SentenceTransformer(LOCAL_SMS_MODEL_PATH)
        except Exception as e:
            print(f"[SMS Error] Local model load failed: {e}")
            return 0.0

    cand = candidate.strip()
    ref = reference.strip()
    if not cand or not ref:
        return 0.0

    cand_sents = [s.strip() for s in cand.split(".") if s.strip()]
    ref_sents = [s.strip() for s in ref.split(".") if s.strip()]

    if len(cand_sents) == 0 or len(ref_sents) == 0:
        return 0.0

    try:
        cand_vec = SMS_MODEL.encode(cand_sents)
        ref_vec = SMS_MODEL.encode(ref_sents)
    except:
        return 0.0

    cand_vec = np.array(cand_vec)
    ref_vec = np.array(ref_vec)

    cost = 1 - np.dot(cand_vec, ref_vec.T) / \
        (np.linalg.norm(cand_vec, axis=1)[:, None] * np.linalg.norm(ref_vec, axis=1)[None, :])

    a = np.ones(len(cand_sents)) / len(cand_sents)
    b = np.ones(len(ref_sents)) / len(ref_sents)

    try:
        transport_cost = ot.emd2(a, b, cost)
    except:
        return 0.0

    sim = np.exp(-transport_cost)
    return float(max(0.0, min(1.0, sim)))

def compute_gptscore(
        candidate: str,
        reference: str,
        model: str = "deepseek-ai/DeepSeek-V3.2-Exp",
        max_tokens: int = 50,
        temperature: float = 0.0
) -> float:
    """
    Compute similarity score using GPT model.
    Args:
        candidate: Text to be evaluated
        reference: Reference text for comparison
        model: Name of the GPT model to use
        max_tokens: Maximum number of tokens for the model response
        temperature: Sampling temperature for the model
    Returns:
        float: Similarity score between 0 and 1
    """
    cand = candidate.strip()
    ref = reference.strip()
    if not cand or not ref:
        return 0.0

    prompt = f"""
I will give you a human news article and a machine-generated article.
Evaluate how similar the machine-generated article is to the human-written one.
Return ONLY a score between 0 and 1.

Human article:
{ref}

Machine article:
{cand}

Score (0-1):
"""

    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        out = resp.choices[0].message.content.strip()
        return _to_float01_maybe(out)
    except:
        return 0.0

def compute_g_eval(
        candidate: str,
        reference: str,
        dimension: str = "consistency",
        model: str = "deepseek-ai/DeepSeek-V3.2-Exp",
        api_key: Optional[str] = None,
        n_samples: int = 1,
        max_retries: int = 3,
        max_token_length: int = 4000,
        temperature: float = 0.3,
        max_tokens: int = 10000
) -> float:
    """
    Evaluate text quality using G-Eval method.
    Args:
        candidate: Text to be evaluated
        reference: Reference text for comparison
        dimension: Evaluation dimension (consistency, fluency, coherence, relevance)
        model: Name of the model to use
        api_key: Optional API key for the model service
        n_samples: Number of samples to generate for evaluation
        max_retries: Maximum number of retries for failed requests
        max_token_length: Maximum length for text truncation
        temperature: Sampling temperature for the model
        max_tokens: Maximum number of tokens for the model response
    Returns:
        float: Normalized evaluation score between 0 and 1
    """
    cand = (candidate or "").strip()
    ref = (reference or "").strip()
    if not (cand and ref):
        return 0.0

    def truncate_text(text: str, max_len: int) -> str:
        if len(text) > max_len:
            return text[:max_len]
        return text

    cand = truncate_text(cand, max_token_length)
    ref = truncate_text(ref, max_token_length)

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        base_url=os.getenv("OPENAI_API_BASE", "your-api-link"),
    )

    prompt_templates = {
        "consistency": """You will be given a news article and a summary. Your task is to rate the summary's consistency with the article on a scale of 1-5 (1=lowest, 5=highest).

Evaluation Criteria:
Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts.

Evaluation Steps:
1. Read the news article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.

Example:

Source Text:
{{Document}}

Summary:
{{Summary}}

Evaluation Form (scores ONLY):
- Consistency:""",
        "fluency": """You will be given one summary written for a news article.
Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
- 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
- 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
- 3: Good. The summary has few or no errors and is easy to read and follow.

Example:

Summary:
{{Summary}}

Evaluation Form (scores ONLY):
- Fluency (1-3):""",
        "coherence": """You will be given one summary written for a news article.
Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Evaluation Steps:
1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.

Example:

Source Text:
{{Document}}

Summary:
{{Summary}}

Evaluation Form (scores ONLY):
- Coherence:""",
        "relevance": """You will be given one summary written for a news article.
Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Evaluation Steps:
1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.

Example:

Source Text:
{{Document}}

Summary:
{{Summary}}

Output REQUIREMENT: Only return a single number (1-5) with no additional text. For example: 4.5
Your score:"""
    }

    if dimension not in prompt_templates:
        raise ValueError(f"Unsupported evaluation dimension: {dimension}")

    prompt = prompt_templates[dimension].replace("{{Document}}", ref).replace("{{Summary}}", cand)

    def parse_score(output: str) -> float:
        output = output.strip()
        if not output:
            return 0.0
        matched = re.search(r"(\d+\.?\d*)", output)
        if matched:
            try:
                return float(matched.group(1))
            except:
                return 0.0
        return 0.0

    all_scores: List[float] = []
    retry_count = 0
    retry_delay = 1

    while len(all_scores) < n_samples and retry_count < max_retries:
        try:
            remaining = n_samples - len(all_scores)
            current_n = 1

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0,
                n=current_n,
                timeout=30
            )

            batch_scores = [parse_score(choice.message.content) for choice in response.choices]
            all_scores.extend(batch_scores)

            retry_count = 0
            time.sleep(0.5)

        except RequestsConnectionError:
            retry_count += 1
            time.sleep(2 ** retry_count)
        except Exception as e:
            error_msg = str(e)
            retry_count += 1
            print(f"[G-Eval Error] Sampling failed: {error_msg}, retrying {retry_count}th time")
            if "rate limit" in error_msg.lower():
                time.sleep(2 ** retry_count)
            elif "authentication" in error_msg.lower() or "model" in error_msg.lower() and "not found" in error_msg.lower():
                break
            else:
                time.sleep(retry_delay)
                retry_delay *= 2

    valid_scores = [s for s in all_scores if s > 0]
    if not valid_scores:
        return 0.0

    max_score = 5 if dimension in ["consistency", "coherence", "relevance"] else 3
    avg_score = sum(valid_scores) / len(valid_scores)
    normalized_score = avg_score / max_score

    return round(max(0.0, min(1.0, normalized_score)), 4)

def evaluate_text_quality(data: list, iteration: Optional[int] = None) -> Dict[str, Any]:
    """
    Evaluate text quality with multiple metrics, including per-sample details.
    Args:
        data: List of data items to evaluate, each containing 'id', 'human_news', 'gpt_news', and optionally 'pre_gpt_news' or 'machine_news'
        iteration: Optional iteration number for tracking
    Returns:
        Dict[str, Any]: Evaluation results including detailed per-sample metrics and average scores
    """
    metrics = {
        'pre': {
            'bert_score': [], 'sms': [], 'gptscore': [], 'g_eval': []
        },
        'current': {
            'bert_score': [], 'sms': [], 'gptscore': [], 'g_eval': []
        }
    }
    
    sample_details: List[Dict[str, Any]] = []

    batch = {
        'ids': [],
        'pre_raw': [], 'current_raw': [], 'human_raw': []
    }

    total_items = len(data)
    print(f"[Evaluation Progress] Calculating per-sample metrics (total: {total_items})")

    for item in tqdm(data, desc="[Collecting Samples]"):
        item_id = item.get('id', f"unknown_{len(batch['ids'])}")
        human_raw = _safe_strip(item.get('human_news'))
        current_gpt_raw = _safe_strip(item.get('gpt_news'))
        pre_raw = _safe_strip(item.get('pre_gpt_news') or item.get("machine_news"))

        if not (human_raw and current_gpt_raw and pre_raw):
            print(f"[Warning] Skipping invalid sample (id: {item_id}) - missing text content")
            continue

        sample_metrics = {
            'id': item_id,
            'pre': {},
            'current': {}
        }

        pre_sms = compute_sentence_movers_similarity(pre_raw, human_raw)
        current_sms = compute_sentence_movers_similarity(current_gpt_raw, human_raw)
        metrics['pre']['sms'].append(pre_sms)
        metrics['current']['sms'].append(current_sms)
        sample_metrics['pre']['sms'] = pre_sms
        sample_metrics['current']['sms'] = current_sms

        batch['ids'].append(item_id)
        batch['pre_raw'].append(pre_raw)
        batch['current_raw'].append(current_gpt_raw)
        batch['human_raw'].append(human_raw)
        sample_details.append(sample_metrics)

    try:
        def gptscore_worker(cand: str, ref: str, sample_id: str) -> Tuple[str, float]:
            try:
                score = compute_gptscore(cand, ref)
                return (sample_id, score)
            except Exception as e:
                print(f"[GPTScore Worker Error] Sample {sample_id}: {str(e)}")
                return (sample_id, 0.0)

        pre_tasks = list(zip(batch['pre_raw'], batch['human_raw'], batch['ids']))
        cur_tasks = list(zip(batch['current_raw'], batch['human_raw'], batch['ids']))

        pre_gpt_scores = {id: 0.0 for id in batch['ids']}
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
            futures = [ex.submit(gptscore_worker, c, r, idx) for c, r, idx in pre_tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="GPTScore pre"):
                sample_id, score = f.result()
                pre_gpt_scores[sample_id] = score

        cur_gpt_scores = {id: 0.0 for id in batch['ids']}
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
            futures = [ex.submit(gptscore_worker, c, r, idx) for c, r, idx in cur_tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="GPTScore current"):
                sample_id, score = f.result()
                cur_gpt_scores[sample_id] = score

        metrics['pre']['gptscore'] = [pre_gpt_scores[id] for id in batch['ids']]
        metrics['current']['gptscore'] = [cur_gpt_scores[id] for id in batch['ids']]
        
        for sample in sample_details:
            sample_id = sample['id']
            sample['pre']['gptscore'] = pre_gpt_scores[sample_id]
            sample['current']['gptscore'] = cur_gpt_scores[sample_id]

    except Exception as e:
        print("[GPTScore ERROR]", e)
        pre_gpt_list = [0.0] * len(batch['pre_raw'])
        cur_gpt_list = [0.0] * len(batch['pre_raw'])
        metrics['pre']['gptscore'] = pre_gpt_list
        metrics['current']['gptscore'] = cur_gpt_list
        for sample in sample_details:
            sample['pre']['gptscore'] = 0.0
            sample['current']['gptscore'] = 0.0

    try:
        LOCAL_BERT_PATH = "./local_models/bert-base-uncased"
        REQUIRED_FILES = [
            "config.json",
            "vocab.txt",
            "pytorch_model.bin"
        ]
    
        def validate_local_model(path: str) -> bool:
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Local model directory does not exist: {path}")
            missing = []
            for f in REQUIRED_FILES:
                file_path = os.path.join(path, f)
                if not os.path.exists(file_path):
                    if f == "pytorch_model.bin" and os.path.exists(os.path.join(path, "model.safetensors")):
                        continue
                    missing.append(f)
            if missing:
                raise FileNotFoundError(f"Local model missing key files: {missing}")
            return True
    
        validate_local_model(LOCAL_BERT_PATH)
        print(f"[BERTScore] Local model validation passed, path: {LOCAL_BERT_PATH}")
    
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_CACHE"] = LOCAL_BERT_PATH
    
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_BERT_PATH,
            local_files_only=True
        )
        print(f"[BERTScore Debug] Tokenizer loaded successfully from local path")
    
        model = AutoModel.from_pretrained(
            LOCAL_BERT_PATH,
            local_files_only=True
        )
        print(f"[BERTScore Debug] Model weights loaded successfully from local path")
    
        def truncate_list(texts: List[str]) -> List[str]:
            out = []
            for t in texts:
                ids = tokenizer(t, truncation=True, max_length=512, return_tensors="pt")
                out.append(tokenizer.decode(ids.input_ids[0], skip_special_tokens=True))
            return out
        pre_t = truncate_list(batch['pre_raw'])
        cur_t = truncate_list(batch['current_raw'])
        hum_t = truncate_list(batch['human_raw'])
    
        num_layers = 12
        print(f"[BERTScore Debug] Starting calculation, model layers: {num_layers}, device: {device}")
        pre_bs_p, pre_bs_r, pre_bs_f = bertscore_score(
            pre_t, hum_t,
            lang="en",
            model_type=LOCAL_BERT_PATH,
            num_layers=num_layers,
            device=device
        )
        cur_bs_p, cur_bs_r, cur_bs_f = bertscore_score(
            cur_t, hum_t,
            lang="en",
            model_type=LOCAL_BERT_PATH,
            num_layers=num_layers,
            device=device
        )
    
        pre_bs_list = pre_bs_f.cpu().tolist()
        cur_bs_list = cur_bs_f.cpu().tolist()
        metrics['pre']['bert_score'] = pre_bs_list
        metrics['current']['bert_score'] = cur_bs_list
        
        for i, sample in enumerate(sample_details):
            sample['pre']['bert_score'] = pre_bs_list[i]
            sample['current']['bert_score'] = cur_bs_list[i]
    
    except Exception as e:
        print(f"[BERTScore ERROR] Detailed error: {str(e)}")
        print(f"[BERTScore ERROR] Stack trace: {traceback.format_exc()}")
        pre_bs_list = [0.0] * len(batch['pre_raw'])
        cur_bs_list = [0.0] * len(batch['pre_raw'])
        metrics['pre']['bert_score'] = pre_bs_list
        metrics['current']['bert_score'] = cur_bs_list
        for sample in sample_details:
            sample['pre']['bert_score'] = 0.0
            sample['current']['bert_score'] = 0.0
            print(f"[Sample {sample['id']}] BERTScore - pre: 0.0000, current: 0.0000 (error)")

    try:
        def worker(cand: str, ref: str, sample_id: str) -> Tuple[str, float]:
            try:
                score = compute_g_eval(cand, ref)
                return (sample_id, score)
            except Exception as e:
                print(f"[G-Eval Worker Error] Sample {sample_id}: {str(e)}")
                return (sample_id, 0.0)

        pre_tasks = list(zip(batch['pre_raw'], batch['human_raw'], batch['ids']))
        cur_tasks = list(zip(batch['current_raw'], batch['human_raw'], batch['ids']))

        pre_geval_scores = {id: 0.0 for id in batch['ids']}
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
            futures = [ex.submit(worker, c, r, idx) for c, r, idx in pre_tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="G-Eval pre"):
                sample_id, score = f.result()
                pre_geval_scores[sample_id] = score

        cur_geval_scores = {id: 0.0 for id in batch['ids']}
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
            futures = [ex.submit(worker, c, r, idx) for c, r, idx in cur_tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="G-Eval current"):
                sample_id, score = f.result()
                cur_geval_scores[sample_id] = score

        metrics['pre']['g_eval'] = [pre_geval_scores[id] for id in batch['ids']]
        metrics['current']['g_eval'] = [cur_geval_scores[id] for id in batch['ids']]
        
        for sample in sample_details:
            sample_id = sample['id']
            sample['pre']['g_eval'] = pre_geval_scores[sample_id]
            sample['current']['g_eval'] = cur_geval_scores[sample_id]

    except Exception as e:
        print("[G-Eval ERROR]", e)
        pre_geval_list = [0.0] * len(batch['pre_raw'])
        cur_geval_list = [0.0] * len(batch['pre_raw'])
        metrics['pre']['g_eval'] = pre_geval_list
        metrics['current']['g_eval'] = cur_geval_list
        for sample in sample_details:
            sample['pre']['g_eval'] = 0.0
            sample['current']['g_eval'] = 0.0
            print(f"[Sample {sample['id']}] G-Eval - pre: 0.0000, current: 0.0000 (error)")

    def mean(xs: List[float]) -> float:
        xs = [x for x in xs if isinstance(x, (int, float))]
        return float(np.mean(xs)) if xs else 0.0

    pre_avg = {
        "bert_score": mean(metrics['pre']['bert_score']),
        "sms": mean(metrics['pre']['sms']),
        "gptscore": mean(metrics['pre']['gptscore']),
        "g_eval": mean(metrics['pre']['g_eval'])
    }

    cur_avg = {
        "bert_score": mean(metrics['current']['bert_score']),
        "sms": mean(metrics['current']['sms']),
        "gptscore": mean(metrics['current']['gptscore']),
        "g_eval": mean(metrics['current']['g_eval'])
    }

    return {
        "iteration": iteration,
        "detailed": {
            "samples": sample_details,
            "metrics": metrics
        },
        "pre_average": pre_avg,
        "current_average": cur_avg
    }
