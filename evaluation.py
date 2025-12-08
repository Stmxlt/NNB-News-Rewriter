"""
@filename:Rewriter.py
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
from nltk.translate.meteor_score import single_meteor_score
from transformers import AutoTokenizer
from rouge_score import rouge_scorer as _rouge_scorer
from bert_score import score as bertscore_score
from sacrebleu import BLEU
from typing import List, Tuple, Optional
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sacrebleu = BLEU(tokenize='13a', effective_order=True)
rouge_scorer = _rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Evaluation Module] Initialization completed, using device: {device}")

LOCAL_BERT_PATH = "./local_models/bert-base-uncased"
FALLBACK_BERT_MODEL = "microsoft/deberta-large-mnli"
BERTSCORE_BATCH_SIZE = int(os.getenv("BERTSCORE_BATCH_SIZE", "16"))
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
    base_url=os.getenv("OPENAI_API_BASE", "your-api-link"),
    timeout=30
)


def _safe_strip(x) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()


_num_regex = re.compile(r"(\d+(?:\.\d+)?)")


def _to_float01_maybe(s: str) -> float:
    if s is None:
        return 0.0
    m = _num_regex.search(str(s))
    if not m:
        return 0.0
    val = float(m.group(1))
    if val > 1.0:
        val = val / 100.0 if val <= 100.0 else 1.0
    return max(0.0, min(1.0, val))


def _pick_bert_model() -> Tuple[str, bool]:
    if os.path.isdir(LOCAL_BERT_PATH):
        expected_files = ["config.json", "pytorch_model.bin", "model.safetensors", "vocab.txt", "tokenizer.json"]
        if any(os.path.exists(os.path.join(LOCAL_BERT_PATH, f)) for f in expected_files):
            print(f"[BERTScore] Using local model at: {LOCAL_BERT_PATH}")
            return LOCAL_BERT_PATH, True
    print(f"[BERTScore] Local path not available, fallback to: {FALLBACK_BERT_MODEL}")
    return FALLBACK_BERT_MODEL, False


def create_dataset(json_path: str) -> list:
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
    text = _safe_strip(text)
    if not text:
        return ""
    cleaned_text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text


def evaluate_text_quality(data: list, iteration: int = None) -> dict:
    # -------------------- Main Eval --------------------
    metrics = {
        'pre': {'meteor': [], 'rouge_l': [], 'bert_score': [], 'g_eval': []},
        'current': {'meteor': [], 'rouge_l': [], 'bert_score': [], 'g_eval': []}
    }
    batch = {
        'pre_raw': [], 'current_raw': [], 'human_raw': [],
        'pre_seg': [], 'current_seg': [], 'human_seg': []
    }

    hypotheses_pre = []
    hypotheses_current = []
    references_human = []

    total_items = len(data)
    print(f"[Evaluation Progress] Calculating per-sample ngram metrics (total items: {total_items})")

    for idx, item in enumerate(tqdm(data, desc="[Evaluation] Processing ngram metrics")):
        item_id = str(item.get('id', 'unknown ID'))

        human_raw = _safe_strip(item.get('human_news', ''))

        current_gpt_raw = _safe_strip(item.get('gpt_news', ''))
        pre_gpt_news = _safe_strip(item.get('pre_gpt_news', ''))
        pre_raw = pre_gpt_news if pre_gpt_news else _safe_strip(item.get('machine_news', ''))

        skip_reason = []
        if not human_raw:
            skip_reason.append("human_news is empty")
        if not current_gpt_raw:
            skip_reason.append("gpt_news is empty")
        if not pre_raw:
            skip_reason.append("both pre_gpt_news and machine_news are empty")
        if skip_reason:
            print(f"\n[Evaluation Warning] ID={item_id} ({'、'.join(skip_reason)}), skipping this sample")
            continue

        human_seg = preprocess_english_text(human_raw)
        current_seg = preprocess_english_text(current_gpt_raw)
        pre_seg = preprocess_english_text(pre_raw)

        hypotheses_pre.append(pre_seg)
        hypotheses_current.append(current_seg)
        references_human.append(human_seg)

        batch['pre_raw'].append(pre_raw.replace('\n', ' ').strip())
        batch['current_raw'].append(current_gpt_raw.replace('\n', ' ').strip())
        batch['human_raw'].append(human_raw.replace('\n', ' ').strip())
        batch['pre_seg'].append(pre_seg)
        batch['current_seg'].append(current_seg)
        batch['human_seg'].append(human_seg)

        pre_meteor = single_meteor_score(reference=human_seg.split(), hypothesis=pre_seg.split(), alpha=0.9, beta=3.0,
                                         gamma=0.5)
        current_meteor = single_meteor_score(reference=human_seg.split(), hypothesis=current_seg.split())
        pre_rouge = rouge_scorer.score(human_seg, pre_seg)['rougeL'].fmeasure
        current_rouge = rouge_scorer.score(human_seg, current_seg)['rougeL'].fmeasure

        metrics['pre']['meteor'].append(pre_meteor)
        metrics['pre']['rouge_l'].append(pre_rouge)
        metrics['current']['meteor'].append(current_meteor)
        metrics['current']['rouge_l'].append(current_rouge)

    valid_sample_count = len(references_human)

    if valid_sample_count == 0:
        print(f"[Evaluation Warning] No valid samples, all metrics return 0")
        zero_avg = {k: 0.0 for k in metrics['pre'].keys()}
        return {'iteration': iteration, 'detailed': metrics,
                'pre_average': zero_avg, 'current_average': zero_avg}

    bleu_pre_score = sacrebleu.corpus_score(hypotheses_pre, [references_human]).score / 100.0
    bleu_current_score = sacrebleu.corpus_score(hypotheses_current, [references_human]).score / 100.0
    print(f"[Evaluation] Corpus BLEU (pre): {bleu_pre_score:.4f}")
    print(f"[Evaluation] Corpus BLEU (current): {bleu_current_score:.4f}")

    print(f"[Evaluation Progress] Starting batch calculation of neural metrics (valid samples: {valid_sample_count})")

    # -------- BERTScore --------
    try:
        model_name, is_local = _pick_bert_model()
        print(f"[BERTScore] Using model: {model_name} (local={is_local})")
        use_baseline = (not is_local)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def truncate_texts(texts, max_length=512):
            truncated = []
            for text in texts:
                encoded = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    return_tensors='pt',
                    padding=False
                )
                truncated_text = tokenizer.decode(encoded.input_ids[0], skip_special_tokens=True)
                truncated.append(truncated_text)
            return truncated

        pre_raw_truncated = truncate_texts(batch['pre_raw'])
        current_raw_truncated = truncate_texts(batch['current_raw'])
        human_raw_truncated = truncate_texts(batch['human_raw'])

        print("[BERTScore] Calculating scores...")
        pre_bs_p, pre_bs_r, pre_bs_f = bertscore_score(
            cands=pre_raw_truncated,
            refs=human_raw_truncated,
            lang="en",
            model_type=model_name,
            device=device,
            rescale_with_baseline=use_baseline,
            batch_size=BERTSCORE_BATCH_SIZE,
            verbose=False,
            num_layers=10
        )
        cur_bs_p, cur_bs_r, cur_bs_f = bertscore_score(
            cands=current_raw_truncated,
            refs=human_raw_truncated,
            lang="en",
            model_type=model_name,
            device=device,
            rescale_with_baseline=use_baseline,
            batch_size=BERTSCORE_BATCH_SIZE,
            verbose=False,
            num_layers=10
        )
        metrics['pre']['bert_score'] = pre_bs_f.detach().cpu().tolist()[:valid_sample_count]
        metrics['current']['bert_score'] = cur_bs_f.detach().cpu().tolist()[:valid_sample_count]
        print("[BERTScore] Calculation completed")
    except Exception as e:
        print(f"[BERTScore Error] Full error message:\n{traceback.format_exc()}")
        metrics['pre']['bert_score'] = [0.0] * valid_sample_count
        metrics['current']['bert_score'] = [0.0] * valid_sample_count

    # -------- G-Eval --------
    try:
        print(f"[Evaluation Progress] Starting parallel G-Eval calculation (number of samples: {valid_sample_count})")
        max_workers = int(os.getenv("G_EVAL_MAX_WORKERS", "8"))
        print(f"[Evaluation Progress] Number of parallel threads: {max_workers}")

        def parallel_worker(candidate: str, reference: str) -> float:
            try:
                return compute_g_eval(candidate, reference)
            except Exception as e:
                print(f"[G-Eval Worker Error] Calculation failed: {str(e)}")
                return 0.0

        pre_tasks = list(zip(batch['pre_raw'], batch['human_raw']))
        pre_g_eval_scores = [0.0] * len(pre_tasks)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(parallel_worker, cand, ref): idx
                for idx, (cand, ref) in enumerate(pre_tasks)
            }

            for future in tqdm(
                    as_completed(future_to_idx),
                    total=len(future_to_idx),
                    desc="[Evaluation] G-Eval (pre)"
            ):
                idx = future_to_idx[future]
                try:
                    pre_g_eval_scores[idx] = future.result()
                except Exception as e:
                    print(f"[G-Eval Result Error] task {idx} result fetch failed: {str(e)}")
                    pre_g_eval_scores[idx] = 0.0
        metrics['pre']['g_eval'] = pre_g_eval_scores

        current_tasks = list(zip(batch['current_raw'], batch['human_raw']))
        current_g_eval_scores = [0.0] * len(current_tasks)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(parallel_worker, cand, ref): idx
                for idx, (cand, ref) in enumerate(current_tasks)
            }

            for future in tqdm(
                    as_completed(future_to_idx),
                    total=len(future_to_idx),
                    desc="[Evaluation] G-Eval (current)"
            ):
                idx = future_to_idx[future]
                try:
                    current_g_eval_scores[idx] = future.result()
                except Exception as e:
                    print(f"[G-Eval Result Error] task {idx} result fetch failed: {str(e)}")
                    current_g_eval_scores[idx] = 0.0
        metrics['current']['g_eval'] = current_g_eval_scores

        print(f"[Evaluation Module] G-Eval computing success")
    except Exception as e:
        print(f"[Evaluation Warning] G-Eval computing failed (reason: {str(e)}), using 0 as placeholder")
        metrics['pre']['g_eval'] = [0.0] * valid_sample_count
        metrics['current']['g_eval'] = [0.0] * valid_sample_count

    # -------- Averages --------
    def mean_score(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    pre_average = {
        'bleu': bleu_pre_score,
        'meteor': mean_score(metrics['pre']['meteor']),
        'rouge_l': mean_score(metrics['pre']['rouge_l']),
        'bert_score': mean_score(metrics['pre']['bert_score']),
        'g_eval': mean_score(metrics['pre']['g_eval'])
    }
    current_average = {
        'bleu': bleu_current_score,
        'meteor': mean_score(metrics['current']['meteor']),
        'rouge_l': mean_score(metrics['current']['rouge_l']),
        'bert_score': mean_score(metrics['current']['bert_score']),
        'g_eval': mean_score(metrics['current']['g_eval'])
    }

    print("\n" + "=" * 180)
    print(f"[Evaluation Result] Iteration count: {iteration if iteration is not None else 'unspecified'}")
    print(f"[Evaluation Result] Previous round/machine-generated vs human original text (average metrics):")
    print(
        f"          BLEU: {pre_average['bleu']:.4f} | METEOR: {pre_average['meteor']:.4f} | ROUGE-L: {pre_average['rouge_l']:.4f} | "
        f"BERTScore: {pre_average['bert_score']:.4f} | G-Eval: {pre_average['g_eval']:.4f}")
    print(f"[Evaluation Result] Current GPT-generated vs human original text (average metrics):")
    print(
        f"          BLEU: {current_average['bleu']:.4f} | METEOR: {current_average['meteor']:.4f} | ROUGE-L: {current_average['rouge_l']:.4f} | "
        f"BERTScore: {current_average['bert_score']:.4f} | G-Eval: {current_average['g_eval']:.4f}")
    print("=" * 180 + "\n")

    final_metrics = metrics
    # final_metrics['pre']['bleu'] = [bleu_pre_score] * valid_sample_count
    # final_metrics['current']['bleu'] = [bleu_current_score] * valid_sample_count

    return {
        'iteration': iteration,
        'detailed': final_metrics,
        'pre_average': pre_average,
        'current_average': current_average
    }


def compute_g_eval(
        candidate: str,
        reference: str,
        dimension: str = "consistency",
        model: str = "deepseek-ai/DeepSeek-V3.2-Exp",
        api_key: Optional[str] = None,
        n_samples: int = 5,
        max_retries: int = 3,
        max_token_length: int = 4000,
        temperature: float = 0.3,
        max_tokens: int = 10000
) -> float:
    cand = (candidate or "").strip()
    ref = (reference or "").strip()
    if not (cand and ref):
        return 0.0

    def truncate_text(text, max_len):
        if len(text) > max_len:
            return text[:max_len]
        return text

    cand = truncate_text(cand, max_token_length)
    ref = truncate_text(ref, max_token_length)

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        base_url=os.getenv("OPENAI_API_BASE", "your-api-link")
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

    from requests.exceptions import ConnectionError as RequestsConnectionError

    while len(all_scores) < n_samples and retry_count < max_retries:
        try:
            remaining = n_samples - len(all_scores)
            current_n = min(remaining, 5)

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0,
                n=current_n
            )

            batch_scores = [parse_score(choice.message.content) for choice in response.choices]
            all_scores.extend(batch_scores)

            retry_count = 0
            time.sleep(0.5)

        except RequestsConnectionError:
            retry_count += 1
            sleep_time = 2 ** retry_count
            time.sleep(sleep_time)

        except Exception as e:
            error_msg = str(e)
            retry_count += 1
            if "rate limit" in error_msg.lower():
                sleep_time = 2 ** retry_count
                time.sleep(sleep_time)
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