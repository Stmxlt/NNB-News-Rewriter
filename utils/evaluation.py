"""
@filename:evaluation.py
@author:Stmxlt
@time:2026-01-26
"""

import os
import re
import json
import time
import traceback
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from bert_score import score as bertscore_score
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from requests.exceptions import ConnectionError as RequestsConnectionError
import ot

# -----------------------------
# Global config
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Evaluation Module] Initialization completed, using device: {device}")

LOCAL_BERT_PATH = "./local_models/bert-base-uncased"
LOCAL_SMS_MODEL_PATH = "./local_models/all-mpnet-base-v2"

BERTSCORE_BATCH_SIZE = int(os.getenv("BERTSCORE_BATCH_SIZE", "16"))

# Required by user:
PARALLEL_WORKERS = 8  # keep as 8

# Default eval model (you can override by passing model=...)
DEFAULT_EVAL_MODEL = os.getenv("EVAL_MODEL", "deepseek-ai/DeepSeek-V3.2-Exp")

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
    base_url=os.getenv("OPENAI_API_BASE", "your-api-link"),
    timeout=30,
)

# -----------------------------
# Utils
# -----------------------------
def _safe_strip(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()

_num_regex = re.compile(r"(\d+(?:\.\d+)?)")

def _to_float01_maybe(s: str) -> float:
    if not s:
        return 0.0
    m = _num_regex.search(str(s))
    if not m:
        return 0.0
    val = float(m.group(1))
    if val > 1:
        val = min(val / 100.0, 1.0)
    return max(0.0, min(1.0, val))

def preprocess_english_text(text: str) -> str:
    text = _safe_strip(text)
    if not text:
        return ""
    cleaned_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text

# -----------------------------
# Dataset
# -----------------------------
def create_dataset(json_path: str) -> list:
    """
    Load dataset JSON and keep only minimally valid entries.

    Required:
      - id
      - human_news (non-empty)
      - summary (non-empty)

    Optional but recommended for evaluation:
      - gpt_news / pre_gpt_news / machine_news (at least one non-empty)
    """
    try:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File does not exist: {json_path}")
        if os.path.getsize(json_path) == 0:
            raise ValueError(f"File is empty: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            raw_dataset = json.load(f)

        valid_dataset = []
        kept = 0
        dropped = 0
        missing_candidates = 0

        for item in tqdm(raw_dataset, desc="[Data Loading] Processing dataset"):
            human_news = _safe_strip(item.get("human_news", ""))
            summary = _safe_strip(item.get("summary", ""))
            if "id" not in item or not human_news or not summary:
                dropped += 1
                continue

            gpt_news = _safe_strip(item.get("gpt_news", ""))
            pre_gpt_news = _safe_strip(item.get("pre_gpt_news", ""))
            machine_news = _safe_strip(item.get("machine_news", ""))

            if not (gpt_news or pre_gpt_news or machine_news):
                missing_candidates += 1

            new_item = dict(item)
            new_item["human_news"] = human_news
            new_item["summary"] = summary
            new_item["gpt_news"] = gpt_news
            new_item["pre_gpt_news"] = pre_gpt_news
            if "machine_news" in item:
                new_item["machine_news"] = machine_news

            valid_dataset.append(new_item)
            kept += 1

        print(
            f"[Data Loading] Loaded dataset: {json_path}, original: {len(raw_dataset)}, kept: {kept}, "
            f"dropped: {dropped}, missing_candidates: {missing_candidates}"
        )
        return valid_dataset
    except Exception as e:
        print(f"[Data Loading] Failed: {str(e)}")
        return []

# -----------------------------
# SMS
# -----------------------------
SMS_MODEL = None

def compute_sentence_movers_similarity(candidate: str, reference: str) -> float:
    global SMS_MODEL
    if SMS_MODEL is None:
        try:
            if not os.path.isdir(LOCAL_SMS_MODEL_PATH):
                raise FileNotFoundError(f"SMS model not found at {LOCAL_SMS_MODEL_PATH}")
            SMS_MODEL = SentenceTransformer(LOCAL_SMS_MODEL_PATH)
        except Exception as e:
            print(f"[SMS Error] Local model load failed: {e}")
            return 0.0

    cand = (candidate or "").strip()
    ref = (reference or "").strip()
    if not cand or not ref:
        return 0.0

    cand_sents = [s.strip() for s in cand.split(".") if s.strip()]
    ref_sents = [s.strip() for s in ref.split(".") if s.strip()]
    if len(cand_sents) == 0 or len(ref_sents) == 0:
        return 0.0

    try:
        cand_vec = SMS_MODEL.encode(cand_sents)
        ref_vec = SMS_MODEL.encode(ref_sents)
    except Exception:
        return 0.0

    cand_vec = np.array(cand_vec)
    ref_vec = np.array(ref_vec)

    denom = (np.linalg.norm(cand_vec, axis=1)[:, None] * np.linalg.norm(ref_vec, axis=1)[None, :])
    denom = np.clip(denom, 1e-9, None)
    cost = 1 - np.dot(cand_vec, ref_vec.T) / denom

    a = np.ones(len(cand_sents)) / len(cand_sents)
    b = np.ones(len(ref_sents)) / len(ref_sents)

    try:
        transport_cost = ot.emd2(a, b, cost)
    except Exception:
        return 0.0

    sim = np.exp(-transport_cost)
    return float(max(0.0, min(1.0, sim)))

# -----------------------------
# GPTScore (kept)
# -----------------------------
def compute_gptscore(
    candidate: str,
    reference: str,
    model: str = DEFAULT_EVAL_MODEL,
    max_tokens: int = 50,
    temperature: float = 0.0,
) -> float:
    cand = (candidate or "").strip()
    ref = (reference or "").strip()
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
""".strip()

    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        out = (resp.choices[0].message.content or "").strip()
        return _to_float01_maybe(out)
    except Exception:
        return 0.0

# -----------------------------
# Scheme A: single-call JSON eval (n_samples = 3 by repeating calls)
# -----------------------------
_json_re = re.compile(r"\{[\s\S]*\}")

def compute_llm_eval_a_once(
    candidate: str,
    reference: str,
    model: str = DEFAULT_EVAL_MODEL,
    max_token_length: int = 4000,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> Dict[str, float]:
    """
    One API call. Returns normalized [0,1] scores:
      - consistency (1-5 -> 0-1)
      - coverage    (1-5 -> 0-1)
      - quality     (1-5 -> 0-1)
    """
    cand = (candidate or "").strip()
    ref = (reference or "").strip()
    if not (cand and ref):
        return {"consistency": 0.0, "coverage": 0.0, "quality": 0.0}

    def truncate_text(text: str, max_len: int) -> str:
        return text[:max_len] if len(text) > max_len else text

    cand = truncate_text(cand, max_token_length)
    ref = truncate_text(ref, max_token_length)

    prompt = f"""
Evaluate the machine-written news article compared to the human-written one.

Please score the following aspects on a scale of 1–5:

1. Factual Consistency:
   Are all statements supported by the human-written news?

2. Content Coverage:
   Does the article cover the key information from the human-written news?

3. Overall News Quality:
   Considering structure, coherence, and language fluency,
   how well-written is this as a news article?

Return ONLY in JSON:
{{
  "consistency": number,
  "coverage": number,
  "quality": number
}}

Human-written news:
{ref}

Machine-written news:
{cand}
""".strip()

    def parse_json_scores(text: str) -> Dict[str, float]:
        if not text:
            return {"consistency": 0.0, "coverage": 0.0, "quality": 0.0}
        s = text.strip()

        try:
            obj = json.loads(s)
        except Exception:
            m = _json_re.search(s)
            if not m:
                return {"consistency": 0.0, "coverage": 0.0, "quality": 0.0}
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return {"consistency": 0.0, "coverage": 0.0, "quality": 0.0}

        def to_01(v: Any) -> float:
            try:
                x = float(v)
            except Exception:
                x = 1.0
            x = max(1.0, min(5.0, x))
            return round((x - 1.0) / 4.0, 4)

        return {
            "consistency": to_01(obj.get("consistency", 1)),
            "coverage": to_01(obj.get("coverage", 1)),
            "quality": to_01(obj.get("quality", 1)),
        }

    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
    )
    out = (resp.choices[0].message.content or "").strip()
    return parse_json_scores(out)

def compute_llm_eval_a(
    candidate: str,
    reference: str,
    model: str = DEFAULT_EVAL_MODEL,
    n_samples: int = 3,           # REQUIRED BY USER
    max_retries: int = 3,
    retry_delay: float = 0.8,
    **kwargs,
) -> Dict[str, float]:
    """
    Repeat Scheme A calls n_samples times and average (robustness),
    still far fewer calls than old 4-dim * n_samples.
    """
    scores_list: List[Dict[str, float]] = []
    attempts = 0
    # We want n_samples successful parses; allow retries per-sample
    while len(scores_list) < n_samples and attempts < n_samples * max_retries:
        attempts += 1
        try:
            s = compute_llm_eval_a_once(candidate, reference, model=model, **kwargs)
            # accept if not all zeros
            if any(v > 0 for v in s.values()):
                scores_list.append(s)
            time.sleep(0.15)  # small pacing to reduce burstiness
        except RequestsConnectionError:
            time.sleep(retry_delay)
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "timeout" in msg or "temporarily" in msg:
                time.sleep(retry_delay)
            else:
                time.sleep(0.2)

    if not scores_list:
        return {"consistency": 0.0, "coverage": 0.0, "quality": 0.0}

    def avg(k: str) -> float:
        vals = [d.get(k, 0.0) for d in scores_list]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    return {"consistency": avg("consistency"), "coverage": avg("coverage"), "quality": avg("quality")}

# Backward-compatible wrapper:
# - coherence <- quality
# - fluency  <- quality
# - relevance <- coverage
# - consistency <- consistency
def compute_four_dimensions(candidate: str, reference: str, model: str = DEFAULT_EVAL_MODEL) -> Dict[str, float]:
    scores = compute_llm_eval_a(candidate=candidate, reference=reference, model=model, n_samples=3)
    return {
        "coherence": scores["quality"],
        "consistency": scores["consistency"],
        "fluency": scores["quality"],
        "relevance": scores["coverage"],
    }

# -----------------------------
# Main evaluation
# -----------------------------
def evaluate_text_quality(data: list, iteration: Optional[int] = None, pre_detailed_metrics: Optional[Dict] = None) -> Dict[str, Any]:
    metrics = {
        "pre": {
            "bert_score": [], "sms": [], "gptscore": [],
            "g_eval_coherence": [], "g_eval_consistency": [],
            "g_eval_fluency": [], "g_eval_relevance": [],
        },
        "current": {
            "bert_score": [], "sms": [], "gptscore": [],
            "g_eval_coherence": [], "g_eval_consistency": [],
            "g_eval_fluency": [], "g_eval_relevance": [],
        },
    }

    sample_details: List[Dict[str, Any]] = []
    batch = {"ids": [], "pre_raw": [], "current_raw": [], "human_raw": []}

    total_items = len(data)
    print(f"[Evaluation Progress] Calculating per-sample metrics (total: {total_items})")

    # Collect + SMS (local)
    for item in tqdm(data, desc="[Collecting Samples]"):
        item_id = item.get("id", f"unknown_{len(batch['ids'])}")
        human_raw = _safe_strip(item.get("human_news"))

        current_gpt_raw = _safe_strip(item.get("gpt_news") or item.get("machine_news") or item.get("pre_gpt_news"))
        pre_raw = _safe_strip(item.get("pre_gpt_news") or item.get("machine_news"))

        if not pre_raw and current_gpt_raw:
            pre_raw = current_gpt_raw

        if not human_raw:
            print(f"[Warning] Skipping invalid sample (id: {item_id}) - missing human_news")
            continue
        if not current_gpt_raw:
            print(f"[Warning] Skipping invalid sample (id: {item_id}) - missing candidate text")
            continue

        sample_metrics = {"id": item_id, "pre": {}, "current": {}}

        pre_sms = compute_sentence_movers_similarity(pre_raw, human_raw)
        current_sms = compute_sentence_movers_similarity(current_gpt_raw, human_raw)
        metrics["pre"]["sms"].append(pre_sms)
        metrics["current"]["sms"].append(current_sms)
        sample_metrics["pre"]["sms"] = pre_sms
        sample_metrics["current"]["sms"] = current_sms

        batch["ids"].append(item_id)
        batch["pre_raw"].append(pre_raw)
        batch["current_raw"].append(current_gpt_raw)
        batch["human_raw"].append(human_raw)
        sample_details.append(sample_metrics)

    # If we have previous detailed metrics, reuse pre metrics for expensive parts
    if pre_detailed_metrics is not None:
        print("[Evaluation] Reusing previous current metrics as pre metrics")
        metrics["pre"]["bert_score"] = pre_detailed_metrics["metrics"]["current"]["bert_score"]
        metrics["pre"]["gptscore"] = pre_detailed_metrics["metrics"]["current"]["gptscore"]
        metrics["pre"]["g_eval_coherence"] = pre_detailed_metrics["metrics"]["current"]["g_eval_coherence"]
        metrics["pre"]["g_eval_consistency"] = pre_detailed_metrics["metrics"]["current"]["g_eval_consistency"]
        metrics["pre"]["g_eval_fluency"] = pre_detailed_metrics["metrics"]["current"]["g_eval_fluency"]
        metrics["pre"]["g_eval_relevance"] = pre_detailed_metrics["metrics"]["current"]["g_eval_relevance"]

        for i, sample in enumerate(sample_details):
            sample["pre"]["bert_score"] = pre_detailed_metrics["samples"][i]["current"].get("bert_score", 0.0)
            sample["pre"]["gptscore"] = pre_detailed_metrics["samples"][i]["current"].get("gptscore", 0.0)
            sample["pre"]["g_eval"] = pre_detailed_metrics["samples"][i]["current"].get("g_eval", {})
    else:
        print("[Evaluation] No previous metrics, computing pre metrics from scratch")

        # ---- GPTScore pre (kept) ----
        try:
            def gptscore_worker(cand: str, ref: str, sample_id: str) -> Tuple[str, float]:
                try:
                    return (sample_id, compute_gptscore(cand, ref))
                except Exception as e:
                    print(f"[GPTScore Worker Error] Sample {sample_id}: {str(e)}")
                    return (sample_id, 0.0)

            pre_tasks = list(zip(batch["pre_raw"], batch["human_raw"], batch["ids"]))
            pre_gpt_scores = {sid: 0.0 for sid in batch["ids"]}
            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
                futures = [ex.submit(gptscore_worker, c, r, sid) for c, r, sid in pre_tasks]
                for f in tqdm(as_completed(futures), total=len(futures), desc="GPTScore pre"):
                    sid, sc = f.result()
                    pre_gpt_scores[sid] = sc

            metrics["pre"]["gptscore"] = [pre_gpt_scores[sid] for sid in batch["ids"]]
            for sample in sample_details:
                sample["pre"]["gptscore"] = pre_gpt_scores.get(sample["id"], 0.0)
        except Exception as e:
            print("[GPTScore ERROR]", e)
            metrics["pre"]["gptscore"] = [0.0] * len(batch["pre_raw"])
            for sample in sample_details:
                sample["pre"]["gptscore"] = 0.0

        # ---- BERTScore pre ----
        try:
            REQUIRED_FILES = ["config.json", "vocab.txt", "pytorch_model.bin"]

            def validate_local_model(path: str) -> bool:
                if not os.path.isdir(path):
                    raise FileNotFoundError(f"Local model directory does not exist: {path}")
                missing = []
                for fn in REQUIRED_FILES:
                    fp = os.path.join(path, fn)
                    if not os.path.exists(fp):
                        if fn == "pytorch_model.bin" and os.path.exists(os.path.join(path, "model.safetensors")):
                            continue
                        missing.append(fn)
                if missing:
                    raise FileNotFoundError(f"Local model missing key files: {missing}")
                return True

            validate_local_model(LOCAL_BERT_PATH)
            print(f"[BERTScore] Local model validation passed, path: {LOCAL_BERT_PATH}")

            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_CACHE"] = LOCAL_BERT_PATH

            tokenizer = AutoTokenizer.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
            model = AutoModel.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)

            def truncate_list(texts: List[str]) -> List[str]:
                out = []
                for t in texts:
                    if not (t or "").strip():
                        out.append("empty text")
                        continue
                    ids = tokenizer(t, truncation=True, max_length=512, return_tensors="pt")
                    out.append(tokenizer.decode(ids.input_ids[0], skip_special_tokens=True))
                return out

            if not batch["pre_raw"] or not batch["human_raw"]:
                pre_bs_list = [0.0] * len(batch["pre_raw"])
            else:
                pre_t = truncate_list(batch["pre_raw"])
                hum_t = truncate_list(batch["human_raw"])
                if not pre_t or not hum_t:
                    pre_bs_list = [0.0] * len(batch["pre_raw"])
                else:
                    pre_p, pre_r, pre_f = bertscore_score(
                        pre_t, hum_t,
                        lang="en",
                        model_type=LOCAL_BERT_PATH,
                        num_layers=12,
                        device=device,
                        batch_size=BERTSCORE_BATCH_SIZE,
                    )
                    pre_bs_list = pre_f.cpu().tolist()

            metrics["pre"]["bert_score"] = pre_bs_list
            for i, sample in enumerate(sample_details):
                sample["pre"]["bert_score"] = pre_bs_list[i] if i < len(pre_bs_list) else 0.0
        except Exception as e:
            print(f"[BERTScore ERROR] {str(e)}")
            print(traceback.format_exc())
            metrics["pre"]["bert_score"] = [0.0] * len(batch["pre_raw"])
            for sample in sample_details:
                sample["pre"]["bert_score"] = 0.0

        # ---- Scheme A "G-Eval" pre (n_samples=3, worker=8) ----
        try:
            def eval_worker(cand: str, ref: str, sample_id: str) -> Tuple[str, Dict[str, float]]:
                try:
                    scores = compute_four_dimensions(cand, ref, model=DEFAULT_EVAL_MODEL)
                    return (sample_id, scores)
                except Exception as e:
                    print(f"[LLM-Eval Worker Error] Sample {sample_id}: {str(e)}")
                    return (sample_id, {"coherence": 0.0, "consistency": 0.0, "fluency": 0.0, "relevance": 0.0})

            pre_tasks = list(zip(batch["pre_raw"], batch["human_raw"], batch["ids"]))
            pre_eval_scores = {sid: {} for sid in batch["ids"]}

            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
                futures = [ex.submit(eval_worker, c, r, sid) for c, r, sid in pre_tasks]
                for f in tqdm(as_completed(futures), total=len(futures), desc="LLM-Eval pre"):
                    sid, scores = f.result()
                    pre_eval_scores[sid] = scores

            for dim in ["coherence", "consistency", "fluency", "relevance"]:
                metrics["pre"][f"g_eval_{dim}"] = [pre_eval_scores[sid].get(dim, 0.0) for sid in batch["ids"]]

            for sample in sample_details:
                sid = sample["id"]
                sample["pre"]["g_eval"] = pre_eval_scores.get(sid, {})
        except Exception as e:
            print("[LLM-Eval ERROR]", e)
            for dim in ["coherence", "consistency", "fluency", "relevance"]:
                metrics["pre"][f"g_eval_{dim}"] = [0.0] * len(batch["pre_raw"])
            for sample in sample_details:
                sample["pre"]["g_eval"] = {"coherence": 0.0, "consistency": 0.0, "fluency": 0.0, "relevance": 0.0}

    # -----------------------------
    # Current metrics
    # -----------------------------
    print("[Evaluation] Computing current metrics from scratch")

    # ---- GPTScore current ----
    try:
        def gptscore_worker(cand: str, ref: str, sample_id: str) -> Tuple[str, float]:
            try:
                return (sample_id, compute_gptscore(cand, ref))
            except Exception as e:
                print(f"[GPTScore Worker Error] Sample {sample_id}: {str(e)}")
                return (sample_id, 0.0)

        cur_tasks = list(zip(batch["current_raw"], batch["human_raw"], batch["ids"]))
        cur_gpt_scores = {sid: 0.0 for sid in batch["ids"]}

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
            futures = [ex.submit(gptscore_worker, c, r, sid) for c, r, sid in cur_tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="GPTScore current"):
                sid, sc = f.result()
                cur_gpt_scores[sid] = sc

        metrics["current"]["gptscore"] = [cur_gpt_scores[sid] for sid in batch["ids"]]
        for sample in sample_details:
            sample["current"]["gptscore"] = cur_gpt_scores.get(sample["id"], 0.0)
    except Exception as e:
        print("[GPTScore ERROR]", e)
        metrics["current"]["gptscore"] = [0.0] * len(batch["current_raw"])
        for sample in sample_details:
            sample["current"]["gptscore"] = 0.0

    # ---- BERTScore current ----
    try:
        REQUIRED_FILES = ["config.json", "vocab.txt", "pytorch_model.bin"]

        def validate_local_model(path: str) -> bool:
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Local model directory does not exist: {path}")
            missing = []
            for fn in REQUIRED_FILES:
                fp = os.path.join(path, fn)
                if not os.path.exists(fp):
                    if fn == "pytorch_model.bin" and os.path.exists(os.path.join(path, "model.safetensors")):
                        continue
                    missing.append(fn)
            if missing:
                raise FileNotFoundError(f"Local model missing key files: {missing}")
            return True

        validate_local_model(LOCAL_BERT_PATH)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_CACHE"] = LOCAL_BERT_PATH

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
        model = AutoModel.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)

        def truncate_list(texts: List[str]) -> List[str]:
            out = []
            for t in texts:
                if not (t or "").strip():
                    out.append("empty text")
                    continue
                ids = tokenizer(t, truncation=True, max_length=512, return_tensors="pt")
                out.append(tokenizer.decode(ids.input_ids[0], skip_special_tokens=True))
            return out

        if not batch["current_raw"] or not batch["human_raw"]:
            cur_bs_list = [0.0] * len(batch["current_raw"])
        else:
            cur_t = truncate_list(batch["current_raw"])
            hum_t = truncate_list(batch["human_raw"])
            if not cur_t or not hum_t:
                cur_bs_list = [0.0] * len(batch["current_raw"])
            else:
                cur_p, cur_r, cur_f = bertscore_score(
                    cur_t, hum_t,
                    lang="en",
                    model_type=LOCAL_BERT_PATH,
                    num_layers=12,
                    device=device,
                    batch_size=BERTSCORE_BATCH_SIZE,
                )
                cur_bs_list = cur_f.cpu().tolist()

        metrics["current"]["bert_score"] = cur_bs_list
        for i, sample in enumerate(sample_details):
            sample["current"]["bert_score"] = cur_bs_list[i] if i < len(cur_bs_list) else 0.0
    except Exception as e:
        print(f"[BERTScore ERROR] {str(e)}")
        print(traceback.format_exc())
        metrics["current"]["bert_score"] = [0.0] * len(batch["current_raw"])
        for sample in sample_details:
            sample["current"]["bert_score"] = 0.0

    # ---- Scheme A "G-Eval" current (n_samples=3, worker=8) ----
    try:
        def eval_worker(cand: str, ref: str, sample_id: str) -> Tuple[str, Dict[str, float]]:
            try:
                scores = compute_four_dimensions(cand, ref, model=DEFAULT_EVAL_MODEL)
                return (sample_id, scores)
            except Exception as e:
                print(f"[LLM-Eval Worker Error] Sample {sample_id}: {str(e)}")
                return (sample_id, {"coherence": 0.0, "consistency": 0.0, "fluency": 0.0, "relevance": 0.0})

        cur_tasks = list(zip(batch["current_raw"], batch["human_raw"], batch["ids"]))
        cur_eval_scores = {sid: {} for sid in batch["ids"]}

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
            futures = [ex.submit(eval_worker, c, r, sid) for c, r, sid in cur_tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="LLM-Eval current"):
                sid, scores = f.result()
                cur_eval_scores[sid] = scores

        for dim in ["coherence", "consistency", "fluency", "relevance"]:
            metrics["current"][f"g_eval_{dim}"] = [cur_eval_scores[sid].get(dim, 0.0) for sid in batch["ids"]]

        for sample in sample_details:
            sid = sample["id"]
            sample["current"]["g_eval"] = cur_eval_scores.get(sid, {})
    except Exception as e:
        print("[LLM-Eval ERROR]", e)
        for dim in ["coherence", "consistency", "fluency", "relevance"]:
            metrics["current"][f"g_eval_{dim}"] = [0.0] * len(batch["current_raw"])
        for sample in sample_details:
            sample["current"]["g_eval"] = {"coherence": 0.0, "consistency": 0.0, "fluency": 0.0, "relevance": 0.0}

    # -----------------------------
    # Aggregate
    # -----------------------------
    def mean(xs: List[float]) -> float:
        xs = [x for x in xs if isinstance(x, (int, float))]
        return float(np.mean(xs)) if xs else 0.0

    # average of 4 g_eval fields per sample
    pre_g_eval_averages = []
    cur_g_eval_averages = []

    pre_len = min(
        len(metrics["pre"]["g_eval_coherence"]),
        len(metrics["pre"]["g_eval_consistency"]),
        len(metrics["pre"]["g_eval_fluency"]),
        len(metrics["pre"]["g_eval_relevance"]),
    ) if metrics["pre"]["g_eval_coherence"] else 0

    cur_len = min(
        len(metrics["current"]["g_eval_coherence"]),
        len(metrics["current"]["g_eval_consistency"]),
        len(metrics["current"]["g_eval_fluency"]),
        len(metrics["current"]["g_eval_relevance"]),
    ) if metrics["current"]["g_eval_coherence"] else 0

    for i in range(pre_len):
        pre_g_eval_averages.append(mean([
            metrics["pre"]["g_eval_coherence"][i],
            metrics["pre"]["g_eval_consistency"][i],
            metrics["pre"]["g_eval_fluency"][i],
            metrics["pre"]["g_eval_relevance"][i],
        ]))

    for i in range(cur_len):
        cur_g_eval_averages.append(mean([
            metrics["current"]["g_eval_coherence"][i],
            metrics["current"]["g_eval_consistency"][i],
            metrics["current"]["g_eval_fluency"][i],
            metrics["current"]["g_eval_relevance"][i],
        ]))

    pre_avg = {
        "bert_score": mean(metrics["pre"]["bert_score"]),
        "sms": mean(metrics["pre"]["sms"]),
        "gptscore": mean(metrics["pre"]["gptscore"]),
        "g_eval_coherence": mean(metrics["pre"]["g_eval_coherence"]),
        "g_eval_consistency": mean(metrics["pre"]["g_eval_consistency"]),
        "g_eval_fluency": mean(metrics["pre"]["g_eval_fluency"]),
        "g_eval_relevance": mean(metrics["pre"]["g_eval_relevance"]),
        "g_eval_average": mean(pre_g_eval_averages),
    }

    cur_avg = {
        "bert_score": mean(metrics["current"]["bert_score"]),
        "sms": mean(metrics["current"]["sms"]),
        "gptscore": mean(metrics["current"]["gptscore"]),
        "g_eval_coherence": mean(metrics["current"]["g_eval_coherence"]),
        "g_eval_consistency": mean(metrics["current"]["g_eval_consistency"]),
        "g_eval_fluency": mean(metrics["current"]["g_eval_fluency"]),
        "g_eval_relevance": mean(metrics["current"]["g_eval_relevance"]),
        "g_eval_average": mean(cur_g_eval_averages),
    }

    return {
        "iteration": iteration,
        "detailed": {"samples": sample_details, "metrics": metrics},
        "pre_average": pre_avg,
        "current_average": cur_avg,
    }
