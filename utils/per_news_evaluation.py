"""
@filename:per_news_evaluation.py
@author:Stmxlt
@time:2026-01-26
"""

import os
import copy
import numpy as np
import json
from typing import Dict, List, Any, Optional
import datetime

from utils.evaluation import evaluate_text_quality


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def _iter_key(iteration: int) -> str:
    return f"iteration_{int(iteration)}"


class PerNewsEvaluation:
    """Store and manage per-news evaluation metrics grouped by news_id (v3 list format)."""

    def __init__(self, storage_path: str = "result/per_news_metrics.json"):
        self.storage_path = storage_path
        self.ensure_storage_exists()

    def ensure_storage_exists(self) -> None:
        """Ensure the storage file exists; create an empty list if it does not."""
        _ensure_dir(self.storage_path)
        if not os.path.exists(self.storage_path):
            self.save_storage([])

    def load_storage(self) -> List[Dict[str, Any]]:
        """
        Load storage list. If old formats are detected, migrate to v3 automatically.

        Supported legacy formats:
        - v1: list of records: {"news_id","iteration","metrics","timestamp"}
        - v2: dict with "iterations": {"1": {"metrics_by_news": {...}}}
        """
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    self.save_storage([])
                    return []
                raw = json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            self.save_storage([])
            return []

        if isinstance(raw, list) and (len(raw) == 0 or isinstance(raw[0], dict)):
            # Could be v1 or v3; detect by fields
            if self._looks_like_v1(raw):
                migrated = self._migrate_v1_to_v3(raw)
                self.save_storage(migrated)
                return migrated
            return raw

        if isinstance(raw, dict) and "iterations" in raw and isinstance(raw["iterations"], dict):
            migrated = self._migrate_v2_to_v3(raw)
            self.save_storage(migrated)
            return migrated

        self.save_storage([])
        return []

    def save_storage(self, data: List[Dict[str, Any]]) -> None:
        _ensure_dir(self.storage_path)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _looks_like_v1(self, raw: List[Dict[str, Any]]) -> bool:
        for rec in raw[:5]:
            if isinstance(rec, dict) and ("iteration" in rec and "metrics" in rec):
                return True
        return False

    def _migrate_v1_to_v3(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        v1 record: {"news_id": "...", "iteration": 1, "metrics": {...}, "timestamp": "..."}
        -> v3 list: [{"news_id":"...", "iteration_1": {...}, ...}, ...]
        """
        by_news: Dict[str, Dict[str, Any]] = {}

        for rec in records:
            try:
                news_id = str(rec.get("news_id", "")).strip()
                it = rec.get("iteration", None)
                metrics = rec.get("metrics", {})
                if not news_id or it is None:
                    continue
                k = _iter_key(int(it))
                obj = by_news.setdefault(news_id, {"news_id": news_id})
                obj[k] = metrics
            except Exception:
                continue

        # stable order by numeric news_id when possible
        def sort_key(x: Dict[str, Any]):
            nid = x.get("news_id", "")
            try:
                return (0, int(nid))
            except Exception:
                return (1, str(nid))

        return sorted(by_news.values(), key=sort_key)

    def _migrate_v2_to_v3(self, raw_v2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        v2 format:
        {"version":2,"iterations":{"1":{"metrics_by_news":{"id":{...}}},...}}
        -> v3 list per news_id
        """
        by_news: Dict[str, Dict[str, Any]] = {}
        iterations = raw_v2.get("iterations", {})

        for it_key, bucket in iterations.items():
            try:
                it_int = int(it_key)
            except Exception:
                continue
            k = _iter_key(it_int)
            mbn = (bucket or {}).get("metrics_by_news", {})
            if not isinstance(mbn, dict):
                continue
            for news_id, metrics in mbn.items():
                nid = str(news_id).strip()
                if not nid:
                    continue
                obj = by_news.setdefault(nid, {"news_id": nid})
                obj[k] = metrics

        def sort_key(x: Dict[str, Any]):
            nid = x.get("news_id", "")
            try:
                return (0, int(nid))
            except Exception:
                return (1, str(nid))

        return sorted(by_news.values(), key=sort_key)

    def _index_by_news_id(self, storage: List[Dict[str, Any]]) -> Dict[str, int]:
        idx = {}
        for i, obj in enumerate(storage):
            nid = str(obj.get("news_id", "")).strip()
            if nid:
                idx[nid] = i
        return idx

    def store_current_iteration_metrics(self, iteration: int, detailed_results: Dict[str, Any]) -> None:
        """
        Store per-news CURRENT metrics into the v3 structure:
        Each news item is a dict: {"news_id":"...", "iteration_k": {...}}
        """
        storage = self.load_storage()
        idx = self._index_by_news_id(storage)
        it_k = _iter_key(iteration)

        samples = detailed_results.get("samples", [])
        for sample in samples:
            news_id = str(sample.get("id", "")).strip()
            if not news_id:
                continue
            current_metrics = sample.get("current", {}) or {}

            if news_id in idx:
                storage[idx[news_id]][it_k] = current_metrics
            else:
                storage.append({"news_id": news_id, it_k: current_metrics})
                idx[news_id] = len(storage) - 1

        # keep stable order
        def sort_key(x: Dict[str, Any]):
            nid = x.get("news_id", "")
            try:
                return (0, int(nid))
            except Exception:
                return (1, str(nid))

        storage.sort(key=sort_key)
        self.save_storage(storage)

    def get_iteration_metrics(self, news_id: str, iteration: int) -> Dict[str, Any]:
        """Get metrics for a specific (news_id, iteration). Returns {} if missing."""
        storage = self.load_storage()
        nid = str(news_id).strip()
        it_k = _iter_key(iteration)
        for obj in storage:
            if str(obj.get("news_id", "")).strip() == nid:
                m = obj.get(it_k, {})
                return m if isinstance(m, dict) else {}
        return {}

    def get_comparison_data(self, news_id: str, iteration: int) -> Dict[str, Any]:
        """
        Compare current iteration vs previous iteration for the same news_id.
        """
        nid = str(news_id).strip()
        cur = self.get_iteration_metrics(nid, iteration)
        prev = self.get_iteration_metrics(nid, iteration - 1) if iteration and iteration > 1 else {}

        return {
            "news_id": nid,
            "iteration": iteration,
            "previous_metrics": prev,
            "current_metrics": cur,
            "has_previous": bool(prev),
        }

    def get_all_news_ids(self) -> List[str]:
        storage = self.load_storage()
        ids = []
        for obj in storage:
            nid = str(obj.get("news_id", "")).strip()
            if nid:
                ids.append(nid)
        return ids

    def get_all_news_comparison(self, iteration: int) -> List[Dict[str, Any]]:
        """
        Get comparisons for ALL news that have metrics at 'iteration'.
        """
        storage = self.load_storage()
        it_k = _iter_key(iteration)

        out: List[Dict[str, Any]] = []
        for obj in storage:
            nid = str(obj.get("news_id", "")).strip()
            if not nid:
                continue
            if it_k in obj and isinstance(obj.get(it_k), dict):
                out.append(self.get_comparison_data(nid, iteration))
        return out

    def export_to_excel(self, output_path: str = "result/per_news_metrics.xlsx") -> None:
        """
        Export to Excel placeholder (kept for compatibility).
        """
        print(f"Per-news metrics JSON path: {self.storage_path}")


def evaluate_with_per_news_tracking(
    data: list,
    iteration: Optional[int] = None,
    per_news_evaluator: Optional[PerNewsEvaluation] = None,
) -> Dict[str, Any]:
    if per_news_evaluator is None:
        per_news_evaluator = PerNewsEvaluation()

    previous_detailed_metrics = None
    if iteration and iteration > 1:
        prev_it = iteration - 1
        storage = per_news_evaluator.load_storage()
        prev_key = _iter_key(prev_it)

        prev_metrics_by_id: Dict[str, Dict[str, Any]] = {}
        for obj in storage:
            nid = str(obj.get("news_id", "")).strip()
            if not nid:
                continue
            m = obj.get(prev_key, None)
            if isinstance(m, dict):
                prev_metrics_by_id[nid] = m

        prev_samples: List[Dict[str, Any]] = []
        for item in data:
            nid = str(item.get("id", "")).strip()
            
            m = prev_metrics_by_id.get(nid, {}) if nid else {}
            if not isinstance(m, dict):
                m = {}
                
            prev_samples.append({
                "id": nid,
                "pre": {},
                "current": m,
            })

        if prev_samples:
            def _get_geval(cur: Dict[str, Any], dim: str) -> float:
                g = cur.get("g_eval", {})
                if isinstance(g, dict) and dim in g:
                    try:
                        return float(g.get(dim, 0) or 0)
                    except Exception:
                        return 0.0
                try:
                    return float(cur.get(f"g_eval_{dim}", 0) or 0)
                except Exception:
                    return 0.0

            previous_detailed_metrics = {
                "samples": prev_samples,
                "metrics": {
                    "current": {
                        "bert_score": [s["current"].get("bert_score", 0) for s in prev_samples],
                        "sms": [s["current"].get("sms", 0) for s in prev_samples],
                        "gptscore": [s["current"].get("gptscore", 0) for s in prev_samples],
                        "g_eval_coherence":   [_get_geval(s["current"], "coherence")   for s in prev_samples],
                        "g_eval_consistency": [_get_geval(s["current"], "consistency") for s in prev_samples],
                        "g_eval_fluency":     [_get_geval(s["current"], "fluency")     for s in prev_samples],
                        "g_eval_relevance":   [_get_geval(s["current"], "relevance")   for s in prev_samples],
                    }
                },
            }

    results = evaluate_text_quality(data, iteration=iteration, pre_detailed_metrics=previous_detailed_metrics)

    rollback_ids = set()
    if iteration:
        samples = results.get("detailed", {}).get("samples", [])
        metrics_arrays = results.get("detailed", {}).get("metrics", {})

        for i, sample in enumerate(samples):
            pre = sample.get("pre", {})
            cur = sample.get("current", {})

            def c_sum(m):
                g = m.get("g_eval", {})
                g_avg = (g.get("coherence", 0) + g.get("consistency", 0) + 
                         g.get("fluency", 0) + g.get("relevance", 0)) / 4.0 if g else 0.0
                return m.get("bert_score", 0) + m.get("sms", 0) + m.get("gptscore", 0) + g_avg

            if c_sum(cur) < c_sum(pre):
                rollback_ids.add(sample.get("id"))
                sample["current"] = copy.deepcopy(pre)
                
                for k in ["bert_score", "sms", "gptscore", "g_eval_coherence", "g_eval_consistency", "g_eval_fluency", "g_eval_relevance"]:
                    if k in metrics_arrays.get("current", {}) and k in metrics_arrays.get("pre", {}):
                        if i < len(metrics_arrays["current"][k]) and i < len(metrics_arrays["pre"][k]):
                            metrics_arrays["current"][k][i] = metrics_arrays["pre"][k][i]

        if rollback_ids:
            print(f"[Rollback Intercept] rollback {len(rollback_ids)} news articles。")
            
            cur_arrays = metrics_arrays.get("current", {})
            def mean(xs): return float(np.mean(xs)) if xs else 0.0
            
            cur_geval_avgs = []
            n = len(cur_arrays.get("g_eval_coherence", []))
            for i in range(n):
                cur_geval_avgs.append(mean([
                    cur_arrays.get("g_eval_coherence", [])[i] if i < len(cur_arrays.get("g_eval_coherence", [])) else 0,
                    cur_arrays.get("g_eval_consistency", [])[i] if i < len(cur_arrays.get("g_eval_consistency", [])) else 0,
                    cur_arrays.get("g_eval_fluency", [])[i] if i < len(cur_arrays.get("g_eval_fluency", [])) else 0,
                    cur_arrays.get("g_eval_relevance", [])[i] if i < len(cur_arrays.get("g_eval_relevance", [])) else 0
                ]))

            results["current_average"] = {
                "bert_score": mean(cur_arrays.get("bert_score", [])),
                "sms": mean(cur_arrays.get("sms", [])),
                "gptscore": mean(cur_arrays.get("gptscore", [])),
                "g_eval_coherence": mean(cur_arrays.get("g_eval_coherence", [])),
                "g_eval_consistency": mean(cur_arrays.get("g_eval_consistency", [])),
                "g_eval_fluency": mean(cur_arrays.get("g_eval_fluency", [])),
                "g_eval_relevance": mean(cur_arrays.get("g_eval_relevance", [])),
                "g_eval_average": mean(cur_geval_avgs),
            }
            
        results["rollback_ids"] = rollback_ids

        per_news_evaluator.store_current_iteration_metrics(iteration, results["detailed"])

    return results



def get_per_news_improvement_suggestions(
    news_id: str,
    iteration: int,
    per_news_evaluator: PerNewsEvaluation,
) -> Dict[str, str]:
    """
    Generate improvement suggestions by comparing iteration vs iteration-1 for the same news_id.
    """
    comparison = per_news_evaluator.get_comparison_data(news_id, iteration)

    if not comparison["has_previous"]:
        return {"message": "This is the first iteration; no previous data is available for comparison."}

    previous_metrics = comparison["previous_metrics"]
    current_metrics = comparison["current_metrics"]

    suggestions: Dict[str, str] = {}

    # BERTScore
    if current_metrics.get("bert_score", 0) < previous_metrics.get("bert_score", 0) - 0.03:
        suggestions["bert_score"] = "Improve semantic similarity: reuse more vocabulary and phrasing from the human news."
    elif current_metrics.get("bert_score", 0) > previous_metrics.get("bert_score", 0) + 0.01:
        suggestions["bert_score"] = "Good improvement: BERTScore increased; keep the current strategy."

    # SMS
    if current_metrics.get("sms", 0) < previous_metrics.get("sms", 0) - 0.03:
        suggestions["sms"] = "Improve sentence structure: adjust ordering and structure to better match the human news."
    elif current_metrics.get("sms", 0) > previous_metrics.get("sms", 0) + 0.01:
        suggestions["sms"] = "Good improvement: structural similarity increased; keep it up."

    # GPTScore
    if current_metrics.get("gptscore", 0) < previous_metrics.get("gptscore", 0) - 0.03:
        suggestions["gptscore"] = "Improve overall quality: enhance content, structure, and language across the article."
    elif current_metrics.get("gptscore", 0) > previous_metrics.get("gptscore", 0) + 0.01:
        suggestions["gptscore"] = "Good improvement: overall quality increased; keep the current strategy."

    # G-Eval dims
    for dim in ["coherence", "consistency", "fluency", "relevance"]:
        key = f"g_eval_{dim}"
        cur_v = current_metrics.get(key, 0)
        prev_v = previous_metrics.get(key, 0)

        if cur_v < prev_v - 0.03:
            if dim == "coherence":
                suggestions[key] = "Strengthen coherence: ensure clear logical links between paragraphs."
            elif dim == "consistency":
                suggestions[key] = "Improve factual consistency: avoid contradictions and unsupported claims."
            elif dim == "fluency":
                suggestions[key] = "Improve fluency: refine grammar, word choice, and sentence flow."
            elif dim == "relevance":
                suggestions[key] = "Improve relevance: focus on core information and remove irrelevant content."
        elif cur_v > prev_v + 0.01:
            suggestions[key] = f"Good improvement: {dim.capitalize()} increased; keep the current strategy."

    return suggestions
