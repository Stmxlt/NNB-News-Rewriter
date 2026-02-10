'''
@filename:Generator.py
@author:Stmxlt 
@time:2026-02-10
'''


from __future__ import annotations

import os
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


USER_SUMMARY_TXT_PATH = "user_news_abstract.txt"
DATASET_JSON_PATH = "dataset/rewrited_cnn_dailymail.json"
SIMILARITY_MODEL_PATH = "./local_models/all-MiniLM-L6-v2"
TOP_K = 5
OUTPUT_LANGUAGE = "English"

GEN_MODEL = os.getenv("GEN_MODEL", "moonshotai/Kimi-K2-Instruct-0905")
TEMPERATURE = 0.4
MAX_TOKENS = 4096

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-tivywyulofozvjneifqxrqqtbkhknkgmfwnrodmywsscozmy")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "120"))

OUTPUT_DIR = "result/"

_file_lock = threading.Lock()

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    timeout=OPENAI_TIMEOUT,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_text_file(txt_path: str, encoding: str = "utf-8") -> str:
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Input summary txt not found: {txt_path}")
    with open(txt_path, "r", encoding=encoding) as f:
        return (f.read() or "").strip()

def json2dict(path: str) -> List[Dict[str, Any]]:
    with _file_lock:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset JSON not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"Dataset JSON is empty: {path}")
        with open(path, "r", encoding="utf-8") as f:
            s = (f.read() or "").strip()
        if not s:
            return []
        return json.loads(s)

def save_text(text: str, out_path: str, encoding: str = "utf-8") -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding=encoding) as f:
        f.write(text)

@dataclass
class SimilarItem:
    id: str
    score: float
    abstract: str
    exemplar_news: str

class SimilarityRetriever:
    """
    Retrieve top-K similar items from CNN/DailyMail dataset based on abstract similarity.
    Uses SentenceTransformer + cosine similarity.

    Exemplar news preference order:
      1) pre_gpt_news (saved last iteration during pretraining)
      2) gpt_news
      3) machine_news
      4) human_news
    """

    def __init__(self, dataset_json: str, model_path: str):
        self.dataset_json = dataset_json
        self.model_path = model_path
        self.model = SentenceTransformer(model_path)

        self._loaded = False
        self._items: List[Dict[str, Any]] = []
        self._abstracts: List[str] = []
        self._ids: List[str] = []
        self._embs = None

    def load(self) -> None:
        if self._loaded:
            return

        raw = json2dict(self.dataset_json)

        items: List[Dict[str, Any]] = []
        abstracts: List[str] = []
        ids: List[str] = []

        for it in raw:
            _id = str(it.get("id", "")).strip()
            abs_ = (it.get("abstract") or "").strip()
            if not _id or not abs_:
                continue
            items.append(it)
            abstracts.append(abs_)
            ids.append(_id)

        if not items:
            raise ValueError(f"No valid items (need 'id' and 'abstract') found in {self.dataset_json}")

        self._items = items
        self._abstracts = abstracts
        self._ids = ids

        self._embs = self.model.encode(self._abstracts, convert_to_tensor=False)
        self._loaded = True

    @staticmethod
    def _pick_exemplar_news(item: Dict[str, Any]) -> str:
        for k in ["pre_gpt_news", "gpt_news", "machine_news", "human_news"]:
            v = item.get(k, "")
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def topk(self, query_abstract: str, k: int) -> List[SimilarItem]:
        query_abstract = (query_abstract or "").strip()
        if not query_abstract:
            raise ValueError("Empty query summary")

        self.load()

        q_emb = self.model.encode([query_abstract], convert_to_tensor=False)[0]
        sims = cosine_similarity([q_emb], self._embs)[0]

        idxs = list(range(len(self._ids)))
        idxs.sort(key=lambda i: float(sims[i]), reverse=True)
        top_idx = idxs[: max(1, int(k))]

        out: List[SimilarItem] = []
        for i in top_idx:
            item = self._items[i]
            out.append(
                SimilarItem(
                    id=self._ids[i],
                    score=float(sims[i]),
                    abstract=self._abstracts[i],
                    exemplar_news=self._pick_exemplar_news(item),
                )
            )
        return out

def build_generation_prompt(user_abstract: str, exemplars: List[SimilarItem], language: str) -> str:
    user_abstract = (user_abstract or "").strip()
    if not user_abstract:
        raise ValueError("User abstract is empty")

    inst = f"""You are an expert news writer.

You will be given several EXEMPLAR pairs:
- News Summary (facts)
- High-quality News Article (style reference)

Then you will be given a NEW summary.
Your job: write a full news article that is ONLY grounded in the NEW summary's facts.
You MUST NOT add any new facts, entities, numbers, dates, locations, quotes, causes, or background knowledge that are not in the NEW summary.

Style requirements:
- Match the journalistic tone, structure, and flow of the exemplar articles.
- Use a clear headline, a concise lead paragraph, a well-structured body, and a short conclusion.
- Preserve any proper nouns and numbers exactly as in the NEW summary (if present).
- Paraphrasing is allowed, but factual meaning must remain identical.

Output language: {language}.
Output ONLY the news article, with no extra commentary.
"""

    ex_blocks: List[str] = []
    for idx, ex in enumerate(exemplars, start=1):
        news_text = ex.exemplar_news.strip() if ex.exemplar_news else "[MISSING EXEMPLAR ARTICLE]"
        ex_blocks.append(
            f"EXEMPLAR {idx} (similarity={ex.score:.4f}, id={ex.id}):\n"
            f"Summary:\n{ex.abstract}\n\n"
            f"Article:\n{news_text}\n"
        )

    return inst + "\n\n" + "\n\n".join(ex_blocks) + "\n\n" + f"NEW SUMMARY:\n{user_abstract}\n\nWRITE THE NEWS ARTICLE NOW:"

def generate_news_from_prompt(prompt: str) -> str:
    messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
    resp = openai_client.chat.completions.create(
        model=GEN_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()

def run_once() -> Tuple[str, str, List[SimilarItem]]:
    user_abs = read_text_file(USER_SUMMARY_TXT_PATH)

    retriever = SimilarityRetriever(dataset_json=DATASET_JSON_PATH, model_path=SIMILARITY_MODEL_PATH)
    similars = retriever.topk(user_abs, k=TOP_K)

    prompt = build_generation_prompt(user_abs, similars, language=OUTPUT_LANGUAGE)
    gen = generate_news_from_prompt(prompt)

    ensure_dir(OUTPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, f"user_generated_news.txt")
    save_text(gen, out_path)

    return out_path, gen, similars


if __name__ == "__main__":
    out_path, _, similars = run_once()