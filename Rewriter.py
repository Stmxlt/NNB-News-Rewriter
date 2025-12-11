"""
@filename:Rewriter.py
@author:Stmxlt
@time:2025-10-17
"""


import os
import json
import backoff
import openai
import tiktoken
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai import RateLimitError
from prompt import make_detection_prompt, make_attacking_prompt, top5_ids_only, json2dict, dict2json
from evaluation import evaluate_text_quality, create_dataset
from visualization import plot_metrics

print('===============Starting iteration===============')
gpt_turbo_encoding = tiktoken.get_encoding("cl100k_base")
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
    base_url=os.getenv("OPENAI_API_BASE", "your-api-link")
)
openai.api_type = "open_ai"
openai.api_version = None

_error_lock = threading.Lock()
file_rw_lock = threading.Lock()

_none_strip_error_ids = {
    "precompute_similar_ids": set(),
    "update_gpt_news": set(),
    "get_save_evaluation_feedback": set(),
    "get_save_attacking_feedback": set(),
}


def _log_none_strip(process_name: str, target_id):
    with _error_lock:
        if process_name not in _none_strip_error_ids:
            _none_strip_error_ids[process_name] = set()
        _none_strip_error_ids[process_name].add(str(target_id))

def _is_none_strip_error(e: Exception) -> bool:
    return "'NoneType' object has no attribute 'strip'" in str(e)

def atomic_update_json_field(json_path: str, target_id, updates: dict) -> bool:
    target_id_str = str(target_id)
    with file_rw_lock:
        data = json2dict(json_path)
        found = False
        for item in data:
            if str(item.get("id", "")) == target_id_str:
                for k, v in updates.items():
                    item[k] = v
                found = True
                break
        if found:
            dict2json(data, json_path)
    return found

def load_ids_from_json(json_path: str):
    try:
        with file_rw_lock:
            data = json2dict(json_path)
        ids = [item["id"] for item in data if "id" in item]
        return ids
    except Exception as e:
        print(f"[Config] Failed to load ids from {json_path}: {e}")
        return []

def precompute_similar_ids(json_path):
    with file_rw_lock:
        news_data = json2dict(json_path)
    total_items = len(news_data)
    print(f"Starting preprocessing of similar news IDs (total: {total_items} items)")

    for idx, item in tqdm(enumerate(news_data), total=total_items, desc="Preprocessing similar IDs"):
        target_id = str(item.get("id", ""))
        if not target_id:
            print(f"Warning: Skipping news item without ID (index: {idx})")
            continue

        existing_similar = item.get("similar", [])
        if isinstance(existing_similar, list) and len(existing_similar) == 5:
            continue

        try:
            similar_ids = top5_ids_only(target_id, json_path)
        except AttributeError as e:
            if _is_none_strip_error(e):
                _log_none_strip("precompute_similar_ids", target_id)
                continue
            else:
                raise
        except Exception as e:
            print(f"[precompute_similar_ids] Failed on id={target_id}: {e}")
            continue

        item["similar"] = similar_ids

    with file_rw_lock:
        dict2json(news_data, json_path)

def update_gpt_news(json_path):
    with file_rw_lock:
        news_data = json2dict(json_path)
        for item in news_data:
            target_id = str(item.get("id", ""))
            try:
                g = item.get("gpt_news", "")
                if g is None:
                    _log_none_strip("update_gpt_news", target_id)
                    g = ""
                if isinstance(g, str) and g.strip():
                    item["pre_gpt_news"] = g
                    item["gpt_news"] = ""
            except AttributeError as e:
                if _is_none_strip_error(e):
                    _log_none_strip("update_gpt_news", target_id)
                    item["gpt_news"] = ""
                else:
                    raise
        dict2json(news_data, json_path)

def save_metrics_to_json(eval_results, json_path):
    save_data = {
        'iteration': eval_results['iteration'],
        'pre_average': eval_results['pre_average'],
        'current_average': eval_results['current_average']
    }
    def round_dict(d, decimal=4):
        return {k: round(v, decimal) for k, v in d.items()}
    save_data['pre_average'] = round_dict(save_data['pre_average'])
    save_data['current_average'] = round_dict(save_data['current_average'])

    try:
        with file_rw_lock:
            data = json2dict(json_path) if os.path.exists(json_path) else []
            data.append(save_data)
            dict2json(data, json_path)
    except Exception as e:
        print(f"Failed to save metrics: {str(e)}")

def reset_metrics_file(json_path):
    try:
        with file_rw_lock:
            dict2json([], json_path)
        print(f"Successfully reset metrics file: {json_path}")
    except Exception as e:
        print(f"Failed to reset metrics file: {str(e)}")

@backoff.on_exception(backoff.expo, RateLimitError, max_time=60, max_tries=5)
def generate_eval_feedback(eval_prompt):
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": eval_prompt}
    ]
    response = openai_client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=messages,
        temperature=0.4,
        max_tokens=8192
    )
    return response.choices[0].message.content

@backoff.on_exception(backoff.expo, RateLimitError, max_time=60, max_tries=5)
def generate_attack_feedback(attack_prompt):
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": attack_prompt}
    ]
    response = openai_client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=messages,
        temperature=0.4,
        max_tokens=16384
    )
    return response.choices[0].message.content

def get_save_evaluation_feedback(json_path, target_id):
    try:
        try:
            example_ids, eval_prompt = make_detection_prompt(json_path, target_id)
        except AttributeError as e:
            if _is_none_strip_error(e):
                _log_none_strip("get_save_evaluation_feedback", target_id)
                return target_id, "error"
            else:
                raise

        try:
            feedback = generate_eval_feedback(eval_prompt)
        except Exception as e:
            print(f"[EvalAPI] id={target_id} call failed: {e}")
            return target_id, "error"

        found = atomic_update_json_field(json_path, target_id, {"evaluation": feedback})
        if not found:
            print(f"[Skip] ID {target_id} not found in dataset (path={json_path})")
            return target_id, "not_found"
        return target_id, "success"

    except Exception as e:
        if _is_none_strip_error(e):
            _log_none_strip("get_save_evaluation_feedback", target_id)
        print(f"Evaluation failed (ID: {target_id}): {str(e)}")
        return target_id, "error"

def get_save_attacking_feedback(json_path, target_id):
    try:
        try:
            example_ids, attack_prompt = make_attacking_prompt(json_path, target_id)
            if not attack_prompt or not attack_prompt.strip():
                print(f"[GenAPI] id={target_id} got empty attack prompt")
                return target_id, "empty_prompt"
        except AttributeError as e:
            if _is_none_strip_error(e):
                _log_none_strip("get_save_attacking_feedback", target_id)
                return target_id, "none_strip_error"
            else:
                print(f"[GenAPI] id={target_id} failed to make prompt (AttributeError): {str(e)}")
                return target_id, "prompt_attribute_error"
        except ValueError as e:
            print(f"[GenAPI] id={target_id} failed to make prompt (ValueError): {str(e)}")
            return target_id, "prompt_value_error"
        except Exception as e:
            print(f"[GenAPI] id={target_id} failed to make prompt (unknown error): {str(e)}")
            return target_id, "prompt_unknown_error"

        try:
            feedback = generate_attack_feedback(attack_prompt)
            if not feedback or not feedback.strip():
                print(f"[GenAPI] id={target_id} generated empty content")
                return target_id, "empty_content"
        except RateLimitError as e:
            print(f"[GenAPI] id={target_id} hit rate limit after retries: {str(e)}")
            return target_id, "rate_limit_error"
        except openai.APIError as e:
            print(f"[GenAPI] id={target_id} OpenAI API error: {str(e)} (code: {e.code})")
            return target_id, "openai_api_error"
        except Exception as e:
            print(f"[GenAPI] id={target_id} call failed (unknown error): {str(e)}")
            return target_id, "api_unknown_error"

        found = atomic_update_json_field(json_path, target_id, {"gpt_news": feedback})
        if not found:
            print(f"[GenAPI] id={target_id} not found in dataset")
            return target_id, "not_found"
        return target_id, "success"

    except Exception as e:
        print(f"[GenAPI] id={target_id} critical failure: {str(e)}")
        return target_id, "critical_error"
        
def sanity_check(json_path, label=""):
    try:
        with file_rw_lock:
            data = json2dict(json_path)
        total = len(data)
        non_empty_gpt = sum(1 for x in data if isinstance(x.get("gpt_news"), str) and x["gpt_news"].strip())
        non_empty_pre = sum(1 for x in data if isinstance(x.get("pre_gpt_news"), str) and x["pre_gpt_news"].strip())
        print(f"[Sanity {label}] gpt_news non-empty: {non_empty_gpt}/{total} ({(non_empty_gpt/total if total else 0):.1%}); "
              f"pre_gpt_news non-empty: {non_empty_pre}/{total} ({(non_empty_pre/total if total else 0):.1%})")
    except Exception as e:
        print(f"[Sanity {label}] check failed: {e}")

def count_status(counter: dict, status: str):
    with _error_lock:
        counter[status] = counter.get(status, 0) + 1

def Rewriter():
    result_path = 'evaluation_result.json'
    json_path = 'dataset/cnn_dailymail_debug.json'

    reset_metrics_file(json_path=result_path)
    precompute_similar_ids(json_path)

    max_workers = 16
    total_iterations = range(1, 6)
    total_ids = load_ids_from_json(json_path)
    total_rounds = len(total_iterations)
    total_samples = len(total_ids)

    print(f"Starting iterations (total: {total_rounds} rounds, parallel workers: {max_workers})")
    print(f"[Run Config] dataset='{json_path}', samples={total_samples}, id_head={total_ids[:3]}{'...' if len(total_ids)>3 else ''}")

    for iteration in tqdm(total_iterations, desc="Overall progress", unit="round"):
        print(f"\n===== Round {iteration}/{total_rounds} =====")

        # -------- Evaluation --------
        print(f"Round {iteration} - Evaluation (total: {total_samples} IDs)")
        eval_counter = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            eval_futures = [executor.submit(get_save_evaluation_feedback, json_path, id) for id in total_ids]
            with tqdm(total=total_samples, desc=f"Round {iteration} - Evaluation", unit="sample") as pbar:
                for future in as_completed(eval_futures):
                    id_, status = future.result()
                    count_status(eval_counter, status)
                    pbar.update(1)
                    pbar.set_postfix(last_id=id_, status=status)
        print(f"[Eval Summary][iter {iteration}] " + " | ".join(f"{k}:{v}" for k, v in sorted(eval_counter.items())))

        # -------- Generation --------
        print(f"Round {iteration} - Generation (total: {total_samples} IDs)")
        gen_counter = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            attack_futures = [executor.submit(get_save_attacking_feedback, json_path, id) for id in total_ids]
            with tqdm(total=total_samples, desc=f"Round {iteration} - Generation", unit="sample") as pbar:
                for future in as_completed(attack_futures):
                    id_, status = future.result()
                    count_status(gen_counter, status)
                    pbar.update(1)
                    pbar.set_postfix(last_id=id_, status=status)
        print(f"[Gen  Summary][iter {iteration}] " + " | ".join(f"{k}:{v}" for k, v in sorted(gen_counter.items())))

        sanity_check(json_path, label=f"after Generation before evaluation (iter {iteration})")

        dataset = create_dataset(json_path)
        eval_results = evaluate_text_quality(dataset, iteration=iteration)
        save_metrics_to_json(eval_results, result_path)
        update_gpt_news(json_path)
        print(f"Round {iteration}/{total_rounds} completed\n")

    plot_metrics()

if __name__ == '__main__':
    Rewriter()
