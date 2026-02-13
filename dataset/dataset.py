# -*- coding: UTF-8 -*-#
"""
@filename:dataset.py
@author:Stmxlt
@time:2025-10-26
"""


import json
import os
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import backoff
from openai import RateLimitError
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_JSON_FILE = "cnn_dailymail.json"
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
    base_url=os.getenv("OPENAI_API_BASE", "your-api-link"),
    timeout=120,
)
PROMPT_TEMPLATE = "Please create a complete and detailed news article based on the following news summary. Keep the content coherent and logically clear, and ensure the language style complies with news reporting standards:\n{summary}"
MAX_WORKERS = 16

@backoff.on_exception(
    backoff.expo,
    RateLimitError,
    max_time=60,
    max_tries=5
)
def generate_news(summary):
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(summary=summary)}
    ]
    response = openai_client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=messages
    )
    return response.choices[0].message.content

def process_json_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Target json file not found! Path: {file_path}")
        return
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_records = len(data)
        print(f"\nProcessing target file: {file_path}")
        print(f"Total records to process: {total_records}")
        
        tasks = [
            (idx, item.get("summary", ""))
            for idx, item in enumerate(data)
        ]
        
        results = data.copy()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {
                executor.submit(generate_news, summary): (idx, summary)
                for idx, summary in tasks
            }
            
            for future in as_completed(future_to_task):
                idx, summary = future_to_task[future]
                try:
                    machine_news = future.result()
                    print(f"Generated news {idx + 1}/{total_records}")
                    results[idx]["machine_news"] = machine_news
                except Exception as e:
                    error_msg = f"Generation failed: {str(e)}"
                    results[idx]["machine_news"] = error_msg
                    print(f"Failed to generate record {idx + 1}: {str(e)}")
        
        output_json = "cnn_dailymail_updated.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nProcessing completed! Saved {len(results)} records to: {output_json}")
    except Exception as e:
        print(f"Error processing target file: {str(e)}")

if __name__ == "__main__":
    print("Starting dataset processing...")
    process_json_file(TARGET_JSON_FILE)
    print("Dataset processing finished!")
