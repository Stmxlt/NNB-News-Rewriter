"""
@filename:visualization.py
@author:Stmxlt
@time:2025-10-21
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['axes.unicode_minus'] = False


def plot_metrics(json_path: str = './result/evaluation_result.json'):
    if not os.path.exists(json_path):
        print("Metric data file not found, cannot visualize")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)

    if not metrics_data:
        print("Metric data is empty, cannot visualize")
        return

    iterations = [item['iteration'] for item in metrics_data]
    first_pre = metrics_data[0]['pre_average']

    # -----------------------------
    # Figure 1: G-Eval (1x3): consistency + coverage + quality
    # -----------------------------
    # Notes:
    # - coverage is stored as g_eval_relevance (mapped from coverage)
    # - quality is stored as g_eval_coherence (and also g_eval_fluency), pick one
    geval_metrics = [
        ("g_eval_consistency", "G-Eval Consistency"),
        ("g_eval_relevance", "G-Eval Coverage"),
        ("g_eval_coherence", "G-Eval Quality"),
    ]

    fig1, axes1 = plt.subplots(1, 3, figsize=(21, 5))
    if not isinstance(axes1, np.ndarray):
        axes1 = np.array([axes1])

    axes1 = axes1.flatten()

    for ax, (key, title) in zip(axes1, geval_metrics):
        cur_vals = [item['current_average'].get(key, 0.0) for item in metrics_data]
        ax.plot(iterations, cur_vals, 'o-', label='Current iteration result')

        baseline = first_pre.get(key, None)
        if baseline is not None:
            ax.axhline(y=baseline, color='r', linestyle='--', label='First generation baseline value')

        ax.set_title(title)
        ax.set_xlabel('Iteration number')
        ax.set_ylabel('Metric value')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs('./result', exist_ok=True)
    plt.savefig('./result/geval.png', dpi=300)
    plt.show()
    plt.close(fig1)

    # -----------------------------
    # Figure 2: Overall (1x4): BERTScore, SMS, GPTScore, G-Eval Average
    # -----------------------------
    overall_metrics = [
        ("bert_score", "BERTScore"),
        ("sms", "SMS"),
        ("gptscore", "GPTScore"),
        ("g_eval_average", "G-Eval Average"),
    ]

    fig2, axes2 = plt.subplots(1, 4, figsize=(28, 5))
    if not isinstance(axes1, np.ndarray):
        axes1 = np.array([axes1])

    axes1 = axes1.flatten()

    for ax, (key, title) in zip(axes2, overall_metrics):
        cur_vals = [item['current_average'].get(key, 0.0) for item in metrics_data]
        ax.plot(iterations, cur_vals, 'o-', label='Current iteration result')

        baseline = first_pre.get(key, None)
        if baseline is not None:
            ax.axhline(y=baseline, color='r', linestyle='--', label='First generation baseline value')

        ax.set_title(title)
        ax.set_xlabel('Iteration number')
        ax.set_ylabel('Metric value')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('./result/overall.png', dpi=300)
    plt.show()
    plt.close(fig2)
