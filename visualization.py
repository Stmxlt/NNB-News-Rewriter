"""
@filename:visualization.py
@author:Stmxlt
@time:2025-10-21
"""


import os
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['axes.unicode_minus'] = False


def plot_metrics(json_path='evaluation_result.json'):
    if not os.path.exists(json_path):
        print("Metric data file not found, cannot visualize")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)

    if not metrics_data:
        print("Metric data is empty, cannot visualize")
        return

    # Use default font for English
    plt.rcParams["axes.unicode_minus"] = False

    first_pre = metrics_data[0]['pre_average']
    iterations = [item['iteration'] for item in metrics_data]

    metrics = ['bert_score', 'sms', 'gptscore', 'g_eval']
    metric_display_names = {
        'bert_score': 'BERTScore',
        'sms': 'SMS',
        'gptscore': 'GPTScore',
        'g_eval': 'G-Eval'
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        current_values = [item['current_average'][metric] for item in metrics_data]
        ax.plot(iterations, current_values, 'o-', label='Current iteration result')

        ax.axhline(y=first_pre[metric], color='r', linestyle='--', label='First generation baseline value')

        ax.set_title(f'{metric_display_names[metric]} Metric Comparison')
        ax.set_xlabel('Iteration number')
        ax.set_ylabel('Metric value')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('metrics_visualization.png', dpi=300)
    plt.show()
    plt.close()
