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


def plot_metrics(json_path='./result/evaluation_result.json'):
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

    metrics = ['bert_score', 'sms', 'gptscore', 'g_eval_coherence', 'g_eval_consistency', 'g_eval_fluency', 'g_eval_relevance']
    metric_display_names = {
        'bert_score': 'BERTScore',
        'sms': 'SMS',
        'gptscore': 'GPTScore',
        'g_eval_coherence': 'G-Eval Coherence',
        'g_eval_consistency': 'G-Eval Consistency',
        'g_eval_fluency': 'G-Eval Fluency',
        'g_eval_relevance': 'G-Eval Relevance'
    }
    
    num_metrics = len(metrics)
    cols = 4
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
    if rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break
        ax = axes[i]
        current_values = [item['current_average'][metric] for item in metrics_data]
        ax.plot(iterations, current_values, 'o-', label='Current iteration result')

        ax.axhline(y=first_pre[metric], color='r', linestyle='--', label='First generation baseline value')

        ax.set_title(f'{metric_display_names[metric]} Metric Comparison')
        ax.set_xlabel('Iteration number')
        ax.set_ylabel('Metric value')
        ax.legend()
        ax.grid(alpha=0.3)

    for i in range(num_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('./result/metrics_visualization.png', dpi=300)
    plt.show()
    plt.close()
