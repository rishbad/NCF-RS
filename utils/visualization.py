"""
=============================================================
 NCF Replication — Visualizations
 Paper: "Neural Collaborative Filtering" He et al. (2017)
=============================================================

Generates all plots from the paper and additional analysis:

  1.  Training loss curves (per model)
  2.  HR@10 vs Epochs
  3.  NDCG@10 vs Epochs
  4.  Model comparison bar chart (GMF vs MLP vs NeuMF)
  5.  Top-K curves (K = 1 to 10)  — reproduces Figure 5
  6.  Effect of embedding size     — reproduces Figure 4
  7.  Effect of MLP depth          — reproduces Table 3 / 4
  8.  Effect of negative sampling ratio — reproduces Figure 7
  9.  Pre-training vs No Pre-training comparison
  10. Combined training/validation loss per epoch
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Consistent color palette across all plots ───────────────
COLORS = {
    'GMF'              : '#2196F3',   # Blue
    'MLP'              : '#FF9800',   # Orange
    'NeuMF_pretrained' : '#4CAF50',   # Green
    'NeuMF_scratch'    : '#9C27B0',   # Purple
    'NeuMF'            : '#4CAF50',   # Alias
    'ItemKNN'          : '#F44336',   # Red
    'BPR'              : '#00BCD4',   # Cyan
    'eALS'             : '#795548',   # Brown
    'ItemPop'          : '#607D8B',   # Grey
}

LINE_STYLES = {
    'GMF'              : '-',
    'MLP'              : '--',
    'NeuMF_pretrained' : '-.',
    'NeuMF_scratch'    : ':',
    'NeuMF'            : '-.',
}

MARKERS = {
    'GMF'              : 'o',
    'MLP'              : 's',
    'NeuMF_pretrained' : '^',
    'NeuMF_scratch'    : 'D',
    'NeuMF'            : '^',
}

def _save(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
#  PLOT 1: Training Loss Curves
# ─────────────────────────────────────────────

def plot_training_loss(results: dict,
                       dataset_name: str,
                       save_dir: str = 'plots'):
    """
    Plot training loss per epoch for GMF, MLP, NeuMF.
    Reproduces Figure 6a from the paper.

    What to look for:
      - NeuMF should converge to lowest loss (most expressive model)
      - Loss should decrease quickly in first ~10 epochs, then plateau
      - Model that over-fits will show loss decreasing but HR declining
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, hist in results.items():
        if 'train_loss' not in hist:
            continue
        label = _model_label(name)
        ax.plot(hist['train_loss'],
                color=COLORS.get(name, 'grey'),
                linestyle=LINE_STYLES.get(name, '-'),
                linewidth=2,
                label=label)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Training Loss (Binary Cross-Entropy)', fontsize=13)
    ax.set_title(f'Training Loss — {dataset_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f'{save_dir}/{dataset_name}_training_loss.png')


# ─────────────────────────────────────────────
#  PLOT 2: HR@10 and NDCG@10 vs Epochs
# ─────────────────────────────────────────────

def plot_metrics_vs_epochs(results: dict,
                            dataset_name: str,
                            save_dir: str = 'plots'):
    """
    Plot HR@10 and NDCG@10 over training epochs.
    Reproduces the trend in Figure 6b, 6c.

    What to look for:
      - Sharp improvement in first 10–15 epochs
      - NeuMF achieves higher plateau than GMF/MLP
      - Overfitting: HR may decrease at later epochs even as loss falls
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in results.items():
        label = _model_label(name)
        c  = COLORS.get(name, 'grey')
        ls = LINE_STYLES.get(name, '-')

        axes[0].plot(hist['HR'], color=c, linestyle=ls, linewidth=2, label=label)
        axes[1].plot(hist['NDCG'], color=c, linestyle=ls, linewidth=2, label=label)

    for ax, metric in zip(axes, ['HR@10', 'NDCG@10']):
        ax.set_xlabel('Epoch', fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(f'{metric} vs Epochs — {dataset_name}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f'{save_dir}/{dataset_name}_metrics_vs_epochs.png')


# ─────────────────────────────────────────────
#  PLOT 3: Model Comparison Bar Chart
# ─────────────────────────────────────────────

def plot_model_comparison(results: dict,
                           dataset_name: str,
                           save_dir: str = 'plots'):
    """
    Bar chart comparing best HR@10 and NDCG@10 across all models.
    Quick visual summary of which model performs best.

    What to look for:
      - NeuMF should clearly outperform GMF and MLP
      - Pre-training should help NeuMF (taller bar than from-scratch)
    """
    models  = list(results.keys())
    labels  = [_model_label(m) for m in models]
    hr_vals   = [results[m]['best_HR']   for m in models]
    ndcg_vals = [results[m]['best_NDCG'] for m in models]
    colors    = [COLORS.get(m, '#607D8B') for m in models]

    x = np.arange(len(models))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, vals, metric in zip(axes, [hr_vals, ndcg_vals], ['HR@10', 'NDCG@10']):
        bars = ax.bar(x, vals, color=colors, width=0.5, edgecolor='white', linewidth=1.2)

        # Annotate bars with values
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(f'{metric} Comparison — {dataset_name}',
                     fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.15)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, f'{save_dir}/{dataset_name}_model_comparison.png')


# ─────────────────────────────────────────────
#  PLOT 4: Top-K Curves (K = 1 to 10)
# ─────────────────────────────────────────────

def plot_topk_curves(topk_results: dict,
                      dataset_name: str,
                      save_dir: str = 'plots'):
    """
    Plot HR@K and NDCG@K for K = 1 to 10.
    Reproduces Figure 5 from the paper.

    topk_results: dict { model_name → {'HR': list[10], 'NDCG': list[10]} }

    What to look for:
      - All models improve as K increases (more chances to hit)
      - NeuMF consistently above all others at every K
      - Gap between models most visible at small K (e.g., K=1)
    """
    K = range(1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, tk in topk_results.items():
        label = _model_label(name)
        c     = COLORS.get(name, 'grey')
        ls    = LINE_STYLES.get(name, '-')
        mk    = MARKERS.get(name, 'o')

        axes[0].plot(K, tk['HR'],   color=c, linestyle=ls, marker=mk,
                     linewidth=2, markersize=5, label=label)
        axes[1].plot(K, tk['NDCG'], color=c, linestyle=ls, marker=mk,
                     linewidth=2, markersize=5, label=label)

    for ax, metric in zip(axes, ['HR@K', 'NDCG@K']):
        ax.set_xlabel('K', fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(f'Top-K Recommendation — {dataset_name}',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(list(K))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f'{save_dir}/{dataset_name}_topk_curves.png')


# ─────────────────────────────────────────────
#  PLOT 5: Effect of Embedding Size
# ─────────────────────────────────────────────

def plot_embed_size_effect(embed_results: dict,
                            dataset_name: str,
                            save_dir: str = 'plots'):
    """
    Plot HR@10 and NDCG@10 vs embedding size (8, 16, 32, 64).
    Reproduces Figure 4 from the paper.

    embed_results: dict {
        model_name: {
            embed_dim: {'HR': value, 'NDCG': value}
        }
    }

    What to look for:
      - Performance generally improves with more factors
      - Too many factors can overfit (performance drops at 64 for some models)
      - NeuMF maintains advantage across all factor sizes
    """
    embed_dims = [8, 16, 32, 64]
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5))

    for name, dim_results in embed_results.items():
        label = _model_label(name)
        c     = COLORS.get(name, 'grey')
        ls    = LINE_STYLES.get(name, '-')
        mk    = MARKERS.get(name, 'o')

        hrs   = [dim_results.get(d, {}).get('HR', 0)   for d in embed_dims]
        ndcgs = [dim_results.get(d, {}).get('NDCG', 0) for d in embed_dims]

        axes[0].plot(embed_dims, hrs,   color=c, linestyle=ls, marker=mk,
                     linewidth=2, markersize=6, label=label)
        axes[1].plot(embed_dims, ndcgs, color=c, linestyle=ls, marker=mk,
                     linewidth=2, markersize=6, label=label)

    for ax, metric in zip(axes, ['HR@10', 'NDCG@10']):
        ax.set_xlabel('Embedding Size (Predictive Factors)', fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(f'{metric} vs Embedding Size — {dataset_name}',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(embed_dims)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f'{save_dir}/{dataset_name}_embed_size.png')


# ─────────────────────────────────────────────
#  PLOT 6: Effect of MLP Depth
# ─────────────────────────────────────────────

def plot_depth_effect(depth_results: dict,
                       dataset_name: str,
                       save_dir: str = 'plots'):
    """
    Plot HR@10 and NDCG@10 vs number of MLP hidden layers (0–4).
    Reproduces Tables 3 and 4 from the paper.

    depth_results: dict {
        num_layers: {'HR': value, 'NDCG': value}
    }

    What to look for:
      - MLP-0 performs poorly (no non-linearity → almost random)
      - Performance increases with more layers
      - Diminishing returns after 3–4 layers
      - Paper conclusion: deeper IS better for recommendation
    """
    layers  = sorted(depth_results.keys())
    hrs     = [depth_results[l]['HR']   for l in layers]
    ndcgs   = [depth_results[l]['NDCG'] for l in layers]
    labels  = [f'MLP-{l}' for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(layers))

    colors_depth = ['#EF9A9A', '#EF5350', '#E53935', '#B71C1C', '#7B1111']

    for ax, vals, metric in zip(axes, [hrs, ndcgs], ['HR@10', 'NDCG@10']):
        bars = ax.bar(x, vals,
                      color=colors_depth[:len(layers)],
                      edgecolor='white', linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(f'Effect of Depth on {metric} — {dataset_name}',
                     fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, f'{save_dir}/{dataset_name}_depth_effect.png')


# ─────────────────────────────────────────────
#  PLOT 7: Effect of Negative Sampling Ratio
# ─────────────────────────────────────────────

def plot_neg_sampling_effect(neg_results: dict,
                              dataset_name: str,
                              save_dir: str = 'plots'):
    """
    Plot HR@10 and NDCG@10 vs number of negatives per positive (1–10).
    Reproduces Figure 7 from the paper.

    neg_results: dict {
        model_name: {
            num_neg: {'HR': value, 'NDCG': value}
        }
    }

    What to look for:
      - 1 negative is insufficient for good performance
      - Optimal ratio ≈ 3–6 (paper finding)
      - Too many negatives (> 7) can hurt performance
      - All NCF methods outperform BPR (which uses only 1 negative)
    """
    num_negs = sorted(list(next(iter(neg_results.values())).keys()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, neg_dict in neg_results.items():
        label = _model_label(name)
        c  = COLORS.get(name, 'grey')
        ls = LINE_STYLES.get(name, '-')
        mk = MARKERS.get(name, 'o')

        hrs   = [neg_dict.get(n, {}).get('HR', 0)   for n in num_negs]
        ndcgs = [neg_dict.get(n, {}).get('NDCG', 0) for n in num_negs]

        axes[0].plot(num_negs, hrs,   color=c, linestyle=ls, marker=mk,
                     linewidth=2, markersize=5, label=label)
        axes[1].plot(num_negs, ndcgs, color=c, linestyle=ls, marker=mk,
                     linewidth=2, markersize=5, label=label)

    for ax, metric in zip(axes, ['HR@10', 'NDCG@10']):
        ax.set_xlabel('Number of Negatives per Positive', fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(f'{metric} vs Negative Sampling Ratio — {dataset_name}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f'{save_dir}/{dataset_name}_neg_sampling.png')


# ─────────────────────────────────────────────
#  PLOT 8: Pre-training vs No Pre-training
# ─────────────────────────────────────────────

def plot_pretraining_comparison(results: dict,
                                 dataset_name: str,
                                 save_dir: str = 'plots'):
    """
    Side-by-side comparison of NeuMF with and without pre-training.
    Reproduces Table 2 from the paper.

    What to look for:
      - Pre-trained NeuMF should reach higher HR@10 and NDCG@10
      - The gain from pre-training demonstrates importance of good initialization
    """
    if 'NeuMF_pretrained' not in results or 'NeuMF_scratch' not in results:
        print("  Skipping pre-training comparison (missing data)")
        return

    pt_hr    = results['NeuMF_pretrained']['HR']
    no_pt_hr = results['NeuMF_scratch']['HR']
    pt_ndcg    = results['NeuMF_pretrained']['NDCG']
    no_pt_ndcg = results['NeuMF_scratch']['NDCG']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(pt_hr) + 1)

    for ax, pt_vals, no_pt_vals, metric in zip(
        axes,
        [pt_hr, pt_ndcg],
        [no_pt_hr, no_pt_ndcg],
        ['HR@10', 'NDCG@10']
    ):
        ax.plot(epochs, pt_vals,    color='#4CAF50', linewidth=2,
                label='NeuMF (With Pre-training)')
        ax.plot(epochs, no_pt_vals, color='#9C27B0', linewidth=2,
                linestyle='--', label='NeuMF (Without Pre-training)')

        ax.set_xlabel('Epoch', fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(f'Pre-training Effect on {metric} — {dataset_name}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f'{save_dir}/{dataset_name}_pretraining.png')


# ─────────────────────────────────────────────
#  PLOT 9: Combined Dashboard
# ─────────────────────────────────────────────

def plot_dashboard(results: dict,
                   dataset_name: str,
                   save_dir: str = 'plots'):
    """
    One comprehensive figure showing all key metrics:
      Row 1: Training Loss | HR@10 vs Epochs | NDCG@10 vs Epochs
      Row 2: Model comparison bar (HR) | Model comparison bar (NDCG) | Pre-training
    """
    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])   # Training loss
    ax2 = fig.add_subplot(gs[0, 1])   # HR@10 vs epochs
    ax3 = fig.add_subplot(gs[0, 2])   # NDCG@10 vs epochs
    ax4 = fig.add_subplot(gs[1, 0])   # Bar chart HR
    ax5 = fig.add_subplot(gs[1, 1])   # Bar chart NDCG
    ax6 = fig.add_subplot(gs[1, 2])   # Pre-training comparison

    # ── Loss and epoch curves ──────────────────────────────────────
    for name, hist in results.items():
        label = _model_label(name)
        c  = COLORS.get(name, 'grey')
        ls = LINE_STYLES.get(name, '-')

        ax1.plot(hist['train_loss'], color=c, linestyle=ls, linewidth=2, label=label)
        ax2.plot(hist['HR'],         color=c, linestyle=ls, linewidth=2, label=label)
        ax3.plot(hist['NDCG'],       color=c, linestyle=ls, linewidth=2, label=label)

    for ax, title in zip([ax1, ax2, ax3],
                          ['Training Loss', 'HR@10 vs Epochs', 'NDCG@10 vs Epochs']):
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Epoch', fontsize=10)

    # ── Bar charts ────────────────────────────────────────────────
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    hr_vals   = [results[m]['best_HR']   for m in model_names]
    ndcg_vals = [results[m]['best_NDCG'] for m in model_names]
    bar_colors = [COLORS.get(m, '#607D8B') for m in model_names]

    ax4.bar(x, hr_vals, color=bar_colors, edgecolor='white')
    ax5.bar(x, ndcg_vals, color=bar_colors, edgecolor='white')

    for ax, vals, metric in zip([ax4, ax5], [hr_vals, ndcg_vals], ['HR@10', 'NDCG@10']):
        ax.set_xticks(x)
        ax.set_xticklabels([_model_label(m) for m in model_names],
                            fontsize=8, rotation=20, ha='right')
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f'Best {metric} Comparison', fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    # ── Pre-training ───────────────────────────────────────────────
    if 'NeuMF_pretrained' in results and 'NeuMF_scratch' in results:
        epochs = range(1, len(results['NeuMF_pretrained']['HR']) + 1)
        ax6.plot(epochs, results['NeuMF_pretrained']['HR'],
                 color='#4CAF50', linewidth=2, label='With Pre-training')
        ax6.plot(epochs, results['NeuMF_scratch']['HR'],
                 color='#9C27B0', linewidth=2, linestyle='--', label='No Pre-training')
        ax6.set_title('NeuMF: Pre-training Effect', fontweight='bold', fontsize=11)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.set_xlabel('Epoch', fontsize=10)
        ax6.set_ylabel('HR@10', fontsize=10)

    fig.suptitle(f'NCF Experiment Dashboard — {dataset_name}',
                 fontsize=16, fontweight='bold', y=1.01)

    _save(fig, f'{save_dir}/{dataset_name}_dashboard.png', dpi=150)


# ─────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────

def _model_label(name: str) -> str:
    label_map = {
        'GMF'              : 'GMF',
        'MLP'              : 'MLP',
        'NeuMF_pretrained' : 'NeuMF (pre-trained)',
        'NeuMF_scratch'    : 'NeuMF (scratch)',
        'NeuMF'            : 'NeuMF',
    }
    return label_map.get(name, name)
