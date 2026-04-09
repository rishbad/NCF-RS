"""
=============================================================
 NCF Replication — Main Experiment Runner
 Paper: "Neural Collaborative Filtering" He et al. (2017)
=============================================================

This is the single entry point. Run this file to:
  1. Load and preprocess MovieLens 1M + Pinterest
  2. Train GMF, MLP, NeuMF (with/without pre-training)
  3. Run ablation studies:
     - Effect of embedding size (8, 16, 32, 64)
     - Effect of MLP depth (0, 1, 2, 3, 4 layers)
     - Effect of negative sampling ratio (1–10)
  4. Generate all plots and save to /plots/

Usage:
  python main.py

Runtime estimate (CPU):
  Quick run  (embed=16, epochs=10): ~15–25 min
  Full run   (embed=32, epochs=20): ~45–90 min
"""

import os
import sys
import pickle
import random
import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data.data_preprocessing import (
    load_movielens, load_pinterest, preprocess_dataset
)
from utils.trainer import train_model, run_full_experiment
from utils.metrics import evaluate_topk_curve
from utils.visualization import (
    plot_training_loss,
    plot_metrics_vs_epochs,
    plot_model_comparison,
    plot_topk_curves,
    plot_embed_size_effect,
    plot_depth_effect,
    plot_neg_sampling_effect,
    plot_pretraining_comparison,
    plot_dashboard,
)


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION — Edit these paths and settings
# ─────────────────────────────────────────────────────────────

CONFIG = {
    # ── File paths ──────────────────────────────────────────
    'movielens_path'    : 'data/raw/ratings.dat',
    'pinterest_posts'   : 'data/raw/Pinterest-posts.csv',
    'pinterest_profiles': 'data/raw/Pinterest-profiles.csv',
    'processed_dir'     : 'data/processed',
    'results_dir'       : 'results',
    'plots_dir'         : 'plots',

    # ── Hyperparameters (matching paper) ───────────────────
    'embed_dim'         : 32,      # Default predictive factors
    'num_layers'        : 3,       # Default MLP hidden layers
    'num_neg'           : 4,       # Negative samples per positive
    'batch_size'        : 256,
    'lr'                : 0.001,
    'epochs'            : 20,      # Reduce to 10 for faster runs
    'dropout'           : 0.0,
    'alpha'             : 0.5,     # Pre-training trade-off
    'top_k'             : 10,

    # ── Ablation study settings ────────────────────────────
    'embed_dims'        : [8, 16, 32, 64],     # For embed size ablation
    'layer_counts'      : [0, 1, 2, 3, 4],    # For depth ablation
    'neg_ratios'        : list(range(1, 11)),  # 1 to 10 negatives

    # ── Quick mode: fewer epochs for ablation studies ─────
    'ablation_epochs'   : 10,

    # ── Random seeds ─────────────────────────────────────
    'seed'              : 42,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> str:
    """Use GPU if available, otherwise CPU."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using: {device.upper()}")
    return device


# ─────────────────────────────────────────────────────────────
#  STEP 1: Load & Preprocess Datasets
# ─────────────────────────────────────────────────────────────

def load_data(cfg: dict) -> dict:
    """Load and preprocess both datasets. Cache to disk."""
    datasets = {}

    # MovieLens
    ml_cache = os.path.join(cfg['processed_dir'], 'processed_movielens.pkl')
    if os.path.exists(ml_cache):
        print("[MovieLens] Loading cached preprocessed data...")
        with open(ml_cache, 'rb') as f:
            datasets['movielens'] = pickle.load(f)
    else:
        ml_df = load_movielens(cfg['movielens_path'])
        datasets['movielens'] = preprocess_dataset(
            'movielens', ml_df, cfg['processed_dir']
        )

    # Pinterest
    pt_cache = os.path.join(cfg['processed_dir'], 'processed_pinterest.pkl')
    if os.path.exists(pt_cache):
        print("[Pinterest] Loading cached preprocessed data...")
        with open(pt_cache, 'rb') as f:
            datasets['pinterest'] = pickle.load(f)
    else:
        pt_df = load_pinterest(cfg['pinterest_posts'], cfg['pinterest_profiles'])
        datasets['pinterest'] = preprocess_dataset(
            'pinterest', pt_df, cfg['processed_dir']
        )

    print("\n[Data Summary]")
    for name, d in datasets.items():
        print(f"  {name:<12} | Users: {d['num_users']:,} | "
              f"Items: {d['num_items']:,} | "
              f"Sparsity: {d['sparsity']*100:.2f}%")

    return datasets


# ─────────────────────────────────────────────────────────────
#  STEP 2: Main Experiment (GMF vs MLP vs NeuMF)
# ─────────────────────────────────────────────────────────────

def run_main_experiment(datasets: dict, cfg: dict, device: str) -> dict:
    """
    Train all 4 models (GMF, MLP, NeuMF+pretrain, NeuMF scratch)
    on both datasets with default hyperparameters.
    """
    model_config = {
        'embed_dim' : cfg['embed_dim'],
        'num_layers': cfg['num_layers'],
        'num_neg'   : cfg['num_neg'],
        'batch_size': cfg['batch_size'],
        'lr'        : cfg['lr'],
        'epochs'    : cfg['epochs'],
        'dropout'   : cfg['dropout'],
        'alpha'     : cfg['alpha'],
        'top_k'     : cfg['top_k'],
    }

    all_results = {}
    all_models  = {}

    for ds_name, data in datasets.items():
        results, models = run_full_experiment(
            data        = data,
            dataset_name= ds_name,
            config      = model_config,
            device      = device,
            results_dir = cfg['results_dir']
        )
        all_results[ds_name] = results
        all_models[ds_name]  = models

    return all_results, all_models


# ─────────────────────────────────────────────────────────────
#  STEP 3: Embedding Size Ablation
# ─────────────────────────────────────────────────────────────

def run_embed_ablation(datasets: dict, cfg: dict, device: str) -> dict:
    """
    For each dataset, train GMF, MLP, NeuMF with embed_dim ∈ {8,16,32,64}.
    Reproduces Figure 4 from the paper.
    """
    print("\n" + "="*60)
    print(" ABLATION: Embedding Size")
    print("="*60)

    embed_results = {ds: {} for ds in datasets}

    for ds_name, data in datasets.items():
        embed_results[ds_name] = {'GMF': {}, 'MLP': {}, 'NeuMF': {}}

        for embed_dim in cfg['embed_dims']:
            print(f"\n[{ds_name}] embed_dim = {embed_dim}")
            ablation_config = {
                **cfg,
                'embed_dim': embed_dim,
                'epochs'   : cfg['ablation_epochs']
            }

            # Train GMF
            gmf_m, gmf_h = train_model('GMF', data, ablation_config, device, verbose=False)
            embed_results[ds_name]['GMF'][embed_dim] = {
                'HR': gmf_h['best_HR'], 'NDCG': gmf_h['best_NDCG']
            }

            # Train MLP
            mlp_m, mlp_h = train_model('MLP', data, ablation_config, device, verbose=False)
            embed_results[ds_name]['MLP'][embed_dim] = {
                'HR': mlp_h['best_HR'], 'NDCG': mlp_h['best_NDCG']
            }

            # Train NeuMF (with pre-training from above GMF + MLP)
            neumf_m, neumf_h = train_model(
                'NeuMF', data, ablation_config, device,
                pretrained_gmf=gmf_m, pretrained_mlp=mlp_m, verbose=False
            )
            embed_results[ds_name]['NeuMF'][embed_dim] = {
                'HR': neumf_h['best_HR'], 'NDCG': neumf_h['best_NDCG']
            }

            print(f"  GMF  HR={gmf_h['best_HR']:.4f}  MLP  HR={mlp_h['best_HR']:.4f}  "
                  f"NeuMF HR={neumf_h['best_HR']:.4f}")

    return embed_results


# ─────────────────────────────────────────────────────────────
#  STEP 4: MLP Depth Ablation
# ─────────────────────────────────────────────────────────────

def run_depth_ablation(datasets: dict, cfg: dict, device: str) -> dict:
    """
    Train MLP with 0, 1, 2, 3, 4 hidden layers.
    Reproduces Tables 3 and 4 from the paper.
    """
    print("\n" + "="*60)
    print(" ABLATION: MLP Depth")
    print("="*60)

    depth_results = {ds: {} for ds in datasets}

    for ds_name, data in datasets.items():
        for num_layers in cfg['layer_counts']:
            print(f"\n[{ds_name}] MLP-{num_layers}")
            ablation_config = {
                **cfg,
                'num_layers': num_layers,
                'epochs'    : cfg['ablation_epochs']
            }
            _, hist = train_model('MLP', data, ablation_config, device, verbose=False)
            depth_results[ds_name][num_layers] = {
                'HR': hist['best_HR'], 'NDCG': hist['best_NDCG']
            }
            print(f"  MLP-{num_layers}: HR={hist['best_HR']:.4f}, "
                  f"NDCG={hist['best_NDCG']:.4f}")

    return depth_results


# ─────────────────────────────────────────────────────────────
#  STEP 5: Negative Sampling Ratio Ablation
# ─────────────────────────────────────────────────────────────

def run_neg_ablation(datasets: dict, cfg: dict, device: str) -> dict:
    """
    Train GMF, MLP, NeuMF with num_neg ∈ 1..10.
    Reproduces Figure 7 from the paper.
    """
    print("\n" + "="*60)
    print(" ABLATION: Negative Sampling Ratio")
    print("="*60)

    neg_results = {ds: {} for ds in datasets}

    for ds_name, data in datasets.items():
        neg_results[ds_name] = {'GMF': {}, 'MLP': {}, 'NeuMF': {}}

        for num_neg in cfg['neg_ratios']:
            print(f"\n[{ds_name}] num_neg = {num_neg}")
            ablation_config = {
                **cfg,
                'num_neg': num_neg,
                'epochs' : cfg['ablation_epochs']
            }

            gmf_m, gmf_h = train_model('GMF',  data, ablation_config, device, verbose=False)
            mlp_m, mlp_h = train_model('MLP',  data, ablation_config, device, verbose=False)
            neumf_m, neumf_h = train_model(
                'NeuMF', data, ablation_config, device,
                pretrained_gmf=gmf_m, pretrained_mlp=mlp_m, verbose=False
            )

            for name, hist in [('GMF', gmf_h), ('MLP', mlp_h), ('NeuMF', neumf_h)]:
                neg_results[ds_name][name][num_neg] = {
                    'HR': hist['best_HR'], 'NDCG': hist['best_NDCG']
                }
            print(f"  NeuMF HR={neumf_h['best_HR']:.4f} at {num_neg} negatives")

    return neg_results


# ─────────────────────────────────────────────────────────────
#  STEP 6: Top-K Curves
# ─────────────────────────────────────────────────────────────

def compute_topk(all_models: dict, datasets: dict, cfg: dict, device: str) -> dict:
    """Compute Top-K curves (K=1..10) for all models."""
    topk_all = {}
    for ds_name, data in datasets.items():
        topk_all[ds_name] = {}
        for model_name, model in all_models[ds_name].items():
            tk = evaluate_topk_curve(
                model, data['test_df'], data['test_negatives'],
                device=device, max_k=10
            )
            topk_all[ds_name][model_name] = tk
    return topk_all


# ─────────────────────────────────────────────────────────────
#  STEP 7: Generate All Plots
# ─────────────────────────────────────────────────────────────

def generate_all_plots(all_results: dict,
                        topk_results: dict,
                        embed_results: dict,
                        depth_results: dict,
                        neg_results: dict,
                        cfg: dict):
    """Generate and save all visualizations."""

    plots_dir = cfg['plots_dir']
    print(f"\n{'='*60}")
    print(f" Generating all plots → {plots_dir}/")
    print(f"{'='*60}")

    for ds_name, results in all_results.items():
        print(f"\n[{ds_name}]")

        # 1. Training loss
        plot_training_loss(results, ds_name, plots_dir)

        # 2. HR/NDCG vs Epochs
        plot_metrics_vs_epochs(results, ds_name, plots_dir)

        # 3. Model comparison bar chart
        plot_model_comparison(results, ds_name, plots_dir)

        # 4. Pre-training comparison
        plot_pretraining_comparison(results, ds_name, plots_dir)

        # 5. Dashboard (combined)
        plot_dashboard(results, ds_name, plots_dir)

        # 6. Top-K curves
        if ds_name in topk_results:
            plot_topk_curves(topk_results[ds_name], ds_name, plots_dir)

        # 7. Embedding size ablation
        if ds_name in embed_results:
            plot_embed_size_effect(embed_results[ds_name], ds_name, plots_dir)

        # 8. Depth ablation
        if ds_name in depth_results:
            plot_depth_effect(depth_results[ds_name], ds_name, plots_dir)

        # 9. Negative sampling ablation
        if ds_name in neg_results:
            plot_neg_sampling_effect(neg_results[ds_name], ds_name, plots_dir)

    print("\n✅ All plots saved.")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    set_seed(CONFIG['seed'])
    device = get_device()

    # Ensure directories exist
    for d in [CONFIG['processed_dir'], CONFIG['results_dir'], CONFIG['plots_dir']]:
        os.makedirs(d, exist_ok=True)

    print("\n" + "="*60)
    print(" NCF Replication — He et al. (2017)")
    print("="*60)

    # ── Step 1: Load data ─────────────────────────────────────
    datasets = load_data(CONFIG)

    # ── Step 2: Main experiment ───────────────────────────────
    print("\n[Phase 2] Main Experiment: GMF vs MLP vs NeuMF")
    all_results, all_models = run_main_experiment(datasets, CONFIG, device)

    # ── Step 3: Top-K curves ──────────────────────────────────
    print("\n[Phase 3] Computing Top-K Curves...")
    topk_results = compute_topk(all_models, datasets, CONFIG, device)

    # ── Step 4: Ablation studies ──────────────────────────────
    print("\n[Phase 4] Running Ablation Studies...")
    embed_results = run_embed_ablation(datasets, CONFIG, device)
    depth_results = run_depth_ablation(datasets, CONFIG, device)
    neg_results   = run_neg_ablation(datasets, CONFIG, device)

    # ── Step 5: Save all ablation results ────────────────────
    with open(os.path.join(CONFIG['results_dir'], 'ablation_results.pkl'), 'wb') as f:
        pickle.dump({
            'embed'  : embed_results,
            'depth'  : depth_results,
            'neg'    : neg_results,
            'topk'   : topk_results,
        }, f)

    # ── Step 6: Generate all plots ────────────────────────────
    generate_all_plots(
        all_results, topk_results,
        embed_results, depth_results, neg_results,
        CONFIG
    )

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print(" FINAL RESULTS SUMMARY")
    print("="*60)
    for ds_name, results in all_results.items():
        print(f"\n  Dataset: {ds_name}")
        print(f"  {'Model':<25} {'HR@10':>8} {'NDCG@10':>10}")
        print(f"  {'─'*45}")
        for model_name, hist in results.items():
            print(f"  {model_name:<25} {hist['best_HR']:>8.4f} {hist['best_NDCG']:>10.4f}")

    print("\n✅ Replication complete!")
    print(f"   Plots saved  → {CONFIG['plots_dir']}/")
    print(f"   Results saved → {CONFIG['results_dir']}/")
