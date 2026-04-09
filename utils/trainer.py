"""
=============================================================
 NCF Replication — Training Engine
 Paper: "Neural Collaborative Filtering" He et al. (2017)
=============================================================

This file handles the full training loop for all three models.

Key training details from the paper:
  - Loss    : Binary Cross-Entropy (log loss)  — Equation 7
  - GMF/MLP : Adam optimizer  (lr tested in {0.0001, 0.0005, 0.001, 0.005})
  - NeuMF   : Pre-train with Adam, then fine-tune with vanilla SGD
              (Adam requires momentum info that's lost after pre-training)
  - Batch   : Tested {128, 256, 512, 1024}  — we use 256 default
  - Negatives: 4 negatives per positive (re-sampled each epoch)
  - Best model tracked by HR@10 on test set (evaluated each epoch)
"""

import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

# Import from our other modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ncf_models import GMF, MLP, NeuMF, build_model
from utils.metrics import evaluate_model
from data.ncf_dataset import get_train_loader


def train_epoch(model: nn.Module,
                loader,
                optimizer,
                criterion,
                device: str) -> float:
    """
    Run one training epoch.

    Args:
      model     : NCF model
      loader    : DataLoader (yields user, item, label batches)
      optimizer : Adam or SGD
      criterion : BCELoss
      device    : 'cpu' or 'cuda'

    Returns:
      avg_loss : average training loss over all batches
    """
    model.train()
    total_loss  = 0.0
    total_batch = 0

    for user_ids, item_ids, labels in loader:
        # Move to device
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        labels   = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(user_ids, item_ids)   # (batch,) in (0,1)

        # Compute binary cross-entropy loss
        # = -[y*log(p) + (1-y)*log(1-p)]
        loss = criterion(predictions, labels)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        total_batch += 1

    return total_loss / total_batch


def train_model(model_type: str,
                data: dict,
                config: dict,
                device: str = 'cpu',
                pretrained_gmf=None,
                pretrained_mlp=None,
                verbose: bool = True) -> tuple:
    """
    Full training pipeline for GMF, MLP, or NeuMF.

    Args:
      model_type    : 'GMF', 'MLP', or 'NeuMF'
      data          : preprocessed data dict from preprocessing pipeline
      config        : hyperparameter dict (see defaults below)
      device        : 'cpu' or 'cuda'
      pretrained_gmf: pre-trained GMF model (only for NeuMF)
      pretrained_mlp: pre-trained MLP model (only for NeuMF)
      verbose       : print progress

    Returns:
      model         : trained model
      history       : dict of per-epoch metrics
    """

    # ── Default hyperparameters (from paper) ──────────────────────
    cfg = {
        'embed_dim'  : 32,      # Predictive factors K  ∈ {8,16,32,64}
        'num_layers' : 3,       # MLP hidden layers     ∈ {0,1,2,3,4}
        'num_neg'    : 4,       # Negatives per positive
        'batch_size' : 256,     # Mini-batch size
        'lr'         : 0.001,   # Learning rate
        'epochs'     : 20,      # Training epochs
        'dropout'    : 0.0,     # Dropout (not used in paper)
        'alpha'      : 0.5,     # Pre-training trade-off for NeuMF
        'top_k'      : 10,      # HR@K and NDCG@K
    }
    cfg.update(config)          # Override with user-provided config

    num_users = data['num_users']
    num_items = data['num_items']
    train_df  = data['train_df']
    test_df   = data['test_df']
    user_items= data['user_items']
    test_neg  = data['test_negatives']

    # ── Build model ───────────────────────────────────────────────
    model = build_model(
        model_type = model_type,
        num_users  = num_users,
        num_items  = num_items,
        embed_dim  = cfg['embed_dim'],
        num_layers = cfg['num_layers'],
        dropout    = cfg['dropout']
    ).to(device)

    # ── Pre-training for NeuMF ────────────────────────────────────
    if model_type.upper() == 'NEUMF' and pretrained_gmf and pretrained_mlp:
        print("[NeuMF] Loading pre-trained weights...")
        model.load_pretrained(pretrained_gmf, pretrained_mlp, alpha=cfg['alpha'])
        # Paper: fine-tune NeuMF with vanilla SGD after pre-training
        optimizer = SGD(model.parameters(), lr=cfg['lr'])
    else:
        # GMF and MLP use Adam; NeuMF from scratch also uses Adam
        optimizer = Adam(model.parameters(), lr=cfg['lr'])

    # ── Loss function: Binary Cross-Entropy ──────────────────────
    # This is the log loss from Equation 7 in the paper
    criterion = nn.BCELoss()

    # ── DataLoader ────────────────────────────────────────────────
    dataset, loader = get_train_loader(
        train_df   = train_df,
        user_items = user_items,
        num_items  = num_items,
        num_neg    = cfg['num_neg'],
        batch_size = cfg['batch_size']
    )

    # ── Training loop ─────────────────────────────────────────────
    history = {
        'train_loss': [],
        'HR'        : [],
        'NDCG'      : [],
        'epoch_time': []
    }

    best_hr    = 0.0
    best_ndcg  = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()

        # Re-sample negatives at the start of each epoch
        # (dynamic negative sampling — matches paper)
        dataset.resample()

        # Train one epoch
        avg_loss = train_epoch(model, loader, optimizer, criterion, device)

        # Evaluate on test set
        metrics = evaluate_model(
            model, test_df, test_neg, device=device, k=cfg['top_k']
        )

        hr   = metrics['HR']
        ndcg = metrics['NDCG']
        elapsed = time.time() - t0

        # Track best model (by HR@10)
        if hr > best_hr:
            best_hr    = hr
            best_ndcg  = ndcg
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Store history
        history['train_loss'].append(avg_loss)
        history['HR'].append(hr)
        history['NDCG'].append(ndcg)
        history['epoch_time'].append(elapsed)

        if verbose:
            print(f"  [{model_type}] Epoch {epoch:3d}/{cfg['epochs']} | "
                  f"Loss: {avg_loss:.4f} | HR@{cfg['top_k']}: {hr:.4f} | "
                  f"NDCG@{cfg['top_k']}: {ndcg:.4f} | Time: {elapsed:.1f}s")

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    if verbose:
        print(f"\n  Best @ Epoch {best_epoch}: "
              f"HR@{cfg['top_k']}={best_hr:.4f}, NDCG@{cfg['top_k']}={best_ndcg:.4f}")

    history['best_HR']    = best_hr
    history['best_NDCG']  = best_ndcg
    history['best_epoch'] = best_epoch

    return model, history


def run_full_experiment(data: dict,
                        dataset_name: str,
                        config: dict,
                        device: str = 'cpu',
                        results_dir: str = 'results') -> dict:
    """
    Run the complete NCF experiment:
      1. Train GMF
      2. Train MLP
      3. Train NeuMF with pre-training (using GMF + MLP weights)
      4. Train NeuMF without pre-training (ablation)
      5. Save all results

    Returns:
      all_results : dict of histories per model
    """
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    print(f"\n{'='*60}")
    print(f" Dataset: {dataset_name}")
    print(f" Config : embed_dim={config.get('embed_dim',32)}, "
          f"layers={config.get('num_layers',3)}")
    print(f"{'='*60}")

    # ── Step 1: Train GMF ─────────────────────────────────────────
    print(f"\n[1/4] Training GMF...")
    gmf_model, gmf_hist = train_model('GMF', data, config, device)
    all_results['GMF'] = gmf_hist

    # ── Step 2: Train MLP ─────────────────────────────────────────
    print(f"\n[2/4] Training MLP...")
    mlp_model, mlp_hist = train_model('MLP', data, config, device)
    all_results['MLP'] = mlp_hist

    # ── Step 3: Train NeuMF with pre-training ────────────────────
    print(f"\n[3/4] Training NeuMF (WITH pre-training)...")
    neumf_pt_model, neumf_pt_hist = train_model(
        'NeuMF', data, config, device,
        pretrained_gmf=gmf_model,
        pretrained_mlp=mlp_model
    )
    all_results['NeuMF_pretrained'] = neumf_pt_hist

    # ── Step 4: Train NeuMF without pre-training (ablation) ───────
    print(f"\n[4/4] Training NeuMF (WITHOUT pre-training)...")
    neumf_model, neumf_hist = train_model(
        'NeuMF', data, config, device,
        pretrained_gmf=None,
        pretrained_mlp=None
    )
    all_results['NeuMF_scratch'] = neumf_hist

    # ── Save results ─────────────────────────────────────────────
    out_path = os.path.join(results_dir, f"results_{dataset_name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n✅ Results saved → {out_path}")

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  SUMMARY — {dataset_name}")
    print(f"{'─'*55}")
    for name, hist in all_results.items():
        print(f"  {name:<22} HR@10={hist['best_HR']:.4f}  "
              f"NDCG@10={hist['best_NDCG']:.4f}")

    return all_results, {
        'GMF'             : gmf_model,
        'MLP'             : mlp_model,
        'NeuMF_pretrained': neumf_pt_model,
        'NeuMF_scratch'   : neumf_model
    }
