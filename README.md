# NCF Replication — He et al. (2017)
## Neural Collaborative Filtering

Complete replication of the paper:
> He et al., "Neural Collaborative Filtering", WWW 2017

---

## Project Structure

```
ncf_replication/
│
├── setup.py                    ← Run this FIRST
├── main.py                     ← Run this to train + evaluate + plot
│
├── data/
│   ├── data_preprocessing.py   ← Loads + preprocesses both datasets
│   ├── ncf_dataset.py          ← PyTorch Dataset with negative sampling
│   ├── raw/                    ← PUT YOUR RAW DATA FILES HERE
│   │   ├── ratings.dat
│   │   ├── Pinterest-posts.csv
│   │   └── Pinterest-profiles.csv
│   └── processed/              ← Auto-generated cached data
│
├── models/
│   └── ncf_models.py           ← GMF, MLP, NeuMF implementations
│
├── utils/
│   ├── metrics.py              ← HR@10, NDCG@10 evaluation
│   ├── trainer.py              ← Training loop + experiment runner
│   └── visualization.py       ← All plotting functions
│
├── results/                    ← Auto-generated .pkl result files
└── plots/                      ← Auto-generated plot images
```

---

## Setup

### Step 1 — Install dependencies
```bash
pip install torch numpy pandas matplotlib scikit-learn tqdm
```

### Step 2 — Copy raw data files
Copy the following files into `data/raw/`:
```
ratings.dat              (MovieLens 1M)
Pinterest-posts.csv
Pinterest-profiles.csv
```

### Step 3 — Verify setup
```bash
python setup.py
```

### Step 4 — Run
```bash
python main.py
```

---

## What Gets Trained

| Model | Description |
|---|---|
| **GMF** | Generalized Matrix Factorization — learns weighted dot product |
| **MLP** | Multi-Layer Perceptron — learns non-linear interactions |
| **NeuMF (pre-trained)** | Fusion of GMF+MLP initialized with pre-trained weights |
| **NeuMF (scratch)** | Same fusion model but trained from random init |

---

## Plots Generated

| File | Description |
|---|---|
| `*_dashboard.png` | Combined overview of all metrics |
| `*_training_loss.png` | BCE loss per epoch |
| `*_metrics_vs_epochs.png` | HR@10 and NDCG@10 per epoch |
| `*_model_comparison.png` | Bar chart of best results |
| `*_topk_curves.png` | HR@K and NDCG@K for K=1..10 |
| `*_embed_size.png` | Effect of embedding dimension |
| `*_depth_effect.png` | Effect of MLP depth (0–4 layers) |
| `*_neg_sampling.png` | Effect of negative sampling ratio |
| `*_pretraining.png` | Pre-training vs no pre-training |

---

## Expected Results (Paper)

| Dataset | Model | HR@10 | NDCG@10 |
|---|---|---|---|
| MovieLens | GMF | ~0.70 | ~0.42 |
| MovieLens | MLP | ~0.69 | ~0.42 |
| MovieLens | NeuMF | **0.726–0.730** | **0.445–0.447** |
| Pinterest | NeuMF | **0.877–0.880** | **0.552–0.558** |

---

## Adjusting Speed vs Accuracy

In `main.py`, edit `CONFIG`:
```python
'epochs'         : 10,   # Faster (paper uses 20)
'ablation_epochs': 5,    # Even faster for ablation studies
'embed_dim'      : 16,   # Smaller = faster (paper default 32)
```

---

## Key Paper Findings to Verify

1. **NeuMF > MLP > GMF** — NeuMF best on both datasets
2. **Pre-training helps** — NeuMF with pre-training > without
3. **Deeper is better** — More MLP layers → better performance
4. **Optimal negatives ≈ 4** — Too few or too many hurts
5. **Log loss > squared loss** — Binary classification framing works

---

## Architecture Reference

```
GMF:
  User_emb(K) ──┐
                ├── ⊙ element-wise product ──► linear(K→1) ──► sigmoid
  Item_emb(K) ──┘

MLP:
  User_emb(2K) ──┐
                 ├── concat(4K) ──► FC(ReLU) ──► FC(ReLU) ──► ... ──► sigmoid
  Item_emb(2K) ──┘

NeuMF:
  GMF_user ── GMF_item ──► ⊙ ──────────────────────┐
  MLP_user ── MLP_item ──► concat ──► FC×N ─────────┤
                                                     ├── concat ──► FC(1) ──► sigmoid
```
