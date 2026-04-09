"""
=============================================================
 NCF Replication — Evaluation Metrics
 Paper: "Neural Collaborative Filtering" He et al. (2017)
=============================================================

Evaluation Protocol (Section 4.1 of paper):
  - Leave-one-out: each user's last interaction = test item
  - Rank the test item among 100 randomly sampled negatives
  - Compute HR@K and NDCG@K  (paper uses K=10)

Metrics:
  HR@K (Hit Ratio):
    = 1 if the test item appears in the top-K ranked items, else 0
    Averaged over all users.
    Intuition: "Did we recommend the right item within the top K?"

  NDCG@K (Normalized Discounted Cumulative Gain):
    = 1/log2(position+1) if test item is in top-K, else 0
    Penalizes hits that appear lower in the list.
    Intuition: "How high up did we rank the correct item?"
"""

import numpy as np
import torch


def hit_ratio(ranked_list: list, test_item: int, k: int = 10) -> float:
    """
    Compute Hit Ratio @K for a single user.

    Args:
      ranked_list : list of item_ids sorted by predicted score (descending)
      test_item   : the ground-truth test item
      k           : cutoff rank

    Returns:
      1.0 if test_item is in ranked_list[:k], else 0.0
    """
    return 1.0 if test_item in ranked_list[:k] else 0.0


def ndcg(ranked_list: list, test_item: int, k: int = 10) -> float:
    """
    Compute NDCG@K for a single user.

    For a single relevant item, NDCG simplifies to:
      NDCG@K = 1 / log2(position + 2)  if item found in top-K
             = 0                         otherwise
    (position is 0-indexed, so +2 = +1 for 0-index correction + 1 for log base)

    Args:
      ranked_list : list of item_ids sorted by predicted score (descending)
      test_item   : the ground-truth test item
      k           : cutoff rank

    Returns:
      NDCG score in [0, 1]
    """
    top_k = ranked_list[:k]
    if test_item in top_k:
        position = top_k.index(test_item)    # 0-indexed position
        return 1.0 / np.log2(position + 2)  # log2(1+1)=1.0 for pos=0 (best case)
    return 0.0


@torch.no_grad()
def evaluate_model(model,
                   test_df,
                   test_negatives: dict,
                   device: str = 'cpu',
                   k: int = 10,
                   batch_size: int = 512) -> dict:
    """
    Evaluate the model using leave-one-out protocol.

    For each user:
      1. Get their test item (ground truth positive)
      2. Get their 100 sampled negative items
      3. Score all 101 items (1 positive + 100 negatives)
      4. Rank by score descending
      5. Compute HR@K and NDCG@K

    Args:
      model          : trained NCF model
      test_df        : DataFrame with columns [user_id, item_id]
      test_negatives : dict { user_id → list of 100 negative item_ids }
      device         : 'cpu' or 'cuda'
      k              : cutoff rank (paper uses 10)
      batch_size     : scoring batch size

    Returns:
      dict with keys: 'HR', 'NDCG', 'HR_list', 'NDCG_list'
    """
    model.eval()

    hr_list   = []
    ndcg_list = []

    for _, row in test_df.iterrows():
        uid       = int(row['user_id'])
        pos_item  = int(row['item_id'])
        neg_items = test_negatives.get(uid, [])

        # Build list of [positive_item] + [100 negative items]
        items_to_score = [pos_item] + neg_items   # length = 101

        # Create tensors for batch scoring
        user_tensor = torch.full((len(items_to_score),), uid,
                                  dtype=torch.long, device=device)
        item_tensor = torch.tensor(items_to_score,
                                    dtype=torch.long, device=device)

        # Score all 101 items
        scores = model(user_tensor, item_tensor)  # (101,)
        scores = scores.cpu().numpy()

        # Sort items by score descending → get ranked list
        sorted_indices = np.argsort(-scores)            # indices into items_to_score
        ranked_items   = [items_to_score[i] for i in sorted_indices]

        # Compute metrics
        hr_list.append(hit_ratio(ranked_items, pos_item, k))
        ndcg_list.append(ndcg(ranked_items, pos_item, k))

    return {
        'HR'       : np.mean(hr_list),
        'NDCG'     : np.mean(ndcg_list),
        'HR_list'  : hr_list,
        'NDCG_list': ndcg_list
    }


@torch.no_grad()
def evaluate_topk_curve(model,
                         test_df,
                         test_negatives: dict,
                         device: str = 'cpu',
                         max_k: int = 10) -> dict:
    """
    Compute HR@K and NDCG@K for K = 1, 2, ..., max_k.
    Used for Figure 5 in the paper (Top-K recommendation curves).

    Returns:
      dict with 'HR' and 'NDCG' keys, each a list of length max_k
    """
    model.eval()

    all_scores = []     # (num_users, 101) — pre-compute all scores
    all_pos    = []     # ground truth positive item per user

    for _, row in test_df.iterrows():
        uid       = int(row['user_id'])
        pos_item  = int(row['item_id'])
        neg_items = test_negatives.get(uid, [])

        items_to_score = [pos_item] + neg_items
        user_t = torch.full((len(items_to_score),), uid, dtype=torch.long, device=device)
        item_t = torch.tensor(items_to_score, dtype=torch.long, device=device)

        scores = model(user_t, item_t).cpu().numpy()
        sorted_idx    = np.argsort(-scores)
        ranked_items  = [items_to_score[i] for i in sorted_idx]

        all_scores.append(ranked_items)
        all_pos.append(pos_item)

    # Compute HR@K and NDCG@K for each K
    hr_at_k   = []
    ndcg_at_k = []

    for k in range(1, max_k + 1):
        hr_k   = np.mean([hit_ratio(ranked, pos, k)
                           for ranked, pos in zip(all_scores, all_pos)])
        ndcg_k = np.mean([ndcg(ranked, pos, k)
                           for ranked, pos in zip(all_scores, all_pos)])
        hr_at_k.append(hr_k)
        ndcg_at_k.append(ndcg_k)

    return {'HR': hr_at_k, 'NDCG': ndcg_at_k}
