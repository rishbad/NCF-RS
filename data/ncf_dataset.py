"""
=============================================================
 NCF Replication — Dataset & DataLoader
 Paper: "Neural Collaborative Filtering" He et al. (2017)
=============================================================

This file defines:
  1. NCFDataset — PyTorch Dataset for training
     - Each sample = (user_id, item_id, label)
     - Positive label = 1  (observed interaction)
     - Negative label = 0  (sampled unobserved item)
     - Negative sampling ratio = 4 negatives per positive (paper default)

  2. get_dataloader — wraps NCFDataset in a DataLoader

Why negative sampling?
  We only have positive interactions.  The model needs to learn
  that unobserved items are less relevant.  We randomly sample
  items the user hasn't seen and label them 0 (negative).
  Ratio of 4:1 was empirically best in the paper (Figure 7).
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NCFDataset(Dataset):
    """
    PyTorch Dataset for NCF training.

    For every positive interaction (u, i):
      → create 1 positive sample: (u, i, 1)
      → create `num_neg` negative samples: (u, j, 0)
        where j is randomly sampled from items user u has NOT seen.

    This is re-sampled every epoch (dynamic negative sampling),
    which is what the paper does — "uniformly sample from
    unobserved interactions in each iteration."
    """

    def __init__(self,
                 train_df,
                 user_items: dict,
                 num_items: int,
                 num_neg: int = 4):
        """
        Args:
          train_df   : DataFrame with columns [user_id, item_id]
          user_items : dict { user_id → set of observed item_ids }
          num_items  : total number of items (for sampling range)
          num_neg    : number of negative samples per positive (paper uses 4)
        """
        self.user_items = user_items
        self.num_items  = num_items
        self.num_neg    = num_neg

        # Store positive pairs as numpy arrays for fast indexing
        self.pos_users = train_df['user_id'].values.astype(np.int64)
        self.pos_items = train_df['item_id'].values.astype(np.int64)
        self.num_pos   = len(self.pos_users)

        # Pre-generate negatives for this epoch
        # We call resample() once before training starts and once per epoch
        self.users  = None
        self.items  = None
        self.labels = None
        self.resample()

    def resample(self):
        """
        Re-generate negative samples. Call this at the start of each epoch
        to get fresh negatives (dynamic negative sampling).
        """
        all_users  = []
        all_items  = []
        all_labels = []

        for idx in range(self.num_pos):
            uid = int(self.pos_users[idx])
            iid = int(self.pos_items[idx])

            # Positive sample
            all_users.append(uid)
            all_items.append(iid)
            all_labels.append(1.0)

            # num_neg negative samples
            seen = self.user_items.get(uid, set())
            count = 0
            while count < self.num_neg:
                neg = random.randint(0, self.num_items - 1)
                if neg not in seen:
                    all_users.append(uid)
                    all_items.append(neg)
                    all_labels.append(0.0)
                    count += 1

        self.users  = np.array(all_users,  dtype=np.int64)
        self.items  = np.array(all_items,  dtype=np.int64)
        self.labels = np.array(all_labels, dtype=np.float32)

    def __len__(self):
        # Total samples = positives × (1 + num_neg)
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
          user  : LongTensor scalar (index into user embedding table)
          item  : LongTensor scalar (index into item embedding table)
          label : FloatTensor scalar (1.0 or 0.0)
        """
        return (
            torch.tensor(self.users[idx],  dtype=torch.long),
            torch.tensor(self.items[idx],  dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


def get_train_loader(train_df,
                     user_items: dict,
                     num_items: int,
                     num_neg: int = 4,
                     batch_size: int = 256,
                     num_workers: int = 0) -> tuple:
    """
    Create training DataLoader with negative sampling.

    Returns:
      dataset    : NCFDataset (so we can call dataset.resample() each epoch)
      dataloader : PyTorch DataLoader
    """
    dataset = NCFDataset(train_df, user_items, num_items, num_neg)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,           # Shuffle every epoch
        num_workers=num_workers,
        pin_memory=False
    )

    return dataset, loader
