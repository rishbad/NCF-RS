"""
=============================================================
 NCF Replication — Phase 1: Data Preprocessing
 Paper: "Neural Collaborative Filtering" He et al. (2017)
=============================================================

What this file does:
  1. Loads MovieLens 1M and Pinterest datasets
  2. Converts them to implicit feedback (0/1 interaction matrix)
  3. Applies leave-one-out split (last interaction = test)
  4. Builds negative samples for evaluation (100 per user)
  5. Saves processed data as .pkl files for training

Usage:
  python data_preprocessing.py
"""

import os
import random
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict


# ─────────────────────────────────────────────
#  SECTION 1: Load MovieLens 1M
# ─────────────────────────────────────────────

def load_movielens(ratings_path: str) -> pd.DataFrame:
    """
    Load MovieLens 1M ratings file.
    Format per line: UserID::MovieID::Rating::Timestamp

    We only need UserID, MovieID, Timestamp.
    Rating is ignored — we convert to implicit (any rating = interaction).
    """
    print("[MovieLens] Loading ratings...")
    df = pd.read_csv(
        ratings_path,
        sep='::',
        engine='python',         # needed for multi-char separator
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        dtype={'user_id': int, 'item_id': int, 'rating': float, 'timestamp': int}
    )

    print(f"  Raw interactions : {len(df):,}")
    print(f"  Unique users     : {df['user_id'].nunique():,}")
    print(f"  Unique items     : {df['item_id'].nunique():,}")

    # Convert to implicit feedback: any interaction → label 1
    # We drop duplicate (user, item) pairs just in case
    df = df.drop_duplicates(subset=['user_id', 'item_id'])

    return df[['user_id', 'item_id', 'timestamp']]


# ─────────────────────────────────────────────
#  SECTION 2: Load Pinterest Dataset
# ─────────────────────────────────────────────

def load_pinterest(posts_path: str, profiles_path: str) -> pd.DataFrame:
    """
    Build implicit feedback from Pinterest data.

    The posts CSV has: user_id, post_id (one row = one pin/interaction).
    We treat each (user_id, post_id) pair as an implicit interaction.

    We also augment interactions using the profiles 'saved' boards:
    Each board a user has created = interaction with that board topic.
    This gives richer data similar to the original Pinterest dataset used
    in the paper (which was user-pin interactions).

    After loading, we filter users with < 5 interactions (the uploaded
    dataset is small, so we relax the paper's threshold of 20).
    """
    print("[Pinterest] Loading posts data...")
    posts = pd.read_csv(posts_path, dtype=str)

    # Build user-post interaction dataframe
    posts = posts[['user_id', 'post_id']].dropna()
    posts.columns = ['user_id', 'item_id']

    # Re-index to integer IDs
    posts['user_id'] = posts['user_id'].astype(str)
    posts['item_id'] = posts['item_id'].astype(str)

    # ── Augment with profile board data ──────────────────────────────
    # Each user's saved boards are (user, board_name) implicit interactions
    # This simulates the richness of the original NCF Pinterest dataset
    import json
    print("[Pinterest] Loading profile board interactions...")
    profiles = pd.read_csv(profiles_path, dtype=str)

    board_rows = []
    for _, row in profiles.iterrows():
        uid = str(row['profile_id'])
        try:
            boards = json.loads(row['saved'])
            for b in boards:
                board_name = b.get('name', '')
                pin_count  = int(b.get('pins', 1))
                # Each board = one interaction with that "item" (board URL as ID)
                board_url = b.get('saved_collection_url', board_name)
                board_rows.append({'user_id': uid, 'item_id': board_url})
        except Exception:
            pass

    board_df = pd.DataFrame(board_rows)
    print(f"  Board interactions extracted: {len(board_df):,}")

    # Combine posts + board interactions
    combined = pd.concat([posts, board_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['user_id', 'item_id'])

    # ── Remap IDs to consecutive integers (required for embedding layers) ──
    user_ids  = combined['user_id'].unique()
    item_ids  = combined['item_id'].unique()
    user2idx  = {u: i for i, u in enumerate(user_ids)}
    item2idx  = {it: i for i, it in enumerate(item_ids)}

    combined['user_id'] = combined['user_id'].map(user2idx)
    combined['item_id'] = combined['item_id'].map(item2idx)

    # ── Filter: keep only users with >= 5 interactions ──────────────
    # (Paper uses >= 20, but our Pinterest subset is smaller)
    user_counts = combined.groupby('user_id').size()
    valid_users = user_counts[user_counts >= 5].index
    combined    = combined[combined['user_id'].isin(valid_users)]

    # Re-index again after filtering
    user_ids = combined['user_id'].unique()
    item_ids = combined['item_id'].unique()
    u2i = {u: i for i, u in enumerate(sorted(user_ids))}
    i2i = {it: i for i, it in enumerate(sorted(item_ids))}
    combined['user_id'] = combined['user_id'].map(u2i)
    combined['item_id'] = combined['item_id'].map(i2i)

    # Add a fake timestamp (row order) for leave-one-out split
    combined = combined.reset_index(drop=True)
    combined['timestamp'] = combined.index

    print(f"  Final interactions : {len(combined):,}")
    print(f"  Final users        : {combined['user_id'].nunique():,}")
    print(f"  Final items        : {combined['item_id'].nunique():,}")

    return combined[['user_id', 'item_id', 'timestamp']]


# ─────────────────────────────────────────────
#  SECTION 3: Remap IDs & Compute Stats
# ─────────────────────────────────────────────

def remap_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-index user_id and item_id to start from 0 consecutively.
    Embedding layers in PyTorch require indices 0..N-1.
    """
    users = sorted(df['user_id'].unique())
    items = sorted(df['item_id'].unique())

    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}

    df = df.copy()
    df['user_id'] = df['user_id'].map(user2idx)
    df['item_id'] = df['item_id'].map(item2idx)

    return df, len(users), len(items)


# ─────────────────────────────────────────────
#  SECTION 4: Leave-One-Out Split
# ─────────────────────────────────────────────

def leave_one_out_split(df: pd.DataFrame):
    """
    For each user, the LAST interaction (by timestamp) goes to test set.
    Everything else stays in the training set.

    Returns:
      train_df  : all interactions except last per user
      test_df   : last interaction per user (ground truth)
    """
    # Sort by user then timestamp to get chronological order
    df_sorted = df.sort_values(['user_id', 'timestamp'])

    # The last item per user = test item
    test_df  = df_sorted.groupby('user_id').last().reset_index()
    test_df  = test_df[['user_id', 'item_id']]

    # Training = everything except the last interaction
    # We use a helper: mark each row's rank from the end
    df_sorted['rank_from_last'] = df_sorted.groupby('user_id').cumcount(ascending=False)
    train_df = df_sorted[df_sorted['rank_from_last'] > 0][['user_id', 'item_id']]
    train_df = train_df.reset_index(drop=True)

    print(f"  Train interactions : {len(train_df):,}")
    print(f"  Test  interactions : {len(test_df):,} (1 per user)")

    return train_df, test_df


# ─────────────────────────────────────────────
#  SECTION 5: Build User Interaction Sets
# ─────────────────────────────────────────────

def build_user_item_sets(train_df: pd.DataFrame, num_items: int):
    """
    Build a dictionary: user_id → set of item_ids the user interacted with.
    This is used to:
      1. Sample negatives during training (avoid sampling positives)
      2. Sample 100 test negatives per user for evaluation
    """
    user_items = defaultdict(set)
    for _, row in train_df.iterrows():
        user_items[int(row['user_id'])].add(int(row['item_id']))
    return dict(user_items)


# ─────────────────────────────────────────────
#  SECTION 6: Sample Test Negatives (100 per user)
# ─────────────────────────────────────────────

def sample_test_negatives(test_df: pd.DataFrame,
                           user_items: dict,
                           num_items: int,
                           num_negatives: int = 100,
                           seed: int = 42) -> dict:
    """
    For each test user, sample 100 items they have NOT interacted with.
    During evaluation, the model ranks the 1 positive test item among
    these 100 negatives.  HR@10 = 1 if the positive appears in top 10.

    Returns:
      test_negatives: dict { user_id → list of 100 negative item_ids }
    """
    rng = random.Random(seed)
    test_negatives = {}

    for _, row in test_df.iterrows():
        uid   = int(row['user_id'])
        pos   = int(row['item_id'])
        seen  = user_items.get(uid, set()) | {pos}  # exclude pos too

        negs = []
        while len(negs) < num_negatives:
            neg = rng.randint(0, num_items - 1)
            if neg not in seen:
                negs.append(neg)
        test_negatives[uid] = negs

    return test_negatives


# ─────────────────────────────────────────────
#  SECTION 7: Main Preprocessing Pipeline
# ─────────────────────────────────────────────

def preprocess_dataset(name: str, df: pd.DataFrame, output_dir: str):
    """
    Full preprocessing pipeline for one dataset.
    Saves: processed_<name>.pkl containing all required data.
    """
    print(f"\n{'='*55}")
    print(f" Processing: {name}")
    print(f"{'='*55}")

    # Step 1: Remap IDs to 0-indexed integers
    df, num_users, num_items = remap_ids(df)
    sparsity = 1 - len(df) / (num_users * num_items)
    print(f"  Users  : {num_users:,}")
    print(f"  Items  : {num_items:,}")
    print(f"  Interactions: {len(df):,}")
    print(f"  Sparsity: {sparsity*100:.2f}%")

    # Step 2: Leave-one-out split
    train_df, test_df = leave_one_out_split(df)

    # Step 3: Build user-item interaction set (from training only)
    user_items = build_user_item_sets(train_df, num_items)

    # Step 4: Sample 100 negatives per test user
    print("  Sampling 100 test negatives per user...")
    test_negatives = sample_test_negatives(test_df, user_items, num_items)

    # Step 5: Package everything and save
    data_dict = {
        'train_df'       : train_df,          # (user_id, item_id) training interactions
        'test_df'        : test_df,            # (user_id, item_id) one per user
        'user_items'     : user_items,         # user → set of seen items
        'test_negatives' : test_negatives,     # user → list of 100 neg items
        'num_users'      : num_users,
        'num_items'      : num_items,
        'sparsity'       : sparsity,
        'dataset_name'   : name,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"processed_{name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"  Saved → {out_path}")
    return data_dict


# ─────────────────────────────────────────────
#  SECTION 8: Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':

    # ── Adjust these paths to match your folder structure ──────────
    MOVIELENS_RATINGS = 'data/raw/ratings.dat'
    PINTEREST_POSTS   = 'data/raw/Pinterest-posts.csv'
    PINTEREST_PROFILES= 'data/raw/Pinterest-profiles.csv'
    OUTPUT_DIR        = 'data/processed'
    # ────────────────────────────────────────────────────────────────

    random.seed(42)
    np.random.seed(42)

    # ── MovieLens 1M ────────────────────────────────────────────────
    ml_df = load_movielens(MOVIELENS_RATINGS)
    ml_data = preprocess_dataset('movielens', ml_df, OUTPUT_DIR)

    # ── Pinterest ───────────────────────────────────────────────────
    pt_df = load_pinterest(PINTEREST_POSTS, PINTEREST_PROFILES)
    pt_data = preprocess_dataset('pinterest', pt_df, OUTPUT_DIR)

    print("\n✅ Preprocessing complete for both datasets.")
    print(f"   MovieLens → {ml_data['num_users']:,} users, {ml_data['num_items']:,} items")
    print(f"   Pinterest → {pt_data['num_users']:,} users, {pt_data['num_items']:,} items")
