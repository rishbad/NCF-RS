"""
=============================================================
 NCF Replication — Model Implementations
 Paper: "Neural Collaborative Filtering" He et al. (2017)
=============================================================

This file implements three models from the paper:

  1. GMF  — Generalized Matrix Factorization  (Section 3.2)
  2. MLP  — Multi-Layer Perceptron             (Section 3.3)
  3. NeuMF — Neural Matrix Factorization       (Section 3.4)
            = Fusion of GMF + MLP

Architecture quick-reference:
─────────────────────────────────────────────────────────────
GMF:
  user_embedding [K] ──┐
                        ├──► element-wise product [K] ──► sigmoid ──► score
  item_embedding [K] ──┘
  (output layer: linear(K → 1) with sigmoid)

MLP:
  user_embedding [K] ──┐
                        ├──► concat [2K] ──► FC(2K→L1, ReLU) ──► ... ──► sigmoid
  item_embedding [K] ──┘
  Tower structure: each layer halves the previous size

NeuMF:
  [GMF path]  user_emb_gmf ─── item_emb_gmf ──► element-wise product ──┐
  [MLP path]  user_emb_mlp ─── item_emb_mlp ──► concat + FC layers  ───┤
                                                                          ├──► concat ──► FC(1) ──► sigmoid
─────────────────────────────────────────────────────────────

Key design choices from the paper:
  - ReLU activation for MLP hidden layers (not sigmoid or tanh)
  - Sigmoid activation at output layer (constrains score to [0,1])
  - Separate embedding tables for GMF and MLP paths in NeuMF
    (allows them to learn different representations)
  - Tower structure for MLP: widest at bottom, narrowest at top
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
#  MODEL 1: GMF — Generalized Matrix Factorization
# ─────────────────────────────────────────────

class GMF(nn.Module):
    """
    Generalized Matrix Factorization.

    Extends classic MF by:
      - Using sigmoid (not identity) as output activation
      - Learning the output weight vector h (not a fixed sum)

    Forward pass:
      embed_user → p_u  (shape: batch × K)
      embed_item → q_i  (shape: batch × K)
      element-wise product → p_u ⊙ q_i  (shape: batch × K)
      fc_output(p_u ⊙ q_i) → scalar → sigmoid → score ∈ (0,1)
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int = 32):
        """
        Args:
          num_users : total number of users  (determines embedding table size)
          num_items : total number of items
          embed_dim : size of latent factor vector (K in the paper)
                      paper tests K ∈ {8, 16, 32, 64}
        """
        super(GMF, self).__init__()

        self.embed_dim = embed_dim

        # Embedding tables: one row per user/item, each row = latent vector of size K
        # These are initialized with Gaussian(0, 0.01) later via init_weights()
        self.embed_user = nn.Embedding(num_users, embed_dim)
        self.embed_item = nn.Embedding(num_items, embed_dim)

        # Output layer: maps the element-wise product (size K) to a single score
        # This is the "h" vector in the paper — learned, not fixed to all-ones
        self.fc_output = nn.Linear(embed_dim, 1, bias=False)

        # Sigmoid to squash score into [0,1] probability
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with Gaussian(0, 0.01) as in the paper."""
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.normal_(self.fc_output.weight,  std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        """
        Args:
          user_ids : LongTensor of shape (batch_size,)
          item_ids : LongTensor of shape (batch_size,)

        Returns:
          scores : FloatTensor of shape (batch_size,)  ← values in (0,1)
        """
        # Look up latent vectors
        p_u = self.embed_user(user_ids)   # (batch, K)
        q_i = self.embed_item(item_ids)   # (batch, K)

        # Element-wise product (Hadamard product)
        # This is the key GMF operation: captures linear interactions
        interaction = p_u * q_i           # (batch, K)

        # Linear layer + sigmoid → score
        score = self.sigmoid(self.fc_output(interaction))  # (batch, 1)
        return score.squeeze(-1)          # (batch,)

    def get_embeddings(self):
        """Return embedding weights for NeuMF pre-training initialization."""
        return self.embed_user.weight.data, self.embed_item.weight.data


# ─────────────────────────────────────────────
#  MODEL 2: MLP — Multi-Layer Perceptron
# ─────────────────────────────────────────────

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for collaborative filtering.

    Key idea: concatenate user & item embeddings, then pass through
    multiple FC layers with ReLU. The depth allows learning complex
    non-linear interactions that inner product cannot capture.

    Tower structure (from paper):
      Input:  concat(user_embed, item_embed) = 2*embed_dim
      Layer1: 2*embed_dim → embed_dim   (ReLU)
      Layer2: embed_dim   → embed_dim/2 (ReLU)
      ...
      Output: smallest_size → 1 (sigmoid)

    Paper default: 3 hidden layers (MLP-3)
    """

    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embed_dim: int = 32,
                 num_layers: int = 3,
                 dropout: float = 0.0):
        """
        Args:
          num_users  : total users
          num_items  : total items
          embed_dim  : embedding size. Note: the FIRST hidden layer has
                       size 2*embed_dim (since we concatenate both embeddings).
                       embed_dim here = size of the LAST (predictive) layer.
                       Paper: "if predictive factors = 8, architecture is 32→16→8"
          num_layers : number of hidden FC layers (paper tests 0–4)
          dropout    : dropout probability between layers (0 = no dropout)
        """
        super(MLP, self).__init__()

        self.embed_dim  = embed_dim
        self.num_layers = num_layers

        # Embedding tables — separate from GMF (as per paper)
        # Size of embedding = embed_dim * 2^(num_layers-1) for tower structure
        # So that after all layers, output has embed_dim neurons
        # e.g., if embed_dim=8, layers=3: embeddings=32, tower=64→32→16→8
        mlp_embed_size = embed_dim * (2 ** (num_layers - 1)) if num_layers > 0 else embed_dim

        self.embed_user = nn.Embedding(num_users, mlp_embed_size)
        self.embed_item = nn.Embedding(num_items, mlp_embed_size)

        # Build tower-structured MLP layers
        # Each layer halves the size of the previous
        layers = []
        input_size = mlp_embed_size * 2   # concat of user + item embeddings

        for i in range(num_layers):
            output_size = input_size // 2
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())            # Paper uses ReLU (Section 3.3)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_size = output_size

        self.mlp_layers = nn.Sequential(*layers)
        self.last_size  = input_size          # Size feeding into output layer

        # Output layer
        self.fc_output = nn.Linear(self.last_size, 1)
        self.sigmoid   = nn.Sigmoid()

        # If num_layers = 0: just concat → sigmoid (ablation study MLP-0)
        if num_layers == 0:
            self.fc_output = nn.Linear(mlp_embed_size * 2, 1)
            self.last_size = mlp_embed_size * 2

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fc_output.weight)
        nn.init.zeros_(self.fc_output.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        p_u = self.embed_user(user_ids)   # (batch, mlp_embed_size)
        q_i = self.embed_item(item_ids)   # (batch, mlp_embed_size)

        # Concatenate user and item embeddings
        # This is different from GMF — no multiplication, just stacking
        x = torch.cat([p_u, q_i], dim=-1)  # (batch, 2*mlp_embed_size)

        # Pass through tower layers
        x = self.mlp_layers(x)             # (batch, last_size)

        # Output
        score = self.sigmoid(self.fc_output(x))  # (batch, 1)
        return score.squeeze(-1)                  # (batch,)

    def get_embeddings(self):
        """Return embedding weights for NeuMF initialization."""
        return self.embed_user.weight.data, self.embed_item.weight.data

    def get_last_layer_output_size(self):
        return self.last_size


# ─────────────────────────────────────────────
#  MODEL 3: NeuMF — Neural Matrix Factorization
# ─────────────────────────────────────────────

class NeuMF(nn.Module):
    """
    Neural Matrix Factorization = GMF ⊕ MLP.

    Architecture (Figure 3 in paper):
      - GMF path: separate user/item embeddings → element-wise product
      - MLP path: separate user/item embeddings → concat → FC layers
      - Concatenate GMF output and MLP last-hidden-layer output
      - Final FC layer → sigmoid → prediction score

    Why separate embeddings?
      GMF needs to capture linear interactions (like MF).
      MLP needs to capture non-linear interactions.
      Different tasks may prefer different latent representations.
      Sharing embeddings would constrain both paths to the same space.

    Pre-training:
      Train GMF and MLP independently first, then use their weights
      to initialize NeuMF (warm start). This helps escape bad local
      minima since NeuMF's objective is non-convex.
    """

    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embed_dim: int = 32,
                 num_layers: int = 3,
                 dropout: float = 0.0):
        """
        Args:
          embed_dim  : predictive factors (size of GMF embedding AND MLP output)
          num_layers : number of MLP hidden layers
        """
        super(NeuMF, self).__init__()

        self.embed_dim  = embed_dim
        self.num_layers = num_layers

        # ── GMF path ──────────────────────────────────────────────
        self.gmf_embed_user = nn.Embedding(num_users, embed_dim)
        self.gmf_embed_item = nn.Embedding(num_items, embed_dim)

        # ── MLP path ──────────────────────────────────────────────
        mlp_embed_size = embed_dim * (2 ** (num_layers - 1)) if num_layers > 0 else embed_dim
        self.mlp_embed_user = nn.Embedding(num_users, mlp_embed_size)
        self.mlp_embed_item = nn.Embedding(num_items, mlp_embed_size)

        # Build MLP tower layers (same structure as standalone MLP)
        mlp_layers = []
        input_size = mlp_embed_size * 2
        for i in range(num_layers):
            output_size = input_size // 2
            mlp_layers.append(nn.Linear(input_size, output_size))
            mlp_layers.append(nn.ReLU())
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            input_size = output_size
        self.mlp_layers   = nn.Sequential(*mlp_layers)
        self.mlp_out_size = input_size  # Size of MLP's final hidden layer

        # ── NeuMF fusion layer ────────────────────────────────────
        # Input = concat(GMF output, MLP last hidden layer)
        # GMF contributes embed_dim neurons
        # MLP contributes mlp_out_size neurons
        fusion_size = embed_dim + (self.mlp_out_size if num_layers > 0 else mlp_embed_size * 2)
        self.fc_output = nn.Linear(fusion_size, 1, bias=False)
        self.sigmoid   = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Gaussian initialization as in paper (overridden by load_pretrained)."""
        nn.init.normal_(self.gmf_embed_user.weight, std=0.01)
        nn.init.normal_(self.gmf_embed_item.weight, std=0.01)
        nn.init.normal_(self.mlp_embed_user.weight, std=0.01)
        nn.init.normal_(self.mlp_embed_item.weight, std=0.01)
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def load_pretrained(self,
                        gmf_model: GMF,
                        mlp_model: MLP,
                        alpha: float = 0.5):
        """
        Initialize NeuMF using pre-trained GMF and MLP weights.
        This is Section 3.4.1 of the paper.

        α controls how much each model contributes to the output layer:
          h ← [α * h_GMF ; (1-α) * h_MLP]
        Paper sets α = 0.5 (equal contribution).

        Args:
          gmf_model : trained GMF model
          mlp_model : trained MLP model
          alpha     : trade-off between GMF and MLP (default 0.5)
        """
        # Copy GMF embeddings
        self.gmf_embed_user.weight.data.copy_(gmf_model.embed_user.weight.data)
        self.gmf_embed_item.weight.data.copy_(gmf_model.embed_item.weight.data)

        # Copy MLP embeddings
        self.mlp_embed_user.weight.data.copy_(mlp_model.embed_user.weight.data)
        self.mlp_embed_item.weight.data.copy_(mlp_model.embed_item.weight.data)

        # Copy MLP hidden layer weights
        mlp_layer_idx = 0
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                # Find corresponding layer in source MLP
                src_layers = [l for l in mlp_model.mlp_layers if isinstance(l, nn.Linear)]
                if mlp_layer_idx < len(src_layers):
                    m.weight.data.copy_(src_layers[mlp_layer_idx].weight.data)
                    m.bias.data.copy_(src_layers[mlp_layer_idx].bias.data)
                mlp_layer_idx += 1

        # Concatenate output weights from GMF and MLP with alpha weighting
        # h = [alpha * h_GMF ; (1-alpha) * h_MLP]
        gmf_out = gmf_model.fc_output.weight.data   # shape: (1, embed_dim)
        mlp_out_weights = mlp_model.fc_output.weight.data  # shape: (1, last_size)

        fused_weights = torch.cat([
            alpha * gmf_out,
            (1 - alpha) * mlp_out_weights
        ], dim=1)  # shape: (1, embed_dim + last_size)

        self.fc_output.weight.data.copy_(fused_weights)
        print("  ✅ Pre-trained weights loaded into NeuMF.")

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        """
        Args:
          user_ids : LongTensor (batch_size,)
          item_ids : LongTensor (batch_size,)

        Returns:
          scores : FloatTensor (batch_size,) in (0,1)
        """
        # ── GMF path ──────────────────────────────────────────────
        gmf_user = self.gmf_embed_user(user_ids)   # (batch, embed_dim)
        gmf_item = self.gmf_embed_item(item_ids)   # (batch, embed_dim)
        gmf_out  = gmf_user * gmf_item             # element-wise product (batch, embed_dim)

        # ── MLP path ──────────────────────────────────────────────
        mlp_user = self.mlp_embed_user(user_ids)
        mlp_item = self.mlp_embed_item(item_ids)
        mlp_x    = torch.cat([mlp_user, mlp_item], dim=-1)  # concat

        if self.num_layers > 0:
            mlp_out = self.mlp_layers(mlp_x)       # (batch, mlp_out_size)
        else:
            mlp_out = mlp_x                        # MLP-0 case

        # ── Fusion ────────────────────────────────────────────────
        # Concatenate GMF output and MLP last hidden representation
        concat = torch.cat([gmf_out, mlp_out], dim=-1)  # (batch, embed_dim + mlp_out_size)

        # Final prediction
        score = self.sigmoid(self.fc_output(concat))     # (batch, 1)
        return score.squeeze(-1)                         # (batch,)


# ─────────────────────────────────────────────
#  Factory Function
# ─────────────────────────────────────────────

def build_model(model_type: str,
                num_users: int,
                num_items: int,
                embed_dim: int = 32,
                num_layers: int = 3,
                dropout: float = 0.0) -> nn.Module:
    """
    Convenience factory to build any NCF model by name.

    Args:
      model_type : 'GMF', 'MLP', or 'NeuMF'
      num_users  : number of users
      num_items  : number of items
      embed_dim  : embedding / predictive factor size
      num_layers : number of MLP hidden layers (for MLP and NeuMF)
      dropout    : dropout rate

    Returns:
      model : nn.Module
    """
    model_type = model_type.upper()
    if model_type == 'GMF':
        return GMF(num_users, num_items, embed_dim)
    elif model_type == 'MLP':
        return MLP(num_users, num_items, embed_dim, num_layers, dropout)
    elif model_type == 'NEUMF':
        return NeuMF(num_users, num_items, embed_dim, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose GMF, MLP, or NeuMF.")
