#!/usr/bin/env python3
"""
Code for Admissible Discrimination: A Framework for Fair Machine Learning
----------------------------------------------------
Components:
- Single training run with checkpoints (15, 30, 50, 100 epochs)
- Adversarial training (for debiasing)
- Correct MI reporting using CLUB for fair comparison with internal training of CLUB
- Clean progress tracking with tqdm
- Added reporting of training loss statistics per epoch.
- Comprehensive training and test evaluation metrics for both P and P_tilde distributions, with correct naming.
- Conditional Demographic Parity (CDP) which uses KNN-based approach for robustness.
- Added confusion matrix components (TP, FP, etc.) as evaluation metrics.
- Linear model architecture is conditionally applied only to the German credit dataset.
- Added 'direct_estimation' stage for comparison, which trains on Z, V, and S, then averages over S post-hoc.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime
import csv
import random
from typing import List, Dict, Tuple
from numbers import Number

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_curve, auc,
    brier_score_loss, log_loss, confusion_matrix
)
from sklearn.neighbors import NearestNeighbors

from fairlearn.metrics import (
    demographic_parity_difference,
    true_positive_rate_difference,
    false_positive_rate_difference,
    equalized_odds_difference,
)
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Checkpoint Epochs
CHECKPOINT_EPOCHS = [15, 30, 50, 100]

# --- KNN-specific parameters for CDP calculation ---
K_NEIGHBORS = 50 # Number of neighbors to consider for KNN-based CDP
MIN_GROUP_SAMPLES_KNN = 5 # Minimum samples for each sensitive group in a neighborhood for valid comparison

# Data configuration
DATA_PARTITION = {
    "adult": {
        "protected_attrs": {
            "sex": {
                "S": ["sex"],
                "Z": ["age", "education-num"],
                "X": ["workclass=", "occupation=", "hours-per-week",
                      "capital-gain", "capital-loss", "marital-status=", "relationship="],
            },
            "race": {
                "S": ["race"],
                "Z": ["age", "education-num"],
                "X": ["workclass=", "occupation=", "capital-gain", "hours-per-week", "capital-loss",
                      "marital-status=", "relationship="],
            }
        },
        "encoder_dim": 16
    },
    "german": {
        "protected_attrs": {
            "sex": {
                "S": ["sex"],
                "Z": ["month", "credit_amount"],
                "X": ["credit_history=", "purpose=", "other_debtors=",
                      "number_of_credits", "savings=", "employment=", "housing="],
            },
            "age": {
                "S": ["age"],
                "Z": ["month", "credit_amount"],
                "X": ["credit_history=", "purpose=", "other_debtors=",
                      "number_of_credits", "savings=", "employment=", "housing="],
            }
        },
        "encoder_dim": 8
    },
    "compas": {
        "protected_attrs": {
            "race": {
                "S": ["race"],
                "Z": ["juv_fel_count", "juv_misd_count", "juv_other_count",
                      "priors_count", "c_charge_degree="],
                "X": ["age_cat=", "c_charge_desc="],
            },
            "sex": {
                "S": ["sex"],
                "Z": ["juv_fel_count", "juv_misd_count", "juv_other_count",
                      "priors_count", "c_charge_degree="],
                "X": ["age_cat=", "c_charge_desc="],
            }
        },
        "encoder_dim": 16
    },
}

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _canon(name: str) -> str:
    """Canonicalize column names"""
    return name.lower().replace("-", "_").replace("=", "_").replace(" ", "_")

def _expand_cols(bases: List[str], all_cols: List[str]) -> List[str]:
    """Expand column patterns to actual column names"""
    canon_map = {_canon(c): c for c in all_cols}
    out = []
    for base in bases:
        key = _canon(base)
        if key in canon_map:
            out.append(canon_map[key])
            continue
        matches = [c for c in all_cols if _canon(c).startswith(key)]
        if not matches:
            raise KeyError(f"{base!r} not found in dataset columns")
        out.extend(matches)
    return list(dict.fromkeys(out))

def _load_raw(name: str):
    """Load raw dataset"""
    return {"adult": AdultDataset, "german": GermanDataset, "compas": CompasDataset}[name]()

def build_partitions(ds, name: str, protected_attr: str):
    """Build feature partitions (X, Z, S, y)"""
    meta = DATA_PARTITION[name]["protected_attrs"][protected_attr]
    df = pd.DataFrame(ds.features, columns=ds.feature_names)
    df["label"] = ds.labels.ravel()

    if name == "german":
        df["label"] = df["label"] - 1  # Convert {1,2} to {0,1}

    for i, attr in enumerate(ds.protected_attribute_names):
        df[attr] = ds.protected_attributes[:, i]

    all_cols = list(df.columns)
    S_col = meta["S"][0]
    Z_cols = _expand_cols(meta["Z"], all_cols)
    X_cols = _expand_cols(meta["X"], all_cols)

    S = df[S_col].values
    Zdf = df[Z_cols] if Z_cols else pd.DataFrame(index=df.index)
    Xdf = df[X_cols] if X_cols else pd.DataFrame(index=df.index)
    y = df["label"].values

    return Xdf, Zdf, S, y

def make_ct(df: pd.DataFrame) -> ColumnTransformer:
    """Create column transformer for preprocessing"""
    num = [c for c in df.columns if df[c].dtype != "object"]
    cat = [c for c in df.columns if df[c].dtype == "object"]
    steps = []
    if num:
        steps.append(("num", StandardScaler(), num))
    if cat:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return ColumnTransformer(steps)

# Neural Network Components
class Encoder(nn.Module):
    """Encoder for feature transformation (non-linear)"""
    def __init__(self, in_dim, v_dim=16, hidden=64):
        super().__init__()
        if in_dim == 0:
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, v_dim)
            )

    def forward(self, x):
        if self.net is None:
            return torch.zeros(x.size(0), 0).to(x.device)
        return self.net(x)

class Predictor(nn.Module):
    """Main prediction network (non-linear)"""
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)

# Linear model alternative for 'linear' architecture
class LinearEncoder(nn.Module):
    """Linear Encoder with no activation functions"""
    def __init__(self, in_dim, v_dim=16):
        super().__init__()
        if in_dim == 0:
            self.net = None
        else:
            self.net = nn.Linear(in_dim, v_dim)
    
    def forward(self, x):
        if self.net is None:
            return torch.zeros(x.size(0), 0).to(x.device)
        return self.net(x)

class LinearPredictor(nn.Module):
    """Linear Predictor with no activation functions"""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Linear(in_dim, 1)
    
    def forward(self, features):
        return self.net(features).squeeze(-1)


class Adversary(nn.Module):
    """Adversarial network for bias detection"""
    def __init__(self, in_dim, num_classes=2, hidden=32):
        super().__init__()
        if in_dim == 0:
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_classes)
            )

    def forward(self, features):
        if self.net is None:
            return torch.zeros(features.size(0), 2).to(features.device)
        return self.net(features)

def compute_weights(z: np.ndarray, v: np.ndarray, s: np.ndarray, clip_quant: float = 0.95):
    """Compute importance weights for reweighting"""
    if v.size == 0 and z.size == 0:
        return np.ones_like(s, "float32"), len(s)
    elif v.size == 0:
        zv = z
    elif z.size == 0:
        zv = v
    else:
        zv = np.concatenate([z, v], axis=1)

    if zv.shape[1] == 0:
        return np.ones_like(s, "float32"), len(s)

    s = s.ravel()

    if len(np.unique(s)) < 2:
        return np.ones_like(s, "float32"), len(s)

    clf = GradientBoostingClassifier(max_depth=3, n_estimators=100, random_state=42)
    clf.fit(zv, s)

    p_s1 = s.mean()
    p_s1_zv = clf.predict_proba(zv)[:, 1].clip(1e-6, 1-1e-6)

    w = np.where(s == 1, p_s1 / p_s1_zv, (1 - p_s1) / (1 - p_s1_zv))

    tau = np.quantile(w, clip_quant)
    w = np.minimum(w, tau)
    w *= len(w) / w.sum()
    ess = w.sum() ** 2 / (w ** 2).sum()
    return w.astype("float32"), ess

def _compute_standard_metrics(y, p, s, sample_weight=None):
    """Compute standard utility and fairness metrics for a given set of data and weights."""
    y_hat = (p > 0.5).astype(int)

    prec, rec, _ = precision_recall_curve(y, p, sample_weight=sample_weight)
    pr_auc = auc(rec, prec)
    brier = brier_score_loss(y, p, sample_weight=sample_weight)
    logloss = log_loss(y, p, sample_weight=sample_weight)

    s_fairlearn = s.reshape(-1) if s.ndim > 1 else s

    tpr_gap = true_positive_rate_difference(y, y_hat, sensitive_features=s_fairlearn, sample_weight=sample_weight)
    fpr_gap = false_positive_rate_difference(y, y_hat, sensitive_features=s_fairlearn, sample_weight=sample_weight)
    avg_odds = equalized_odds_difference(y, y_hat, sensitive_features=s_fairlearn, sample_weight=sample_weight)
    dp = demographic_parity_difference(y, y_hat, sensitive_features=s_fairlearn, sample_weight=sample_weight)

    return {
        "acc": accuracy_score(y, y_hat, sample_weight=sample_weight),
        "auc": roc_auc_score(y, p, sample_weight=sample_weight),
        "pr_auc": pr_auc, "brier": brier, "logloss": logloss,
        "tpr_gap": tpr_gap, "fpr_gap": fpr_gap, "avg_odds": avg_odds,
        "dp": dp
    }

def _compute_confusion_metrics(y, p, s, sample_weight=None):
    """Compute confusion matrix components (TP, FP, etc.) for each S group."""
    y_hat = (p > 0.5).astype(int)
    
    groups = np.unique(s)
    metrics = {}
    for group in groups:
        group_idx = (s == group)
        y_g = y[group_idx]
        y_hat_g = y_hat[group_idx]
        w_g = sample_weight[group_idx] if sample_weight is not None else None
        
        # Ensure labels are explicitly [0, 1] to avoid issues with missing classes in a group
        tn, fp, fn, tp = confusion_matrix(y_g, y_hat_g, sample_weight=w_g, labels=[0, 1]).ravel()
        
        metrics.update({
            f'tn_s{group}': tn,
            f'fp_s{group}': fp,
            f'fn_s{group}': fn,
            f'tp_s{group}': tp,
        })
    return metrics

def compute_prediction_bias(predictions, sensitive_attr):
    """Compute I(Ŷ;S) - bias in predictions (using correlation for simplicity)"""
    predictions = predictions.ravel()
    sensitive_attr = sensitive_attr.ravel()

    if len(np.unique(predictions)) <= 1 or len(np.unique(sensitive_attr)) <= 1:
        return 0.0
    correlation = np.abs(np.corrcoef(predictions, sensitive_attr)[0, 1])
    return correlation

def _compute_conditional_demographic_parity(y_pred_probs, s_binary, z, k_neighbors=50, min_group_samples=5):
    """
    Computes Conditional Demographic Parity (CDP) using KNN to define local neighborhoods.

    y_pred_probs: Predicted probabilities or outcomes (numpy array).
    s_binary: Binarized sensitive attribute (numpy array, e.g., 0 and 1).
    z: Admissible features (numpy array). Can be 1D or 2D.
    k_neighbors: Number of neighbors to consider for each local neighborhood.
    min_group_samples: Minimum number of samples required for *each* sensitive group
                       within a local neighborhood to be included in CDP calculation.
    """
    y_pred = y_pred_probs # Use raw predicted outcomes for mean calculations

    # Ensure z is 2D for KNN (even if it's a single feature). If z is empty, handle gracefully.
    if z.shape[1] == 0:
        # If there are no conditional features, CDP is equivalent to demographic parity
        # Use fairlearn's demographic_parity_difference for this case
        dp = demographic_parity_difference(
            y_true=(y_pred_probs > 0.5).astype(int), # Need labels for fairlearn metrics
            y_pred=(y_pred_probs > 0.5).astype(int),
            sensitive_features=s_binary
        )
        return dp

    if z.ndim == 1:
        z = z.reshape(-1, 1)

    # Scale Z features for meaningful distance calculations
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    # Fit KNN model on scaled Z features
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(z_scaled)

    max_cdp = 0.0
    s_unique = np.unique(s_binary)
    if len(s_unique) < 2:
        return 0.0 # Cannot compute CDP if there's only one sensitive group after binarization

    # Iterate through each data point to define its local neighborhood
    for i in range(len(y_pred)):
        # Find k-nearest neighbors (indices) for the current point z_scaled[i]
        # kneighbors returns (distances, indices)
        distances, indices = nn_model.kneighbors(z_scaled[i].reshape(1, -1))
        
        # Select data points within this neighborhood
        neighborhood_y_pred = y_pred[indices[0]]
        neighborhood_s_binary = s_binary[indices[0]]

        all_s_present_and_sufficient = True
        rates_for_s_groups = {}

        for s_val in s_unique:
            s_sub_group = neighborhood_y_pred[neighborhood_s_binary == s_val]
            if len(s_sub_group) < min_group_samples: # Check minimum samples
                all_s_present_and_sufficient = False
                break
            rates_for_s_groups[s_val] = np.mean(s_sub_group) # Calculate mean prediction for the group in this neighborhood
        
        if all_s_present_and_sufficient:
            # Calculate all pairwise differences within this specific neighborhood
            for s1_val in s_unique:
                for s2_val in s_unique:
                    if s1_val < s2_val: # Only compare each pair once
                        rate1 = rates_for_s_groups[s1_val]
                        rate2 = rates_for_s_groups[s2_val]
                        max_cdp = max(max_cdp, abs(rate1 - rate2))
    
    return max_cdp

# --- CLUB Estimator --- Taken from the paper 
"""@inproceedings{cheng2020club,
  title={Club: A contrastive log-ratio upper bound of mutual information},
  author={Cheng, Pengyu and Hao, Weituo and Dai, Shuyang and Liu, Jiachang and Gan, Zhe and Carin, Lawrence},
  booktitle={International conference on machine learning},
  pages={1779--1788},
  year={2020},
  organization={PMLR}
}"""
class CLUBForCategorical(nn.Module):
    '''
    This class provide a CLUB estimator to calculate MI upper bound between vector-like embeddings and categorical labels.
    Estimate I(X,Y), where X is continuous vector and Y is discrete label.
    '''
    def __init__(self, input_dim, label_num, hidden_size=None):
        super().__init__()

        if input_dim == 0:
            self.variational_net = None
        else:
            if hidden_size is None:
                self.variational_net = nn.Linear(input_dim, label_num)
            else:
                self.variational_net = nn.Sequential(
                    nn.Linear(input_dim, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, label_num)
                )

    def forward(self, inputs, labels):
        '''
        inputs : shape [batch_size, input_dim], a batch of embeddings
        labels : shape [batch_size], a batch of label index
        '''
        if self.variational_net is None or inputs.shape[1] == 0:
            return torch.tensor(0.0).to(inputs.device)

        logits = self.variational_net(inputs)  # [sample_size, label_num]
        sample_size, label_num = logits.shape

        logits_extend = logits.unsqueeze(1).repeat(1, sample_size, 1)  # shape [sample_size, sample_size, label_num]
        labels_extend = labels.unsqueeze(0).repeat(sample_size, 1)     # shape [sample_size, sample_size]

        # log of conditional probability of negative sample pairs
        log_mat = - nn.functional.cross_entropy(
            logits_extend.reshape(-1, label_num),
            labels_extend.reshape(-1, ),
            reduction='none'
        )

        log_mat = log_mat.reshape(sample_size, sample_size)
        positive = torch.diag(log_mat).mean()
        negative = log_mat.mean()
        return positive - negative

    def loglikeli(self, inputs, labels):
        if self.variational_net is None or inputs.shape[1] == 0: return torch.tensor(0.0).to(inputs.device)
        logits = self.variational_net(inputs)
        return - nn.functional.cross_entropy(logits, labels)

    def learning_loss(self, inputs, labels):
        return - self.loglikeli(inputs, labels)


# taken from https://fairlearn.org/main/user_guide/mitigation/adversarial.html
# https://aif360.readthedocs.io/en/stable/modules/generated/aif360.sklearn.inprocessing.AdversarialDebiasing.html

class AdversarialTrainer:
    """Adversarial training framework"""

    def __init__(self, stage, encoder, predictor, adversary, lr=1e-3, adv_weight=0.1):
        self.stage = stage
        self.encoder = encoder
        self.predictor = predictor
        self.adversary = adversary # This adversary is for debiasing only
        self.adv_weight = adv_weight

        # Optimizers - handle None encoder/adversary case
        self.opt_encoder = optim.Adam(encoder.parameters(), lr) if encoder is not None and encoder.net is not None else None
        self.opt_predictor = optim.Adam(predictor.parameters(), lr)
        self.opt_adversary = optim.Adam(adversary.parameters(), lr) if adversary is not None and adversary.net is not None else None

        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def step(self, xb, zb, sb, yb, wb):
        """Single training step"""
        xb, zb, sb, yb, wb = (t.to(DEVICE) for t in (xb, zb, sb, yb, wb))

        if self.stage == "naive":
            return self._step_naive(xb, zb, yb)
        elif self.stage == "full_debias":
            return self._step_full_debias(xb, zb, sb, yb)
        elif self.stage == "ours":
            return self._step_ours(xb, zb, sb, yb, wb)
        elif self.stage == "direct_estimation": # NEW STAGE
            return self._step_direct_estimation(xb, zb, sb, yb)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

    def _step_naive(self, xb, zb, yb):
        """Naive training - no debiasing."""
        features = torch.cat([zb, xb], 1) if xb.shape[1] > 0 else zb
        logits = self.predictor(features)
        loss = self.bce(logits, yb.float()).mean()

        self.opt_predictor.zero_grad()
        loss.backward()
        self.opt_predictor.step()

        return {"pred_loss": loss.item(), "adv_loss": 0.0} # No adv loss for naive stage

    def _step_full_debias(self, xb, zb, sb, yb):
        """Full debiasing - adversarial training on all features (combined into V)."""
        # Combine all features for encoder input
        combined_features_for_encoder = torch.cat([zb, xb], 1)

        # Step 1: Train adversary (if exists) to predict S from V
        adv_loss = torch.tensor(0.0) # Initialize in case adversary is None
        if self.opt_adversary is not None:
            V = self.encoder(combined_features_for_encoder).detach() # Detach V for adversary training
            adv_logits = self.adversary(V)
            adv_loss = F.cross_entropy(adv_logits, sb.long())

            self.opt_adversary.zero_grad()
            adv_loss.backward()
            self.opt_adversary.step()

        # Step 2: Train encoder + predictor with gradient modification (to make V independent of S)
        V = self.encoder(combined_features_for_encoder) # V for predictor is not detached
        pred_logits = self.predictor(V) # Predictor takes V as input
        pred_loss = self.bce(pred_logits, yb.float()).mean()

        total_loss = pred_loss
        if self.adversary is not None and self.adversary.net is not None:
            # Adversarial component for gradient modification
            adv_logits_for_grad = self.adversary(V)
            adv_loss_for_grad = F.cross_entropy(adv_logits_for_grad, sb.long())

            # Total loss for encoder/predictor: minimize pred_loss, maximize adv_loss (hence -adv_weight)
            total_loss = pred_loss - self.adv_weight * adv_loss_for_grad

        if self.opt_encoder:
            self.opt_encoder.zero_grad()
        self.opt_predictor.zero_grad()

        total_loss.backward()

        if self.opt_encoder:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.opt_encoder.step()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.opt_predictor.step()

        return {"pred_loss": pred_loss.item(), "adv_loss": adv_loss.item()}

    def _step_ours(self, xb, zb, sb, yb, wb):
        """Our method - debias X only (into V), use with Z."""
        # If no X features or encoder, behave like naive with Z
        if xb.shape[1] == 0 or self.encoder is None or self.encoder.net is None:
            features = zb
            logits = self.predictor(features)
            loss = (wb * self.bce(logits, yb.float())).mean()

            self.opt_predictor.zero_grad()
            loss.backward()
            self.opt_predictor.step()

            return {"pred_loss": loss.item(), "adv_loss": 0.0}

        # Step 1: Train adversary on V = encoder(X) (for debiasing V from S)
        adv_loss = torch.tensor(0.0) # Initialize in case adversary is None
        if self.opt_adversary is not None:
            V = self.encoder(xb).detach() # Detach V for adversary training
            adv_logits = self.adversary(V) # Adversary for 'ours' expects V only
            adv_loss = F.cross_entropy(adv_logits, sb.long())

            self.opt_adversary.zero_grad()
            adv_loss.backward()
            self.opt_adversary.step()

        # Step 2: Train encoder + predictor (predictor uses Z and V)
        V = self.encoder(xb) # V for predictor is not detached
        features = torch.cat([zb, V], 1)
        pred_logits = self.predictor(features)
        pred_loss = (wb * self.bce(pred_logits, yb.float())).mean()

        total_loss = pred_loss
        if self.adversary is not None and self.adversary.net is not None:
            # Adversarial component for gradient modification (still operating on V)
            adv_logits_for_grad = self.adversary(V)
            adv_loss_for_grad = F.cross_entropy(adv_logits_for_grad, sb.long())

            total_loss = pred_loss - self.adv_weight * adv_loss_for_grad

        self.opt_encoder.zero_grad()
        self.opt_predictor.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.opt_encoder.step()
        self.opt_predictor.step()

        return {"pred_loss": pred_loss.item(), "adv_loss": adv_loss.item()}

    def _step_direct_estimation(self, xb, zb, sb, yb): # NEW METHOD FOR DIRECT ESTIMATION STAGE
        """Direct estimation training - predictor takes Z, V, and S."""
        # Encode X into V
        # If there's no X or encoder, V should be an empty tensor of correct batch size
        if self.encoder is None or xb.shape[1] == 0 or self.encoder.net is None:
            V = torch.zeros(xb.size(0), 0).to(xb.device)
        else:
            V = self.encoder(xb)

        # Predictor inputs: Z, V, S
        # S needs to be unsqueezed and converted to float for concatenation
        # Assume S is already numerical (0/1) for direct input
        features = torch.cat([zb, V, sb.unsqueeze(1).float()], 1)
        logits = self.predictor(features)
        loss = self.bce(logits, yb.float()).mean()

        # Optimize predictor and encoder (if it exists)
        self.opt_predictor.zero_grad()
        if self.opt_encoder: # Only zero grad if encoder exists
            self.opt_encoder.zero_grad()
        
        loss.backward()

        self.opt_predictor.step()
        if self.opt_encoder: # Only step if encoder exists
            self.opt_encoder.step()

        return {"pred_loss": loss.item(), "adv_loss": 0.0} # No adv loss for direct estimation stage


    def _predict_and_collect_data(self, data_loader, p_s0_marginal=None, p_s1_marginal=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        """
        Runs inference on the model and collects predictions, true labels, sensitive attributes,
        weights, admissible features (Z), and features for MI calculation.
        For 'direct_estimation' stage, it performs post-hoc averaging over S.
        """
        self.predictor.eval()
        if self.encoder:
            self.encoder.eval()

        all_probs, all_ys, all_ss, all_ws, all_zs = [], [], [], [], []
        all_features_for_mi_club = []

        with torch.no_grad():
            for xb, zb, sb, yb, wb in data_loader:
                xb, zb, sb_cpu, yb_cpu, wb_cpu = xb.to(DEVICE), zb.to(DEVICE), sb.cpu(), yb.cpu(), wb.cpu() # Keep original s, y, w for final concat on cpu

                features_for_prediction = None
                features_for_mi = None
                probs = None # Initialize probs for current batch

                # Feature handling logic remains the same regardless of specific Encoder/Predictor class (linear/non-linear)
                # This logic is about how X, Z are combined and whether X is encoded to V.
                if self.stage == "naive":
                    features_for_prediction = torch.cat([zb, xb], 1) if xb.shape[1] > 0 else zb
                    features_for_mi = features_for_prediction.cpu()
                    logits = self.predictor(features_for_prediction)
                    probs = torch.sigmoid(logits)
                elif self.stage == "full_debias":
                    combined = torch.cat([zb, xb], 1)
                    V = self.encoder(combined) if self.encoder and self.encoder.net else torch.zeros(combined.size(0), 0).to(combined.device)
                    features_for_prediction = V
                    features_for_mi = V.cpu()
                    logits = self.predictor(features_for_prediction)
                    probs = torch.sigmoid(logits)
                elif self.stage == "ours":
                    if xb.shape[1] > 0 and self.encoder and self.encoder.net:
                        V = self.encoder(xb)
                        features_for_prediction = torch.cat([zb, V], 1)
                        features_for_mi = features_for_prediction.cpu()
                    else:
                        features_for_prediction = zb
                        features_for_mi = zb.cpu() # if X is empty, MI is on Z
                    logits = self.predictor(features_for_prediction)
                    probs = torch.sigmoid(logits)
                elif self.stage == "direct_estimation": # NEW STAGE
                    # For direct_estimation, we predict for S=0 and S=1 and average
                    # Encode X into V
                    V = self.encoder(xb) if self.encoder and self.encoder.net else torch.zeros(xb.size(0), 0).to(xb.device)
                    
                    # Create dummy S tensors for counterfactual prediction (same batch size)
                    dummy_s0 = torch.zeros_like(sb.unsqueeze(1).float()).to(DEVICE)
                    dummy_s1 = torch.ones_like(sb.unsqueeze(1).float()).to(DEVICE)
                    
                    # Predict for S=0
                    features_s0 = torch.cat([zb, V, dummy_s0], 1)
                    logits_s0 = self.predictor(features_s0)
                    probs_s0 = torch.sigmoid(logits_s0)

                    # Predict for S=1
                    features_s1 = torch.cat([zb, V, dummy_s1], 1)
                    logits_s1 = self.predictor(features_s1)
                    probs_s1 = torch.sigmoid(logits_s1)

                    # Calculate the weighted average using marginal probabilities of S
                    # These marginals are passed in from run_single_experiment
                    avg_probs_s = p_s0_marginal * probs_s0 + p_s1_marginal * probs_s1
                    
                    probs = avg_probs_s # Use these averaged probabilities for metrics
                    # Features for MI calculation would be Z + V (as S is averaged out from prediction)
                    features_for_mi = torch.cat([zb, V], 1).cpu()
                else:
                    raise ValueError(f"Unknown stage for prediction: {self.stage}")


                all_probs.append(probs.cpu())
                all_ys.append(yb_cpu)
                all_ss.append(sb_cpu)
                all_ws.append(wb_cpu)
                all_zs.append(zb.cpu()) # Collect Z features
                all_features_for_mi_club.append(features_for_mi)

        # Concatenate collected data
        p = torch.cat(all_probs).numpy()
        y = torch.cat(all_ys).numpy()
        s = torch.cat(all_ss).numpy()
        w = torch.cat(all_ws).numpy()
        z = torch.cat(all_zs).numpy() # Concatenate Z features
        features_for_mi_tensor = torch.cat(all_features_for_mi_club)
        
        # Reset to training mode
        self.predictor.train()
        if self.encoder:
            self.encoder.train()
        if self.adversary and self.adversary.net is not None:
            self.adversary.train()

        return p, y, s, w, z, features_for_mi_tensor
    
    def _calculate_all_evaluation_metrics(self, p: np.ndarray, y: np.ndarray, s: np.ndarray, w: np.ndarray, z: np.ndarray, features_for_mi_tensor: torch.Tensor, num_s_classes: int, prefix_scope: str = "") -> Dict[str, float]:
        """
        Calculates all utility, fairness, and MI metrics for P and P_tilde distributions.
        Args:
            p (np.ndarray): Predicted probabilities.
            y (np.ndarray): True labels.
            s (np.ndarray): Sensitive attributes.
            w (np.ndarray): Importance weights for P_tilde.
            z (np.ndarray): Admissible features for CDP calculation.
            features_for_mi_tensor (torch.Tensor): Features used for CLUB MI estimation.
            num_s_classes (int): Number of unique sensitive classes.
            prefix_scope (str): Scope prefix, either 'tr_' or 'te_'.
        Returns:
            dict: Dictionary of all computed metrics.
        """
        metrics = {}

        # Ensure s is treated as categorical for fairlearn metrics
        s_fairlearn = s.reshape(-1) if s.ndim > 1 else s
        
        # 1. P-distribution metrics (using uniform weights or no weights)
        p_metrics_orig = _compute_standard_metrics(y, p, s_fairlearn, sample_weight=None)
        confusion_metrics_orig = _compute_confusion_metrics(y, p, s_fairlearn, sample_weight=None)
        
        for k, v in {**p_metrics_orig, **confusion_metrics_orig}.items():
            metrics[f'{k}_{prefix_scope}P'] = v # This correctly forms 'acc_tr_P', 'auc_te_P', etc.
        
        # 2. P-tilde distribution metrics (using importance weights)
        # P_tilde metrics only apply to 'ours' stage, but we populate for consistency
        # If not 'ours', P_tilde metrics will be identical to P metrics.
        if self.stage == "ours":
            p_metrics_tilde = _compute_standard_metrics(y, p, s_fairlearn, sample_weight=w)
            confusion_metrics_tilde = _compute_confusion_metrics(y, p, s_fairlearn, sample_weight=w)
            for k, v in {**p_metrics_tilde, **confusion_metrics_tilde}.items():
                metrics[f'{k}_{prefix_scope}Ptilde'] = v # Correctly forms 'acc_tr_Ptilde', etc.
        else:
            for k, v in {**p_metrics_orig, **confusion_metrics_orig}.items():
                metrics[f'{k}_{prefix_scope}Ptilde'] = v # Will be same as _P, but with _Ptilde suffix


        # 3. Mutual Information (CLUB)
        club_input_dim = features_for_mi_tensor.shape[1]
        mi_estimate_club = 0.0
        if club_input_dim > 0 and num_s_classes > 1:
            club_estimator = CLUBForCategorical(club_input_dim, num_s_classes, hidden_size=64).to(DEVICE)
            club_optimizer = optim.Adam(club_estimator.parameters(), lr=1e-3)

            club_epochs = 20
            club_batch_size = min(256, features_for_mi_tensor.shape[0])
            club_dataloader = DataLoader(TensorDataset(features_for_mi_tensor, torch.tensor(s).long()),
                                         batch_size=club_batch_size, shuffle=True)

            club_estimator.train()
            for _ in range(club_epochs):
                for mi_features_batch, mi_s_batch in club_dataloader:
                    mi_features_batch = mi_features_batch.to(DEVICE)
                    mi_s_batch = mi_s_batch.to(DEVICE)
                    club_optimizer.zero_grad()
                    loss = club_estimator.learning_loss(mi_features_batch, mi_s_batch)
                    loss.backward()
                    club_optimizer.step()
            
            club_estimator.eval()
            with torch.no_grad():
                mi_estimate_club = club_estimator(features_for_mi_tensor.to(DEVICE),
                                                  torch.tensor(s).long().to(DEVICE)).item()
            mi_estimate_club = max(0.0, mi_estimate_club)
        
        metrics[f'mi_{prefix_scope[:-1]}'] = mi_estimate_club # `prefix_scope` is "tr_", so `[:-1]` makes it "tr"

        # 4. I(Y_hat; S) using correlation.
        metrics[f'I_pred_S_{prefix_scope[:-1]}'] = compute_prediction_bias(p, s)
        
        # 5. Conditional Demographic Parity (CDP) using KNN
        # Pass s_fairlearn (reshaped s) as s_binary to the KNN function
        cdp = _compute_conditional_demographic_parity(p, s_fairlearn, z, k_neighbors=K_NEIGHBORS, min_group_samples=MIN_GROUP_SAMPLES_KNN)
        metrics[f'cdp_{prefix_scope[:-1]}'] = cdp

        return metrics


def run_single_experiment(dataset: str, protected_attr: str, stage: str,
                         seed: int, max_epochs: int = 100):
    """Run single experiment with checkpoints and capture comprehensive training and test stats."""

    set_seed(seed)

    # Load and partition data
    encoder_dim = DATA_PARTITION[dataset]["encoder_dim"]
    ds = _load_raw(dataset)
    Xdf, Zdf, S, y = build_partitions(ds, dataset, protected_attr)

    # Preprocessing
    ctX = make_ct(Xdf)
    X = ctX.fit_transform(Xdf).astype("float32") if not Xdf.empty else np.zeros((len(y), 0), "float32")
    ctZ = make_ct(Zdf)
    Z = ctZ.fit_transform(Zdf).astype("float32") if not Zdf.empty else np.zeros((len(y), 0), "float32")

    # Train/test split
    Xtr, Xte, Ztr, Zte, Str, Ste, ytr, yte = train_test_split(
        X, Z, S, y, test_size=0.2, stratify=S, random_state=seed)

    # Determine marginal probabilities of S for 'direct_estimation' stage
    # S needs to be numerical for this. Assuming S is binary (0/1) for simplicity.
    unique_s, counts_s = np.unique(Str, return_counts=True)
    if len(unique_s) == 2:
        p_s0_marginal = counts_s[unique_s == 0].sum() / len(Str) # Probability of S=0
        p_s1_marginal = counts_s[unique_s == 1].sum() / len(Str) # Probability of S=1
    else: # Handle cases where S might not be perfectly binary (e.g., continuous S in linear sim, or more categories)
          # For simplicity in this context, if not binary, just set to 0.5/0.5 or raise error if critical
        logging.warning(f"Sensitive attribute has {len(unique_s)} classes. 'direct_estimation' post-hoc averaging assumes binary S. Using 0.5/0.5 marginals.")
        p_s0_marginal = 0.5
        p_s1_marginal = 0.5
    

    # Model setup
    x_dim, z_dim = Xtr.shape[1], Ztr.shape[1]
    
    # DETERMINE MODEL CLASSES BASED ON DATASET (Linear for German, Non-linear otherwise)
    if dataset == "german":
        EncoderClass = LinearEncoder
        PredictorClass = LinearPredictor
    else:
        EncoderClass = Encoder
        PredictorClass = Predictor
    
    encoder, predictor, adversary = None, None, None

    # Instantiate models based on stage and selected classes (Linear or Non-Linear)
    if stage == "naive":
        predictor = PredictorClass(z_dim + x_dim).to(DEVICE)
        adversary = Adversary(z_dim + x_dim, len(np.unique(Str))).to(DEVICE) if (x_dim + z_dim > 0 and len(np.unique(Str)) > 1) else None
    elif stage == "full_debias":
        total_input_dim = z_dim + x_dim
        v_dim = encoder_dim if total_input_dim > 0 else 0
        encoder = EncoderClass(total_input_dim, v_dim).to(DEVICE) if total_input_dim > 0 else None
        predictor = PredictorClass(v_dim).to(DEVICE)
        adversary = Adversary(v_dim, len(np.unique(Str))).to(DEVICE) if (v_dim > 0 and len(np.unique(Str)) > 1) else None
    elif stage == "ours":
        v_dim = encoder_dim if x_dim > 0 else 0
        encoder = EncoderClass(x_dim, v_dim).to(DEVICE) if x_dim > 0 else None
        predictor_input_dim = z_dim + v_dim
        predictor = PredictorClass(predictor_input_dim).to(DEVICE)
        adversary = Adversary(v_dim, len(np.unique(Str))).to(DEVICE) if (v_dim > 0 and len(np.unique(Str)) > 1) else None
    elif stage == "direct_estimation": # NEW STAGE: Predictor takes Z, V, and S
        v_dim = encoder_dim if x_dim > 0 else 0 # Encoder for X only, as in 'ours'
        encoder = EncoderClass(x_dim, v_dim).to(DEVICE) if x_dim > 0 else None
        predictor_input_dim = z_dim + v_dim + 1 # +1 for the sensitive attribute S
        predictor = PredictorClass(predictor_input_dim).to(DEVICE)
        # Adversary not used for debiasing in this stage, but for MI calculation, use combined features (Z, V, S)
        # However, for the MI of prediction on S, it's I(Y_hat; S), and Y_hat here is already averaged out S,
        # so MI(Y_hat;S) might be zero by construction for the fair prediction.
        # For CLUB MI, we should estimate I(features_for_prediction; S), which is I(Z, V, S; S) -> this is just I(S;S)
        # So we should consider MI calculation carefully for this stage.
        # For consistency with the overall framework's MI calculation, we'll keep the adversary for MI if needed.
        # For the purpose of this stage, the primary objective is post-hoc S averaging, not adversarial debiasing.
        adversary = None # No adversarial debiasing needed here as S is averaged out post-hoc. MI will be calculated on final predictions.
    else:
        raise ValueError(f"Unknown stage: {stage}")

    num_s_classes = len(np.unique(Str))

    # Initialize weights for "ours" stage
    ess = len(ytr)
    if stage == "ours":
        Vtr0 = np.zeros((len(ytr), 0), "float32")
        Vte0 = np.zeros((len(yte), 0), "float32")
        
        if encoder is not None and encoder.net is not None:
            with torch.no_grad():
                encoder.eval()
                if isinstance(encoder, Encoder) or isinstance(encoder, LinearEncoder):
                    Vtr0 = encoder(torch.tensor(Xtr).to(DEVICE)).cpu().numpy()
                    Vte0 = encoder(torch.tensor(Xte).to(DEVICE)).cpu().numpy()
                encoder.train()

        w_tr, ess = compute_weights(Ztr, Vtr0, Str)
        w_te, _ = compute_weights(Zte, Vte0, Ste)
    else:
        w_tr = np.ones_like(ytr, "float32")
        w_te = np.ones_like(yte, "float32")

    # Data loaders
    dl_tr_current = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ztr), torch.tensor(Str), torch.tensor(ytr), torch.tensor(w_tr)), batch_size=512, shuffle=True)
    dl_te_current = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(Zte), torch.tensor(Ste), torch.tensor(yte), torch.tensor(w_te)), batch_size=512, shuffle=False)

    trainer = AdversarialTrainer(stage, encoder, predictor, adversary)

    all_results = []

    for epoch in range(1, max_epochs + 1):
        epoch_pred_losses = []
        epoch_adv_losses = []
        
        # Recalculate weights periodically for "ours" stage if encoder changes V
        if stage == "ours" and epoch % 5 == 0 and encoder is not None and encoder.net is not None:
            with torch.no_grad():
                encoder.eval()
                if isinstance(encoder, Encoder) or isinstance(encoder, LinearEncoder):
                    Vtr = encoder(torch.tensor(Xtr).to(DEVICE)).cpu().numpy()
                    Vte = encoder(torch.tensor(Xte).to(DEVICE)).cpu().numpy()
                encoder.train()

            w_tr, ess = compute_weights(Ztr, Vtr, Str)
            w_te, _ = compute_weights(Zte, Vte, Ste)

            # Re-create DataLoaders with updated weights
            dl_tr_current = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ztr), torch.tensor(Str), torch.tensor(ytr), torch.tensor(w_tr)), batch_size=512, shuffle=True)
            dl_te_current = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(Zte), torch.tensor(Ste), torch.tensor(yte), torch.tensor(w_te)), batch_size=512, shuffle=False)


        # Training steps
        for batch in dl_tr_current: # Use dl_tr_current as it might be updated
            losses = trainer.step(*batch)
            epoch_pred_losses.append(losses["pred_loss"])
            epoch_adv_losses.append(losses["adv_loss"])

        avg_train_pred_loss = np.mean(epoch_pred_losses) if epoch_pred_losses else 0.0
        avg_train_adv_loss = np.mean(epoch_adv_losses) if epoch_adv_losses else 0.0


        # Evaluate at checkpoints (or at every epoch if desired for detailed plots)
        if epoch in CHECKPOINT_EPOCHS:
            # Pass marginal probabilities of S to the prediction function for 'direct_estimation' stage
            p_tr, y_tr, s_tr, w_tr_actual, z_tr, features_tr_for_mi_tensor = trainer._predict_and_collect_data(dl_tr_current, p_s0_marginal, p_s1_marginal)
            train_metrics = trainer._calculate_all_evaluation_metrics(
                p=p_tr, y=y_tr, s=s_tr, w=w_tr_actual, z=z_tr,
                features_for_mi_tensor=features_tr_for_mi_tensor,
                num_s_classes=num_s_classes,
                prefix_scope="tr_"
            )

            p_te, y_te, s_te, w_te_actual, z_te, features_te_for_mi_tensor = trainer._predict_and_collect_data(dl_te_current, p_s0_marginal, p_s1_marginal)
            test_metrics = trainer._calculate_all_evaluation_metrics(
                p=p_te, y=y_te, s=s_te, w=w_te_actual, z=z_te,
                features_for_mi_tensor=features_te_for_mi_tensor,
                num_s_classes=num_s_classes,
                prefix_scope="te_"
            )

            result = {
                "dataset": dataset,
                "protected_attr": protected_attr,
                "stage": stage,
                "seed": seed,
                "epoch": epoch,
                "encoder_dim": encoder_dim,

                # Training losses
                "train_pred_loss": avg_train_pred_loss,
                "train_adv_loss": avg_train_adv_loss,
                
                # All comprehensive training metrics
                **train_metrics,

                # All comprehensive test metrics
                **test_metrics,

                "ess": ess,
            }

            all_results.append(result)

    return all_results


def run_experiments(datasets: List[str] = ["adult", "german", "compas"],
                   stages: List[str] = ["naive", "full_debias", "ours", "direct_estimation"], # ADDED direct_estimation
                   seeds: List[int] = list(range(5)),
                   max_epochs: int = 100):
    """Run comprehensive experiments"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = Path(f"adversarial_results_CLUB_MI_{timestamp}.csv")

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    total_configs = sum(len(DATA_PARTITION[ds]["protected_attrs"]) for ds in datasets) * len(stages) * len(seeds)

    logging.info(f"--- Starting comprehensive experiments ---")
    logging.info(f"Datasets: {datasets}")
    logging.info(f"Stages: {stages}")
    logging.info(f"Seeds: {seeds}")
    logging.info(f"Max Epochs: {max_epochs}")
    logging.info(f"Checkpoint Epochs: {CHECKPOINT_EPOCHS}")
    logging.info(f"Total experiment configurations: {total_configs}")
    logging.info(f"Results will be saved to: {outfile}")

    all_results = []
    with tqdm(total=total_configs, desc="Running Experiments") as pbar:
        for dataset in datasets:
            for protected_attr in DATA_PARTITION[dataset]["protected_attrs"].keys():
                for stage in stages:
                    for seed in seeds:
                        log_msg = f"--- Running experiment: Dataset={dataset}, Protected_attr={protected_attr}, Stage={stage}, Seed={seed} ---"
                        logging.info(log_msg)
                        try:
                            results = run_single_experiment(
                                dataset, protected_attr, stage, seed, max_epochs)
                            all_results.extend(results)

                            pbar.set_postfix({
                                'ds': dataset,
                                'attr': protected_attr,
                                'stg': stage,
                                'seed': seed
                            })

                        except Exception as e:
                            logging.error(f"Failed to run experiment: Dataset={dataset}, Attr={protected_attr}, Stage={stage}, Seed={seed}. Error: {e}")
                            logging.exception(e) # Print full traceback
                        pbar.update(1)

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(outfile, index=False)
        logging.info(f"All experiments completed. Results saved to: {outfile}")
        logging.info(f"Total individual results (checkpoints): {len(all_results)}")

        # Quick summary for the full run - Adjusting summary logs to reflect new column names
        logging.info("\n--- Overall Summary by Stage (P distribution, final epoch) ---")
        for stage in df['stage'].unique():
            # Filter for the last checkpoint epoch
            stage_data = df[(df['stage'] == stage) & (df['epoch'] == max_epochs)]
            if not stage_data.empty:
                # Use 'te_' prefix for test data metrics in summary
                logging.info(f"  {stage:<18}: AUC={stage_data['auc_te_P'].mean():.3f} ± {stage_data['auc_te_P'].std():.3f}, "
                             f"DP={stage_data['dp_te_P'].mean():.3f} ± {stage_data['dp_te_P'].std():.3f}, "
                             f"CDP={stage_data['cdp_te'].mean():.3f} ± {stage_data['cdp_te'].std():.3f}, "
                             f"MI(F;S)={stage_data['mi_te'].mean():.3f} ± {stage_data['mi_te'].std():.3f}")

        if "ours" in df['stage'].unique():
            logging.info("\n--- Overall Summary for 'ours' stage (P-tilde distribution, final epoch) ---")
            stage_data = df[(df['stage'] == "ours") & (df['epoch'] == max_epochs)]
            if not stage_data.empty:
                # Use 'te_' prefix for test data metrics in summary
                logging.info(f"  ours (P-tilde)     : AUC={stage_data['auc_te_Ptilde'].mean():.3f} ± {stage_data['auc_te_Ptilde'].std():.3f}, "
                             f"DP={stage_data['dp_te_Ptilde'].mean():.3f} ± {stage_data['dp_te_Ptilde'].std():.3f}")
    else:
        logging.info("No results were generated for the comprehensive experiment run.")
    logging.info("--- Comprehensive experiments finished ---")

    return all_results

def main():
    """Main execution function - runs comprehensive experiments as defined."""

    datasets_to_run = ["german", "compas", "adult"]
    stages_to_run = ["naive", "full_debias", "ours", "direct_estimation"] # ADDED 'direct_estimation'
    seeds_to_run = list(range(5)) # 5 seeds: 0, 1, 2, 3, 4
    max_epochs_to_run = 100

    print(f"Starting comprehensive experiment run on {DEVICE}...")
    print(f"Datasets: {datasets_to_run}")
    print(f"Stages: {stages_to_run}")
    print(f"Seeds: {seeds_to_run}")
    print(f"Max Epochs: {max_epochs_to_run}")
    print(f"Checkpoints: {CHECKPOINT_EPOCHS}")
    print("-" * 60)

    run_experiments(
        datasets=datasets_to_run,
        stages=stages_to_run,
        seeds=seeds_to_run,
        max_epochs=max_epochs_to_run
    )

if __name__ == "__main__":
    main()