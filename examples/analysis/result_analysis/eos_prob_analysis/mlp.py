import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------
BASE_DIR = "/root/vllm/vllm/examples/analysis/data/eos_result/"

MODEL_NAMES = ["llama"]
DATASET_NAMES = ["alpaca"]

DATASET_NAME_MAP = {"alpaca": "Alpaca"}
MODEL_NAME_MAP = {"llama": "Llama"}

TRAIN_END = 4000          # first 4000 for training
MAX_ROWS = 10000          # use eos_df.loc[:MAX_ROWS]
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
SEED = 42

# Pairwise accuracy evaluation:
# - full pairwise is O(N^2); for N~6000 it's 36M pairs which is heavy.
# - We provide Monte-Carlo sampling for scalability.
PAIR_SAMPLES = 2_000_000  # set to None to do full (may be slow / memory heavy)
CHUNK_PAIRS = 500_000     # chunking for sampling evaluation


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Data loading (same semantics)
# -----------------------------
def load_eos_prob_data():
    dfs = []
    for model_name in MODEL_NAMES:
        for dataset_name in DATASET_NAMES:
            path = f"{BASE_DIR}{model_name}_{dataset_name}_eos_prob_result.csv"
            df = pd.read_csv(path)
            df["model_dataset"] = MODEL_NAME_MAP[model_name] + " " + DATASET_NAME_MAP[dataset_name]
            dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all[df_all["eos_prob"] != 0]
    return df_all


def build_prompt_len_table(eos_df, max_rows=MAX_ROWS):
    return (
        eos_df.loc[:max_rows, ["model_dataset", "request_id", "prompt_len", "token_num"]]
        .groupby(["model_dataset", "request_id"])
        .max()
        .reset_index()
    )


# -----------------------------
# Model: two-layer perceptron
# -----------------------------
class TwoLayerMLP(nn.Module):
    """Linear -> ReLU -> Linear; hidden=512"""
    def __init__(self, in_dim=1, hidden_dim=512, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x


# -----------------------------
# Pairwise ranking accuracy
# -----------------------------
@torch.no_grad()
def pairwise_accuracy(pred: torch.Tensor, truth: torch.Tensor,
                      samples: int | None = PAIR_SAMPLES,
                      chunk: int = CHUNK_PAIRS,
                      device: torch.device | None = None) -> float:
    """
    pred, truth: shape [N] on CPU or GPU
    Accuracy definition matches your original:
      correct if (pred_i >= pred_j and truth_i <= truth_j) OR (pred_i <= pred_j and truth_i >= truth_j)

    For scalability, default uses random pair sampling.
    """
    if device is None:
        device = pred.device

    pred = pred.to(device)
    truth = truth.to(device)

    n = pred.numel()
    if n < 2:
        return float("nan")

    # Full exact (O(N^2)) — only do if explicitly requested
    if samples is None:
        # Compute in blocks to avoid gigantic NxN tensors
        # Note: still heavy; use only for small N.
        correct = 0
        total = 0
        block = 1024
        for i0 in range(0, n, block):
            i1 = min(i0 + block, n)
            pi = pred[i0:i1].unsqueeze(1)   # [b,1]
            ti = truth[i0:i1].unsqueeze(1)  # [b,1]

            pj = pred.unsqueeze(0)          # [1,N]
            tj = truth.unsqueeze(0)         # [1,N]

            f1 = pi >= pj
            f2 = ti <= tj
            f3 = pi <= pj
            f4 = ti >= tj
            ok = (f1 & f2) | (f3 & f4)

            correct += ok.sum().item()
            total += ok.numel()
        return correct / total

    # Sampling mode
    correct = 0
    total = 0
    remaining = samples
    while remaining > 0:
        cur = min(chunk, remaining)
        i = torch.randint(0, n, (cur,), device=device)
        j = torch.randint(0, n, (cur,), device=device)

        pi = pred[i]
        pj = pred[j]
        ti = truth[i]
        tj = truth[j]

        ok = ((pi >= pj) & (ti <= tj)) | ((pi <= pj) & (ti >= tj))
        correct += ok.sum().item()
        total += cur
        remaining -= cur

    return correct / total


# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(SEED)
    device = pick_device()
    print(f"[INFO] device = {device}")

    eos_df = load_eos_prob_data()
    prompt_len_result = build_prompt_len_table(eos_df)

    # Train/test split like your code
    train_df = prompt_len_result.iloc[:TRAIN_END].copy()
    test_df = prompt_len_result.iloc[TRAIN_END:].copy()
    print(f"[INFO] total={len(prompt_len_result)}, train={len(train_df)}, test={len(test_df)}")

    # Features: prompt_len only, shape [N,1]
    X_train = train_df["prompt_len"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_train = train_df["token_num"].to_numpy(dtype=np.float32).reshape(-1, 1)

    X_all = prompt_len_result["prompt_len"].to_numpy(dtype=np.float32).reshape(-1, 1)

    # Standardize (like sklearn StandardScaler), using train stats
    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True) + 1e-6

    X_train = (X_train - x_mean) / x_std
    X_all = (X_all - x_mean) / x_std

    # To torch
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_all_t = torch.from_numpy(X_all).to(device)

    # Model
    model = TwoLayerMLP(in_dim=1, hidden_dim=512, out_dim=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training loop (MSE regression)
    n = X_train_t.size(0)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # shuffle
        idx = torch.randperm(n, device=device)
        Xs = X_train_t[idx]
        ys = y_train_t[idx]

        total_loss = 0.0
        for i0 in range(0, n, BATCH_SIZE):
            i1 = min(i0 + BATCH_SIZE, n)
            xb = Xs[i0:i1]
            yb = ys[i0:i1]

            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * (i1 - i0)

        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            print(f"[TRAIN] epoch {epoch:4d}/{EPOCHS}  mse={total_loss / n:.6f}")

    # Predict on all rows
    model.eval()
    with torch.no_grad():
        pred_all = model(X_all_t).squeeze(1)  # [N]
    pred_all_cpu = pred_all.detach().cpu().numpy()

    prompt_len_result["test_y"] = pred_all_cpu

    # Evaluate accuracy on test split (same semantics)
    # Use GPU tensors for evaluation
    test_pred = torch.from_numpy(prompt_len_result.iloc[TRAIN_END:]["test_y"].to_numpy(np.float32)).to(device)
    test_truth = torch.from_numpy(prompt_len_result.iloc[TRAIN_END:]["token_num"].to_numpy(np.float32)).to(device)

    acc = pairwise_accuracy(test_pred, test_truth, samples=PAIR_SAMPLES, chunk=CHUNK_PAIRS, device=device)
    print(f"[EVAL] pairwise ranking accuracy (test) = {acc:.6f}  (samples={PAIR_SAMPLES})")


if __name__ == "__main__":
    main()
