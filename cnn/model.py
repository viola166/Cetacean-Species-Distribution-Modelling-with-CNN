import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score
from data.prepareDataset import preprocess_and_build
from data.prepareCSVData import load_csv_data, SPECIES_LIST
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

N_EPOCHS = 60

# ---------------------------------------------------------------- 2. Architecture
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, strided_dilated=True, final=False, padding=2):
#         super().__init__()
#         s, d = (2, 2) if strided_dilated else (1, 1)
#         self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=s,
#                               dilation=d, padding=padding if strided_dilated else 1,
#                               bias=False)
#         self.bn = nn.BatchNorm2d(out_ch)
#         self.pool = nn.AdaptiveAvgPool2d(1) if final else nn.MaxPool2d(2)

#     def forward(self, x):
#         return self.pool(F.relu(self.bn(self.conv(x))))

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, strided_dilated=True, final=False, pool=True, padding=2):
        super().__init__()
        s, d = (2, 2) if strided_dilated else (1, 1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=s,
                              dilation=d, padding=padding if strided_dilated else 1,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        if final:
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class SpeciesCNN(nn.Module):
    def __init__(self, in_channels=17, n_species=len(SPECIES_LIST), dropout=0.4, head_hidden=0):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 64, strided_dilated=True, pool=False),
            ConvBlock(64, 128, strided_dilated=False, pool=True),
            ConvBlock(128, 256, strided_dilated=False, pool=False),
            ConvBlock(256, 512, strided_dilated=False, final=True),  # last block: plain conv
        )
        self.in_channels = in_channels
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout                 # the float, for saving
        # self.fc = nn.Linear(512, n_species)      # fully connected layers
        
        self.head_hidden = head_hidden

        if head_hidden == 0:
            self.heads = nn.ModuleList([nn.Linear(512, 1) for _ in range(n_species)])
        else:
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(512, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
                for _ in range(n_species)
            ])

        # MLR: c = 1/(1+b^2) per species; b=1 -> c=0.5 at init
        self.b = nn.Parameter(torch.ones(n_species))

        self.preproc = None   # set before train_sdm: dict with transforms, stats, etc.

    # def forward_logits(self, x):
    #     z = self.features(x).flatten(1)        # B x 512
    #     return self.fc(self.dropout(z))

    def forward_logits(self, x):
        z = self.dropout(self.backbone(x).flatten(1))            # B x 512
        return torch.cat([h(z) for h in self.heads], dim=1)      # B x S
    
    def forward(self, x):
        return torch.sigmoid(self.forward_logits(x))   # species pseudo-probabilities

    @property
    def c(self):                       # (S,) learned label frequency
        return 1.0 / (1.0 + self.b ** 2)

    @torch.no_grad()
    def predict_presence(self, x):     # P(present | x), MLR-corrected
        z = self.forward_logits(x)
        return torch.sigmoid(z + torch.log1p(self.b ** 2))

    # prepare_data must be called before and its outputs serve as input + important for initializing model with correct num of input channels
    def train_sdm(self, loaders, pi=None, device="cuda", loss_type="mlr", epochs=N_EPOCHS, run_dir=None):
        """
        Train the CNN with the nnPU loss and keep the weights of the epoch with the best
        mean validation AUC.

        loaders : dict {"train", "val", "test"} -> DataLoader yielding (x, y, w), as returned
                by preprocess_and_build. Only "train" and "val" are used here.
        pi      : (S,) tensor, class prior per species for nnPU
        device  : "cuda" or "cpu"
        epochs  : number of passes over the training data

        Returns the model (self) with the best-epoch weights loaded.
        """
        # ---- setup: model and prior must live on the same device as the batches ----
        self.to(device)
        if pi is not None:
            pi = pi.to(device)

        # plain SGD (no momentum); weight_decay shrinks the weights a little each step (regularisation)
        # opt = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.0, weight_decay=1e-3)
        other = [p for n, p in self.named_parameters() if n != "b"]
        if loss_type == "mlr":
            opt = torch.optim.SGD(
                [{"params": other},
                {"params": [self.b], "lr": 1e-2, "weight_decay": 0.0}],   # tune this lr
                lr=1e-3, momentum=0.0, weight_decay=1e-3)
        else:
            opt = torch.optim.SGD(other, lr=1e-3, momentum=0.0, weight_decay=1e-3)
        # learning rate x 0.1 once, at the halfway epoch: big steps first, fine-tuning afterwards
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[epochs // 2], gamma=0.1)

        # best-model tracking; -1.0 makes sure the first epoch counts as an improvement
        best_auc, best_state, best_epoch = -1.0, None, None
        history = []

        for epoch in range(epochs):
            # ---- training pass ----
            self.train()                       # dropout on, BatchNorm uses batch statistics
            running, n = 0.0, 0                # for the sample-weighted mean loss of this epoch
            lr = opt.param_groups[0]["lr"]

            n_batches = len(loaders["train"])
            log_every = max(1, n_batches // 10)  
            for i, (x, y, w) in enumerate(loaders["train"]):
                # x: (B, C, 13, 13) patches, y: (B, S) multi-hot labels, w: (B, S) de-clustering weights
                x, y, w = x.to(device), y.to(device), w.to(device)

                # loss = nnpu_loss(self.forward_logits(x), y, pi, w)    # self.forward_logits(x) = logits (B, S); y=1 positive, y=0 unlabelled
                logits = self.forward_logits(x)
                loss = (mlr_loss(logits, y, self.b, w) if loss_type == "mlr"
                        else nnpu_loss(logits, y, pi, w))

                opt.zero_grad()                # clear gradients from the previous batch
                loss.backward()                # gradients of the loss w.r.t. all weights
                opt.step()                     # update the weights

                running += loss.item() * x.size(0)     # x.size(0) = actual batch size B
                n += x.size(0)

                if i % log_every == 0:
                    print(f"  ep {epoch} batch {i}/{n_batches} loss {loss.item():.4f} "
                    f"running {running / n:.4f}", flush=True)
            sched.step()                       # advance the LR schedule once per epoch (after the optimizer steps)

            # ---- validation pass ----
            self.eval()                        # dropout off, BatchNorm uses running statistics
            logits, ys = [], []
            with torch.no_grad():              # no gradient tracking: saves memory and time
                for x, y, _ in loaders["val"]: # weights are not used for the AUC
                    logits.append(self.forward_logits(x.to(device)).cpu())   # back to CPU for numpy / sklearn
                    ys.append(y)                              # labels never left the CPU

            # join batches into (N_val, S); sigmoid turns logits into probabilities
            p, y = torch.cat(logits).sigmoid().numpy(), torch.cat(ys).numpy()

            # per-species AUC; skip species that have no positives or only positives in the val split
            # (y=0 means "unlabelled" here, so AUC is a relative measure: use it to compare runs)
            aucs = {sp: roc_auc_score(y[:, s], p[:, s]) for s, sp in enumerate(SPECIES_LIST)
                    if 0 < y[:, s].sum() < len(y)}
            mean_auc = float(np.mean(list(aucs.values()))) if aucs else float("nan")

            history.append({"epoch": epoch, "lr": lr, "train_loss": running/n, "val_mean_auc": mean_auc, **{f"auc_{sp}": v for sp, v in aucs.items()}, "mean_c": float(self.c.mean()) if loss_type == "mlr" else float("nan"),})

            # ---- keep the best epoch ----
            if mean_auc > best_auc:            # NaN compares False, so it never counts as an improvement
                best_auc = mean_auc
                best_epoch = epoch
                # detached CPU clone: later training steps cannot overwrite this snapshot
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

            # ---- logging ----
            print(epoch, f"loss {running / n:.4f}  val AUC {mean_auc:.3f}",
                {k: round(v, 3) for k, v in aucs.items()})

        # ---- restore the weights of the best epoch (not the last one) ----
        if best_state is not None:             # None if no epoch had a valid AUC
            self.load_state_dict(best_state)
        self.history = pd.DataFrame(history)
        self.best_epoch, self.best_auc = best_epoch, best_auc
        if run_dir is not None:
            self.save_run(run_dir, pi)
        return self


    def save_run(self, run_dir, pi, extra=None):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # weights of the best epoch (already loaded into self); save the state_dict, not the whole model
        torch.save({k: v.cpu() for k, v in self.state_dict().items()}, run_dir / "model_best.pt")

        self.history.to_csv(run_dir / "history.csv", index=False)

        meta = {
            "species_list": list(SPECIES_LIST),
            "in_channels": self.in_channels,
            "dropout": self.dropout_p,
            "best_epoch": self.best_epoch,
            "best_val_mean_auc": self.best_auc,
            "n_epochs": len(self.history),
            "preprocessing": self.preproc,
            "head_hidden": self.head_hidden,
            "pi": None if pi is None else pi.cpu().tolist(),
            "loss_type": getattr(self, "loss_type", None),
            "c_hat": self.c.detach().cpu().tolist(),
            **(extra or {}),      # e.g. fold, in_channels, use_mask, max_ratio, batch_size, seed
        }
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # @classmethod
    # def load_run(cls, run_dir, device="cpu"):
    #     run_dir = Path(run_dir)
    #     with open(run_dir / "meta.json") as f:
    #         meta = json.load(f)
    #     model = cls(in_channels=meta["in_channels"],
    #         n_species=len(meta["species_list"]),
    #         dropout=meta.get("dropout", 0.4),
    #         head_hidden=meta.get("head_hidden", 0))
    #     model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device))
    #     model.preproc = meta["preprocessing"]
    #     return model.to(device).eval(), meta
    @classmethod
    def load_run(cls, run_dir, device="cpu"):
        run_dir = Path(run_dir)
        with open(run_dir / "meta.json") as f:
            meta = json.load(f)
        model = cls(in_channels=meta["in_channels"],
            n_species=len(meta["species_list"]),
            dropout=meta.get("dropout", 0.4),
            head_hidden=meta.get("head_hidden", 0))

        sd = torch.load(run_dir / "model_best.pt", map_location=device)
        sd.setdefault("b", model.b.detach().cpu())   # older (pre-MLR) checkpoints have no "b"
        model.load_state_dict(sd)

        model.preproc = meta["preprocessing"]
        return model.to(device).eval(), meta

    @torch.no_grad()
    def evaluate_sdm(self, loader, device="cuda"):
        """Per-species AUC on one loader (e.g. the test split). Returns {species: auc}."""
        self.to(device).eval()
        logits, ys = [], []
        for x, y, _ in loader:
            logits.append(self.forward_logits(x.to(device)).cpu()); ys.append(y)
        p, y = torch.cat(logits).sigmoid().numpy(), torch.cat(ys).numpy()
        return {sp: float(roc_auc_score(y[:, s], p[:, s]))
                for s, sp in enumerate(SPECIES_LIST) if 0 < y[:, s].sum() < len(y)}


    @torch.no_grad()
    def evaluate_boyce(self, loader, device="cuda", min_presences=10):
        """Per-species continuous Boyce index on one loader. Returns {species: boyce}."""
        self.to(device).eval()
        zs, ys, ws = [], [], []
        for x, y, w in loader:
            zs.append(self.forward_logits(x.to(device)).cpu()); ys.append(y); ws.append(w)
        z, y, w = torch.cat(zs).numpy(), torch.cat(ys).numpy(), torch.cat(ws).numpy()

        out = {}
        for s, sp in enumerate(SPECIES_LIST):
            pres = y[:, s] == 1
            if pres.sum() < min_presences:
                continue
            b, _, _ = boyce_index(z[pres, s], z[:, s], w_pres=w[pres, s], w_bg=w[:, s])
            if not np.isnan(b):
                out[sp] = b
        return out

# ---------------------------------------------------------------- 3. Loss
def species_weights(labels: np.ndarray, mask: np.ndarray):
    """pi_k = observation frequency of species k in the training set;
    weight = 1/pi_k. Returns (K,) tensor."""
    pi = (labels * mask).sum(0) / max(len(labels), 1)
    pi = np.clip(pi, 1e-6, None)
    return torch.tensor(1.0 / pi, dtype=torch.float32)


def masked_weighted_bce(logits, y, mask, inv_freq, absence_scale=0.1):
    """BCE evaluated only where mask == 1.
    Weight: 1/pi_k for y=1, 0.1/pi_k for y=0 (pseudo-absences)."""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    w = inv_freq.unsqueeze(0) * torch.where(y > 0.5, 1.0, absence_scale)
    return (bce * w * mask).sum() / mask.sum().clamp(min=1.0)

def boyce_index(s_pres, s_bg, n_bins=100, window=0.1, w_pres=None, w_bg=None):
    """
    s_pres : scores at recorded presences      (n_p,)
    s_bg   : scores at background/all samples  (n_b,)
    window : width of the moving window, as a fraction of the score range
    Returns (boyce, bin_centers, pe_ratio).
    """
    s_pres, s_bg = np.asarray(s_pres), np.asarray(s_bg)
    if len(s_pres) < 5:
        return np.nan, None, None

    w_pres = np.ones_like(s_pres) if w_pres is None else np.asarray(w_pres)
    w_bg   = np.ones_like(s_bg)   if w_bg   is None else np.asarray(w_bg)

    lo, hi = s_bg.min(), s_bg.max()
    if hi - lo < 1e-9:
        return np.nan, None, None
    half = window * (hi - lo) / 2
    centers = np.linspace(lo + half, hi - half, n_bins)

    P, E = [], []
    for c in centers:
        P.append(w_pres[(s_pres >= c - half) & (s_pres <= c + half)].sum() / w_pres.sum())
        E.append(w_bg  [(s_bg   >= c - half) & (s_bg   <= c + half)].sum() / w_bg.sum())
    P, E = np.array(P), np.array(E)

    ok = E > 0                                   # skip windows with no background
    if ok.sum() < 3:
        return np.nan, None, None
    pe = P[ok] / E[ok]
    rho = spearmanr(centers[ok], pe).correlation
    return float(rho), centers[ok], pe


def nnpu_loss(logits, y, pi, w=None, gamma=1.0, beta=0.0):
    """
    Non-negative PU risk (Kiryo et al., 2017), evaluated per species.

    logits : (B, S) raw model outputs
    y      : (B, S) 1 = presence (labelled positive), 0 = unlabelled
    pi     : (S,) class prior per species, in (0, 1)
    w      : (B, S) optional sample weights
    gamma  : step factor of the corrective (ascent) step, in (0, 1]
    beta   : tolerance; the correction triggers when neg_risk < -beta
    """
    pos = (y == 1).float()
    unl = (y == 0).float()
    if w is None:
        w = torch.ones_like(y)

    loss_pos = F.softplus(-logits)   # l(f, +1)
    loss_neg = F.softplus(logits)    # l(f, -1)

    n_pos = (w * pos).sum(0).clamp(min=1)
    n_unl = (w * unl).sum(0).clamp(min=1)

    r_p_pos = (w * pos * loss_pos).sum(0) / n_pos   # E_P[l(f,+1)]
    r_p_neg = (w * pos * loss_neg).sum(0) / n_pos   # E_P[l(f,-1)]
    r_u_neg = (w * unl * loss_neg).sum(0) / n_unl   # E_U[l(f,-1)]

    neg_risk = r_u_neg - pi * r_p_neg               # (S,)

    # per species: normal risk if neg_risk is acceptable,
    # otherwise ascent on neg_risk only (minimizing -gamma * neg_risk)
    risk = torch.where(
        neg_risk >= -beta,
        pi * r_p_pos + neg_risk,
        -gamma * neg_risk,
    )
    return risk.sum()               # or risk.mean()


def mlr_loss(logits, y, b, w=None, eps=1e-8):
    """
    Modified logistic regression for PU learning (Jaskie, Elkan & Spanias, 2019).
    Models P(labelled | x) = 1 / (1 + b^2 + exp(-z)), so label frequency c = 1/(1+b^2).

    logits : (B, S) raw model outputs z
    y      : (B, S) 1 = recorded presence, 0 = unlabelled
    b      : (S,)   learned parameter, one per species
    w      : (B, S) optional sample weights
    """
    y = y.float()
    b2 = b ** 2 + eps
    log_den = torch.logaddexp(torch.log1p(b2).expand_as(logits), -logits)   # log(1 + b^2 + e^{-z})
    log_g   = -log_den                                                      # log P(s=1|x)
    log_1mg = torch.logaddexp(torch.log(b2).expand_as(logits), -logits) - log_den  # log P(s=0|x)

    nll = -(y * log_g + (1.0 - y) * log_1mg)                                # (B, S)
    if w is None:
        return nll.mean()
    return ((w * nll).sum(0) / w.sum(0).clamp(min=1e-8)).mean()