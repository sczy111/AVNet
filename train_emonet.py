import argparse
from pathlib import Path
import json
import csv
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from emonet.models import EmoNet  # adjust import if your module path differs
from tqdm import tqdm


# Utils: seeds, metrics, I/O
def set_seed(seed: int):
    # Make results more reproducible across Python, NumPy, and PyTorch (CPU+CUDA).
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ccc(x, y, eps=1e-8):
    """
    Concordance Correlation Coefficient ([-1, 1]).
    Higher is better; 1.0 = perfect agreement.
    """
    x = x.float(); y = y.float()
    xm, ym = x.mean(), y.mean()
    xv, yv = x.var(unbiased=False), y.var(unbiased=False)
    cov = ((x - xm) * (y - ym)).mean()
    return (2 * cov) / (xv + yv + (xm - ym) ** 2 + eps)

def ccc_loss(x, y): return 1.0 - ccc(x, y)

def rmse(x, y): return torch.sqrt(torch.mean((x - y) ** 2))
def mae(x, y):  return torch.mean(torch.abs(x - y))

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# Canonical class orders used by EmoNet expression head.
CANONICAL_8 = ["neutral","happy","sad","surprise","fear","disgust","anger","contempt"]
CANONICAL_5 = ["neutral","happy","sad","surprise","fear"] 

# Normalize incoming label strings to lowercase without surrounding spaces
def normalize_label(s: str) -> str:
    return str(s).strip().lower()

def build_fixed_label_map(nclasses: int):
    """
    Fixed label mapping (name → id) matching EmoNet’s expected class order.
    Restrict to 5 or 8 classes for consistency with pretrained heads.
    """
    if nclasses == 8:
        order = CANONICAL_8
    elif nclasses == 5:
        order = CANONICAL_5
    else:
        raise ValueError("nclasses must be 5 or 8")
    return {name: i for i, name in enumerate(order)}



#  dataset 
class EmoNetCSV(Dataset):
    """
    CSV columns expected: pth,label,valence,arousal
      - pth     : image file name or relative path
      - label   : string expression (optional use)
      - valence : float in [-1,1]
      - arousal : float in [-1,1]
    """
    def __init__(self, csv_path, root, size=256, use_expr=True, label2id=None, augment=False):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)
        self.size = size
        self.use_expr = use_expr
        self.label2id = label2id or {}
        self.augment = augment

        required = ["pth", "valence", "arousal"]
        for c in required:
            assert c in self.df.columns, f"CSV must contain column '{c}'"
        if self.use_expr:
            assert "label" in self.df.columns, "CSV must contain 'label' when use_expr=True"

    def __len__(self): return len(self.df)

    def _augment(self, img):
        # mild & expression-safe aug
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        if random.random() < 0.3:
            h, w = img.shape[:2]
            ang = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        if random.random() < 0.3:
            alpha = 1.0 + random.uniform(-0.1, 0.1) # contrast
            beta = random.uniform(-10, 10)          # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img

    def __getitem__(self, idx):
        # Load & preprocess image
        row = self.df.iloc[idx]
        img_path = self.root / str(row["pth"])
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(img_path)
        bgr = cv2.resize(bgr, (self.size, self.size))
        if self.augment: bgr = self._augment(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        v = float(row["valence"]); a = float(row["arousal"])
        y = {"valence": torch.tensor(v, dtype=torch.float32),
             "arousal": torch.tensor(a, dtype=torch.float32)}
        if self.use_expr:
            lab = normalize_label(row["label"])
            y["expr"] = torch.tensor(self.label2id[lab], dtype=torch.long)

        return x, y


# Evaluation
def evaluate(model, loader, device, use_expr):
    """
    Run validation over a loader and compute VA metrics (CCC/RMSE/MAE),
    plus expression accuracy if enabled.
    """
    model.eval()
    v_pred, v_true, a_pred, a_true = [], [], [], []
    expr_correct, expr_total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            vp = out["valence"].view(-1).cpu()
            ap = out["arousal"].view(-1).cpu()
            v_pred.append(vp); a_pred.append(ap)
            v_true.append(y["valence"]); a_true.append(y["arousal"])
            if use_expr and "expression" in out and "expr" in y:
                pred = out["expression"].argmax(1).cpu()
                expr_correct += (pred == y["expr"]).sum().item()
                expr_total += pred.numel()
    v_pred = torch.cat(v_pred); v_true = torch.cat(v_true)
    a_pred = torch.cat(a_pred); a_true = torch.cat(a_true)

    metrics = {
        "ccc_v": ccc(v_pred, v_true).item(),
        "ccc_a": ccc(a_pred, a_true).item(),
        "rmse_v": rmse(v_pred, v_true).item(),
        "rmse_a": rmse(a_pred, a_true).item(),
        "mae_v": mae(v_pred, v_true).item(),
        "mae_a": mae(a_pred, a_true).item(),
    }
    metrics["ccc_mean"] = 0.5 * (metrics["ccc_v"] + metrics["ccc_a"])
    if use_expr and expr_total > 0:
        metrics["expr_acc"] = expr_correct / expr_total
    return metrics

# One training epoch
def train_one_epoch(model, loader, device, optimizer, scaler, use_expr, lambda_expr=1.0, epoch = 1, epochs = 1):
    model.train()
    ce = nn.CrossEntropyLoss() if use_expr else None
    running = {"loss": 0.0, "loss_va": 0.0, "loss_expr": 0.0}
    n = 0

    bar = tqdm(loader, desc=f"Train Epoch {epoch}/{epochs}", leave=False)
    for x, y in bar:
        x = x.to(device, non_blocking=True)
        v = y["valence"].to(device)
        a = y["arousal"].to(device)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            out = model(x)

            # VA regression losses
            loss_v = ccc_loss(out["valence"].view(-1), v)
            loss_a = ccc_loss(out["arousal"].view(-1), a)
            loss_va = loss_v + loss_a
            loss = loss_va
            if use_expr and "expression" in out and "expr" in y:
                logits = out["expression"]
                loss_expr = ce(logits, y["expr"].to(device))
                loss = loss + lambda_expr * loss_expr
            else:
                loss_expr = torch.tensor(0.0, device=device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

        bs = x.size(0); n += bs
        running["loss"] += loss.item() * bs
        running["loss_va"] += loss_va.item() * bs
        running["loss_expr"] += loss_expr.item() * bs

        # live numbers on the bar
        bar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "va": f"{loss_va.item():.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
        })

    # Normalize by total samples
    for k in running: running[k] /= max(1, n)
    return running



#  main 
def main():
    ap = argparse.ArgumentParser()
    # data paths
    ap.add_argument("--train_csv",  type=str, required=True,
                    help="CSV for training split (columns: pth[,label],valence,arousal)")
    ap.add_argument("--test_csv",   type=str, required=True,
                    help="CSV for test/validation split (same columns)")
    ap.add_argument("--train_root", type=str, required=True,
                    help="Folder with training images (paths in train_csv are relative to here)")
    ap.add_argument("--test_root",  type=str, required=True,
                    help="Folder with test images (paths in test_csv are relative to here)")
    # model / training
    ap.add_argument("--nclasses",  type=int, default=8, choices=[5, 8], help="expression classes")
    ap.add_argument("--use_expr",  action="store_true",
                    help="include expression head training (needs 'label' column)")
    ap.add_argument("--epochs",    type=int, default=40)
    ap.add_argument("--batch",     type=int, default=32)
    ap.add_argument("--size",      type=int, default=256)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--lr",        type=float, default=3e-4, help="LR for trainable (last) layers")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lambda_expr",  type=float, default=1.0)
    ap.add_argument("--warmup_epochs", type=int, default=0,
                    help="epochs to keep backbone frozen (already frozen by default)")
    ap.add_argument("--unfreeze_backbone_after", type=int, default=0,
                    help="epoch to start unfreezing backbone (>0 enables). Example: 10")
    ap.add_argument("--outdir", type=str, default="runs/emonet_train")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--labelmap_out", type=str, default="runs/label2id.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # datasets
    df_train = pd.read_csv(args.train_csv)
    label2id = None
    if args.use_expr:
        label2id = build_fixed_label_map(args.nclasses)
        # sanity-check dataset labels
        seen = set(df_train["label"].astype(str).str.strip().str.lower().unique())
        expected = set(label2id.keys())
        unknown = seen - expected
        if unknown:
            raise ValueError(
                f"Unknown labels in CSV: {sorted(unknown)}. "
                f"Expected one of: {sorted(expected)} (case-insensitive)."
            )
        print(f"Label map (fixed to EmoNet order): {label2id}")

    # Datasets & loaders
    train_ds = EmoNetCSV(args.train_csv, args.train_root, size=args.size,
                         use_expr=args.use_expr, label2id=label2id, augment=True)
    test_ds  = EmoNetCSV(args.test_csv,  args.test_root,  size=args.size,
                         use_expr=args.use_expr, label2id=label2id, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Model: initialize with desired expression classes 
    model = EmoNet(n_expression=args.nclasses).to(device)

    # load pretrained weights
    default_pretrained = Path(__file__).parent / "pretrained" / f"emonet_{args.nclasses}.pth"
    if default_pretrained.exists():
        print(f"Loading pretrained weights: {default_pretrained}")
        state = torch.load(str(default_pretrained), map_location="cpu")
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)

    # only train trainable (last) layers by default
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if (args.amp and device == "cuda") else None

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    best, best_path = -1e9, outdir / "ckpt_best.pth"
    log_csv = outdir / "metrics.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["epoch","train_loss","train_va","train_expr",
                  "ccc_v","ccc_a","ccc_mean","rmse_v","rmse_a","mae_v","mae_a"]
        if args.use_expr: header.append("expr_acc")
        w.writerow(header)

    # optional: unfreeze backbone after some epochs
    def unfreeze_backbone():
        unfrozen = 0
        for _, p in model.named_parameters():
            if not p.requires_grad:
                p.requires_grad = True
                unfrozen += 1
        print(f"Unfroze {unfrozen} params. Consider lowering LR.")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Optional staged fine-tuning: unfreeze backbone at a chosen epoch
        if args.unfreeze_backbone_after > 0 and epoch == args.unfreeze_backbone_after:
            unfreeze_backbone()
            for g in optimizer.param_groups:
                g["lr"] = args.lr * 0.3             # Reduce LR to avoid destabilizing training after unfreezing
            print(f"Lowered LR to {optimizer.param_groups[0]['lr']} after unfreezing.")

        # Train + Validate
        tr = train_one_epoch(model, train_loader, device, optimizer, scaler,
                     use_expr=args.use_expr, lambda_expr=args.lambda_expr,
                     epoch=epoch, epochs=args.epochs)

        vl = evaluate(model, test_loader, device, use_expr=args.use_expr)

        # Console log
        dt = time.time() - t0
        msg = (f"[{epoch:03d}/{args.epochs}] "
               f"loss={tr['loss']:.4f} va_loss={tr['loss_va']:.4f} "
               f"| ccc(V)={vl['ccc_v']:.3f} ccc(A)={vl['ccc_a']:.3f} mean={vl['ccc_mean']:.3f} "
               f"rmse(V)={vl['rmse_v']:.3f} rmse(A)={vl['rmse_a']:.3f} "
               f"mae(V)={vl['mae_v']:.3f} mae(A)={vl['mae_a']:.3f} time={dt:.1f}s")
        if args.use_expr and "expr_acc" in vl:
            msg += f" | expr_acc={vl['expr_acc']:.3f}"
        print(msg)

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            row = [epoch, tr["loss"], tr["loss_va"], tr["loss_expr"],
                   vl["ccc_v"], vl["ccc_a"], vl["ccc_mean"],
                   vl["rmse_v"], vl["rmse_a"], vl["mae_v"], vl["mae_a"]]
            if args.use_expr and "expr_acc" in vl: row.append(vl["expr_acc"])
            w.writerow(row)

        # Track and save the best checkpoint by mean CCC over V/A
        if vl["ccc_mean"] > best:
            best = vl["ccc_mean"]
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ saved best to {best_path} (ccc_mean={best:.3f})")

    # Final export of last-epoch weights (best already saved separately)
    export = outdir / f"emonet_{args.nclasses}_finetuned.pth"
    torch.save(model.state_dict(), export)
    print(f"Done. Best ckpt: {best_path}\nExport: {export}")


if __name__ == "__main__":
    main()
