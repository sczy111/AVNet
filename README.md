# AVNet
# EmoNet Fine-Tuning — `train_emonet.py`

A PyTorch script for **transfer learning on EmoNet**, supporting **Valence–Arousal (VA) regression** and optional **facial expression classification** (5 or 8 classes). Includes staged fine-tuning, mixed precision, metric logging, and checkpointing.

---

## 1️⃣ Environment

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python pandas numpy tqdm
```


Ensure `emonet.models.EmoNet` is accessible (e.g., via `PYTHONPATH` or editable install).

---

## 2️⃣ Data Format

Two CSV files: **train** and **test**.

| column    | type  | description                           |
| --------- | ----- | ------------------------------------- |
| `pth`     | str   | image path (relative to `--*_root`)   |
| `valence` | float | target in [-1,1]                      |
| `arousal` | float | target in [-1,1]                      |
| `label`   | str   | expression (required if `--use_expr`) |

**Expression sets:**

* 8-class: `neutral, happy, sad, surprise, fear, disgust, anger, contempt`
* 5-class: `neutral, happy, sad, surprise, fear`

```csv
pth,label,valence,arousal
img1.jpg,happy,0.6,0.4
img2.jpg,sad,-0.4,0.1
```

---

## 3️⃣ Pretrained Weights

Automatically loads if found:

```
pretrained/emonet_5.pth
pretrained/emonet_8.pth
```

---

## 4️⃣ Core Idea

* **Backbone**: frozen EmoNet feature extractor.
* **Heads**:

  * VA: CCC loss for valence & arousal.
  * Expr (optional): CrossEntropy loss.
* **Total Loss** = VA loss + λ × Expr loss.
* Optional **unfreeze** backbone after N epochs to fine-tune fully.

---

## 5️⃣ Usage

### VA only

```bash
python train_emonet.py \
  --train_csv data/train.csv --test_csv data/val.csv \
  --train_root data/images --test_root data/images \
  --nclasses 8 --epochs 40 --batch 32
```

### VA + Expr

```bash
python train_emonet.py \
  --train_csv data/train.csv --test_csv data/val.csv \
  --train_root data/images --test_root data/images \
  --nclasses 8 --use_expr --lambda_expr 1.0
```

### Mixed Precision + Unfreeze

```bash
python train_emonet.py ... --amp --unfreeze_backbone_after 10
```

Key args:

* `--train_csv`, `--test_csv`, `--train_root`, `--test_root` — dataset paths
* `--nclasses {5,8}` — expression head size
* `--use_expr` — enable expression training
* `--epochs`, `--batch`, `--lr`, `--amp` — training settings
* `--unfreeze_backbone_after` — staged fine-tuning

---

## 6️⃣ Outputs

Saved under `--outdir` (default `runs/emonet_train/`):

* `ckpt_best.pth` — best by `ccc_mean`
* `metrics.csv` — per-epoch metrics (losses, CCC, RMSE, MAE, expr acc)
* `emonet_<N>_finetuned.pth` — final model

---

## 7️⃣ Tips

* Start with **frozen backbone**, unfreeze later if dataset is large or different.
* Monitor **`ccc_mean`** (main VA metric).
* Use `--amp` to reduce GPU memory.
* Label set must match canonical 5/8 classes.

---

## 8️⃣ Example Full Command

```bash
python train_emonet.py \
  --train_csv data/train.csv --test_csv data/val.csv \
  --train_root data/img --test_root data/img \
  --nclasses 8 --use_expr --lambda_expr 0.5 \
  --epochs 40 --batch 32 --lr 3e-4 \
  --amp --unfreeze_backbone_after 8 \
  --outdir runs/demo
```

---

## 9️⃣ Troubleshooting

* ❌ *Unknown labels* → match canonical label set.
* ❌ *OOM errors* → lower `--batch` or `--size`, enable `--amp`.
* ❌ *Divergence after unfreeze* → lower LR or delay unfreeze.
* ✅ Use `metrics.csv` to track progress over epochs.

---

## 🔟 Licensing

Follow EmoNet’s and dataset licenses. Ensure legal/ethical compliance for facial data use (e.g., GDPR).

```
```
