# Cell 0 - GPU check and Google Drive mount.

import os, gc, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
gc.collect(); torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f'GPU  : {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info()
    print(f'VRAM : {free/1024**3:.1f} / {total/1024**3:.1f} GiB free')
else:
    raise RuntimeError('No GPU — switch to a GPU runtime in Colab.')

from google.colab import drive
drive.mount('/content/drive')

# Cell 1 - Install required Python packages.

import subprocess, sys
for pkg in ['timm==0.9.16', 'albumentations>=1.3.0', 'einops', 'tqdm', 'scikit-learn']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
print('packages ready')

# Cell 2 - Import dependencies, set seeds, and configure evaluation.

import zipfile, random, json, hashlib, csv, shutil, warnings
import numpy as np
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, balanced_accuracy_score,
                              confusion_matrix)
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda')

CFG = {
    'output_dir'    : '/content/drive/MyDrive/SSL_FYDP/Outputs',
    'labeled_root'  : '/content/labeled_data',
    'kvasir_v2_dir' : '/content/drive/MyDrive/SSL_FYDP/kvasir-dataset-v2',
    'proto_ckpt'    : '/content/drive/MyDrive/SSL_FYDP/Outputs/protonet_best.pth',
    'split_file'    : '/content/drive/MyDrive/SSL_FYDP/Outputs/split_indices.json',
    'backbone'      : 'swin_tiny_patch4_window7_224',
    'embed_dim'     : 768,
    'img_size'      : 224,
    'dropout_rate'  : 0.1,
    'temperature'   : 0.5,
    'mc_passes'     : 20,
    'max_per_class' : 300,
    'k_shot_train'  : 5,
    'n_query'       : 15,
}
SIZE = CFG['img_size']

EVAL_DIR    = f"{CFG['output_dir']}/indomain_test"
os.makedirs(EVAL_DIR, exist_ok=True)
METRICS_CSV = f'{EVAL_DIR}/indomain_episode_metrics.csv'
PRED_LOG    = f'{EVAL_DIR}/indomain_predictions.jsonl'

K_SHOTS        = [1, 5]
N_QUERY        = 15
FULL_SEEDS     = [42, 43, 44, 45, 46]
FULL_EPISODES  = 600
SWEEP_SEEDS    = [42, 43, 44]
SWEEP_EPISODES = 150

print(f'Metrics log     : {METRICS_CSV}')
print(f'Predictions log : {PRED_LOG}')

# Cell 3 - Rebuild the labeled dataset and load the held-out test split.

def find_kv2_root(base):
    base = Path(base)
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if len(subdirs) >= 6:
        return base
    for sub in subdirs:
        if len([d for d in sub.iterdir() if d.is_dir()]) >= 6:
            return sub
    raise RuntimeError(f'Could not locate Kvasir v2 class folders under {base}')

def resolve_kv2_root(cfg):
    local_dir = Path('/content/kvasir-dataset-v2')
    if local_dir.exists() and any(local_dir.rglob('*.jpg')):
        return find_kv2_root(local_dir)

    configured = Path(cfg['kvasir_v2_dir'])
    zip_path = None
    if configured.exists() and configured.suffix == '.zip':
        zip_path = configured
    elif configured.with_suffix('.zip').exists():
        zip_path = configured.with_suffix('.zip')
    elif configured.exists() and configured.is_dir():
        return find_kv2_root(configured)
    else:
        parent = configured.parent if configured.parent.exists() else Path(cfg['output_dir']).parent
        candidates = sorted(parent.glob('*kvasir*.zip')) + sorted(parent.glob('*Kvasir*.zip'))
        if candidates:
            zip_path = candidates[0]

    assert zip_path is not None, (
        f"Could not find an extracted Kvasir v2 folder or a matching zip near "
        f"{cfg['kvasir_v2_dir']}. Set CFG['kvasir_v2_dir'] to the exact zip path.")

    print(f'Extracting {zip_path} -> {local_dir}')
    local_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(local_dir)
    return find_kv2_root(local_dir)

kv2_root = resolve_kv2_root(CFG)

rng_dataset = random.Random(SEED)
if Path(CFG['labeled_root']).exists():
    shutil.rmtree(CFG['labeled_root'])

MAX         = CFG['max_per_class']
MIN_NEEDED  = CFG['k_shot_train'] + CFG['n_query'] + 10
all_classes = []

for cls_dir in sorted(kv2_root.iterdir()):
    if not cls_dir.is_dir():
        continue
    imgs = sorted(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.jpeg')) +
                  list(cls_dir.glob('*.png')))
    if len(imgs) < MIN_NEEDED:
        continue
    rng_dataset.shuffle(imgs)
    dst = Path(CFG['labeled_root']) / cls_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    for src in imgs[:MAX]:
        shutil.copy(src, dst / src.name)
    all_classes.append(cls_dir.name)

print(f'{len(all_classes)} classes rebuilt')

class LabeledPolyp(torch.utils.data.Dataset):
    def __init__(self, root_dirs):
        self.samples = []
        self.classes = []
        for root in root_dirs:
            for d in sorted(Path(root).iterdir()):
                if d.is_dir() and d.name not in self.classes:
                    self.classes.append(d.name)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for root in root_dirs:
            for d in sorted(Path(root).iterdir()):
                if d.is_dir():
                    lbl = self.class_to_idx[d.name]
                    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                        for p in sorted(d.glob(ext)):
                            self.samples.append((str(p), lbl))
    def __len__(self):
        return len(self.samples)

full_ds = LabeledPolyp([CFG['labeled_root']])
print(f'Rebuilt full dataset: {len(full_ds)} samples, classes={full_ds.classes}')

with open(CFG['split_file']) as f:
    split_data = json.load(f)
test_idx = sorted(split_data['test'])

test_hash = hashlib.sha256(json.dumps(test_idx).encode()).hexdigest()
print(f'test_idx count  : {len(test_idx)}')
print(f'test_idx sha256 : {test_hash}')
print('Compare against Part 3a Cell 3 — must match exactly.')

assert len(full_ds) == split_data['n_total'], (
    f'Rebuilt dataset size {len(full_ds)} != split n_total {split_data["n_total"]}. '
    f'Dataset rebuild does not match the split saved in Part 2 — stop and investigate '
    f'before trusting any downstream result.')

test_samples = [full_ds.samples[i] for i in test_idx]
test_by_class = {}
for path, lbl in test_samples:
    test_by_class.setdefault(lbl, []).append(path)

idx_to_class = {v: k for k, v in full_ds.class_to_idx.items()}
print('Test images per class:')
for lbl, paths in sorted(test_by_class.items()):
    print(f'  {idx_to_class[lbl]}: {len(paths)}')

# Cell 4 - Define and load the trained SwinProtoNet checkpoint.

class SwinProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            CFG['backbone'], pretrained=False,
            num_classes=0, img_size=CFG['img_size'])
        actual_dim = self.encoder.num_features
        D = CFG['embed_dim']
        self.drop = nn.Dropout(p=CFG['dropout_rate'])
        self.proj = nn.Sequential(
            nn.Linear(actual_dim, D * 2), nn.LayerNorm(D * 2), nn.GELU(),
            nn.Dropout(p=CFG['dropout_rate']),
            nn.Linear(D * 2, D), nn.LayerNorm(D),
        )

    def encode(self, x):
        f = self.encoder.forward_features(x)
        if f.dim() == 4: f = f.mean(dim=[1, 2])
        elif f.dim() == 3: f = f.mean(dim=1)
        return F.normalize(self.proj(self.drop(f)), dim=-1)

    def compute_prototypes(self, emb, labels, n_way):
        return torch.stack([emb[labels == c].mean(0) for c in range(n_way)])

    def forward(self, sup, sup_lbl, qry, n_way):
        se = self.encode(sup); qe = self.encode(qry)
        p = self.compute_prototypes(se, sup_lbl, n_way)
        return -torch.cdist(qe.unsqueeze(0), p.unsqueeze(0)).squeeze(0)

    def predict_with_uncertainty(self, sup, sup_lbl, qry, n_way, n_passes=None):
        n_passes = n_passes or CFG['mc_passes']
        self.train()
        probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                p = F.softmax(self(sup, sup_lbl, qry, n_way) / CFG['temperature'], dim=-1)
                probs.append(p.unsqueeze(0))
        self.eval()
        probs  = torch.cat(probs, 0)
        mean_p = probs.mean(0)
        unc    = probs.var(0).mean(-1)
        return mean_p.argmax(-1), mean_p, unc

protonet = SwinProtoNet().to(DEVICE)
state = torch.load(CFG['proto_ckpt'], map_location=DEVICE, weights_only=False)
protonet.load_state_dict(state)
protonet.eval()
print('SwinProtoNet loaded from', CFG['proto_ckpt'])

# Cell 5 - Filter eligible test classes and define episode sampling.

MAX_SHOT   = max(K_SHOTS)
MIN_NEEDED = MAX_SHOT + N_QUERY

qualifying = {lbl: paths for lbl, paths in test_by_class.items()
              if len(paths) >= MIN_NEEDED}
dropped    = {lbl: len(paths) for lbl, paths in test_by_class.items()
              if len(paths) < MIN_NEEDED}

if dropped:
    print(f'Classes dropped from in-domain episodes (below {MIN_NEEDED} test images):')
    for lbl, n in dropped.items():
        print(f'  {idx_to_class[lbl]}: {n}')

N_WAY_MAX = len(qualifying)
assert N_WAY_MAX >= 2, 'Not enough in-domain test classes with sufficient images'
INDOMAIN_LABELS         = sorted(qualifying.keys())
INDOMAIN_LABEL_TO_LOCAL = {lbl: i for i, lbl in enumerate(INDOMAIN_LABELS)}
INDOMAIN_CLASS_NAMES    = [idx_to_class[lbl] for lbl in INDOMAIN_LABELS]
INDOMAIN_SAMPLES = {
    INDOMAIN_LABEL_TO_LOCAL[lbl]: sorted(qualifying[lbl]) for lbl in INDOMAIN_LABELS
}
print(f'N_WAY_MAX = {N_WAY_MAX} classes: {INDOMAIN_CLASS_NAMES}')

N_WAYS = list(range(2, N_WAY_MAX + 1))
print(f'N-way sweep points: {N_WAYS}')

est_full  = len(K_SHOTS) * len(FULL_SEEDS) * FULL_EPISODES
est_sweep = max(0, len(N_WAYS) - 1) * len(K_SHOTS) * len(SWEEP_SEEDS) * SWEEP_EPISODES
print(f'Full-budget episodes (N={N_WAY_MAX} only): {est_full}')
print(f'Reduced-budget episodes (N<{N_WAY_MAX}): {est_sweep}')
print(f'Total target episodes: {est_full + est_sweep}')

val_tfm = A.Compose([
    A.Resize(SIZE, SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def load_image(path):
    img = np.array(Image.open(path).convert('RGB'))
    return val_tfm(image=img)['image']

def pick_classes(n_way, seed, k_shot, episode_idx):
    if n_way == N_WAY_MAX:
        return list(range(N_WAY_MAX))
    rng = random.Random(seed * 10_000_000 + n_way * 1_000_000 +
                         k_shot * 10_000 + episode_idx)
    return rng.sample(range(N_WAY_MAX), n_way)

def sample_episode(n_way, k_shot, seed, episode_idx):
    rng = random.Random(seed * 1_000_000 + n_way * 100_000 +
                         k_shot * 10_000 + episode_idx)
    class_ids = pick_classes(n_way, seed, k_shot, episode_idx)
    sup_imgs, sup_lbls, qry_imgs, qry_lbls = [], [], [], []
    for local, cls in enumerate(class_ids):
        pool   = INDOMAIN_SAMPLES[cls]
        chosen = rng.sample(pool, k_shot + N_QUERY)
        for path in chosen[:k_shot]:
            sup_imgs.append(load_image(path)); sup_lbls.append(local)
        for path in chosen[k_shot:]:
            qry_imgs.append(load_image(path)); qry_lbls.append(local)
    return (torch.stack(sup_imgs), torch.tensor(sup_lbls),
            torch.stack(qry_imgs), torch.tensor(qry_lbls))

# Cell 6 - Run the resumable N-way and K-shot evaluation and save metrics.

def completed_episodes():
    done = set()
    if Path(METRICS_CSV).exists():
        with open(METRICS_CSV) as f:
            for row in csv.DictReader(f):
                done.add((int(row['n_way']), int(row['k_shot']),
                          int(row['seed']), int(row['episode_idx'])))
    return done

def append_metrics_row(row):
    write_header = not Path(METRICS_CSV).exists()
    with open(METRICS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush(); os.fsync(f.fileno())

def append_prediction_log(record):
    with open(PRED_LOG, 'a') as f:
        f.write(json.dumps(record) + '\n')
        f.flush(); os.fsync(f.fileno())

done = completed_episodes()
print(f'Episodes already completed: {len(done)}')

from tqdm.auto import tqdm

for n_way in N_WAYS:
    seeds, n_episodes = (FULL_SEEDS, FULL_EPISODES) if n_way == N_WAY_MAX \
                        else (SWEEP_SEEDS, SWEEP_EPISODES)
    for k_shot in K_SHOTS:
        for seed in seeds:
            for episode_idx in tqdm(range(n_episodes),
                                     desc=f'N={n_way} k={k_shot} seed={seed}'):
                if (n_way, k_shot, seed, episode_idx) in done:
                    continue

                sup, sup_lbl, qry, qry_lbl = sample_episode(n_way, k_shot, seed, episode_idx)
                sup, sup_lbl = sup.to(DEVICE), sup_lbl.to(DEVICE)
                qry, qry_lbl = qry.to(DEVICE), qry_lbl.to(DEVICE)

                with torch.no_grad():
                    logits = protonet(sup, sup_lbl, qry, n_way)
                    probs  = F.softmax(logits / CFG['temperature'], dim=-1)

                y_true = qry_lbl.cpu().numpy()
                y_pred = probs.argmax(-1).cpu().numpy()

                acc      = float((y_pred == y_true).mean())
                macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                bal_acc  = balanced_accuracy_score(y_true, y_pred)

                append_metrics_row({
                    'n_way': n_way, 'k_shot': k_shot, 'seed': seed,
                    'episode_idx': episode_idx,
                    'accuracy': acc, 'macro_f1': macro_f1, 'balanced_acc': bal_acc,
                })
                if n_way == N_WAY_MAX:
                    append_prediction_log({
                        'n_way': n_way, 'k_shot': k_shot, 'seed': seed,
                        'episode_idx': episode_idx,
                        'y_true': y_true.tolist(), 'y_pred': y_pred.tolist(),
                        'y_prob': probs.cpu().numpy().tolist(),
                    })

print('Evaluation sweep complete')

# Cell 7 - Aggregate episode-level statistics and save the summary.

import pandas as pd

df = pd.read_csv(METRICS_CSV)
summary_rows = []
for n_way in N_WAYS:
    for k_shot in K_SHOTS:
        sub = df[(df['n_way'] == n_way) & (df['k_shot'] == k_shot)]
        if len(sub) == 0:
            continue
        acc  = sub['accuracy'].values
        ci95 = 1.96 * acc.std(ddof=1) / np.sqrt(len(acc)) if len(acc) > 1 else float('nan')
        summary_rows.append({
            'n_way'             : n_way,
            'k_shot'            : k_shot,
            'n_episodes'        : len(acc),
            'mean_accuracy'     : acc.mean(),
            'ci95'              : ci95,
            'mean_macro_f1'     : sub['macro_f1'].mean(),
            'mean_balanced_acc' : sub['balanced_acc'].mean(),
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'{EVAL_DIR}/indomain_summary.csv', index=False)
print(summary_df.to_string(index=False))

# Cell 8 - Compute pooled confusion matrices, per-class metrics, AUC, and combined metric.

pooled = {k: {'y_true': [], 'y_pred': [], 'y_prob': []} for k in K_SHOTS}
with open(PRED_LOG) as f:
    for line in f:
        r = json.loads(line)
        if r['n_way'] != N_WAY_MAX:
            continue
        k = r['k_shot']
        pooled[k]['y_true'].extend(r['y_true'])
        pooled[k]['y_pred'].extend(r['y_pred'])
        pooled[k]['y_prob'].extend(r['y_prob'])

per_class_records  = []
confusion_matrices = {}
auc_by_kshot        = {}
combined_by_kshot    = {}

for k_shot in K_SHOTS:
    y_true = np.array(pooled[k_shot]['y_true'])
    y_pred = np.array(pooled[k_shot]['y_pred'])
    y_prob = np.array(pooled[k_shot]['y_prob'])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_WAY_MAX)))
    confusion_matrices[k_shot] = cm

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr',
                             labels=list(range(N_WAY_MAX)))
    except ValueError:
        auc = float('nan')
    auc_by_kshot[k_shot] = auc

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    combined_by_kshot[k_shot] = (auc + bal_acc) / 2 if not np.isnan(auc) else float('nan')

    for c in range(N_WAY_MAX):
        yt = (y_true == c).astype(int)
        yp = (y_pred == c).astype(int)
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        per_class_records.append({
            'k_shot'              : k_shot,
            'class'               : INDOMAIN_CLASS_NAMES[c],
            'precision'           : precision_score(yt, yp, zero_division=0),
            'recall_sensitivity'  : recall_score(yt, yp, zero_division=0),
            'specificity'         : specificity,
            'f1'                  : f1_score(yt, yp, zero_division=0),
        })

per_class_df = pd.DataFrame(per_class_records)
per_class_df.to_csv(f'{EVAL_DIR}/indomain_per_class_metrics.csv', index=False)
print(per_class_df.to_string(index=False))
for k_shot in K_SHOTS:
    print(f'k={k_shot}  AUC={auc_by_kshot[k_shot]:.4f}  '
          f'combined_metric(UNVERIFIED)={combined_by_kshot[k_shot]:.4f}')

# Cell 9 - Generate and save publication-style evaluation figures.

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'        : ['Times New Roman', 'DejaVu Serif'],
    'font.size'         : 9,
    'axes.linewidth'    : 0.8,
    'axes.edgecolor'    : '#3A3A3A',
    'axes.labelcolor'   : '#222222',
    'text.color'        : '#222222',
    'xtick.color'       : '#3A3A3A',
    'ytick.color'       : '#3A3A3A',
    'axes.grid'         : True,
    'grid.alpha'        : 0.25,
    'grid.linewidth'    : 0.5,
    'legend.frameon'    : False,
    'savefig.dpi'       : 300,
    'figure.dpi'        : 150,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
})

PALETTE    = ['#4C72B0', '#DD8452', '#55A868', '#C44E52',
              '#8172B2', '#937860', '#64B5CD', '#CCB974']
FIG_SINGLE = (3.45, 2.6)
FIG_WIDE   = (7.16, 3.0)

fig, ax = plt.subplots(figsize=FIG_SINGLE)
for i, k_shot in enumerate(K_SHOTS):
    sub = summary_df[summary_df['k_shot'] == k_shot].sort_values('n_way')
    ax.errorbar(sub['n_way'], sub['mean_accuracy'] * 100, yerr=sub['ci95'] * 100,
                marker='o', ms=4, lw=1.4, color=PALETTE[i], capsize=2.5,
                label=f'k={k_shot}')
ax.set_xlabel('N-way')
ax.set_ylabel('Accuracy (%)')
ax.set_title('In-Domain: Accuracy vs N-way', fontsize=9.5)
ax.legend()
fig.tight_layout()
fig.savefig(f'{EVAL_DIR}/fig_indomain_accuracy_vs_nway.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=FIG_SINGLE)
sub = summary_df[summary_df['n_way'] == N_WAY_MAX].sort_values('k_shot')
ax.errorbar(sub['k_shot'], sub['mean_accuracy'] * 100, yerr=sub['ci95'] * 100,
            marker='o', ms=5, lw=1.6, color=PALETTE[0], capsize=3,
            capthick=1.2, elinewidth=1.2)
ax.set_xticks(sub['k_shot'])
ax.set_xlabel('K-shot')
ax.set_ylabel('Accuracy (%)')
ax.set_title(f'In-Domain: Accuracy vs K-shot (N={N_WAY_MAX})', fontsize=9.5)
fig.tight_layout()
fig.savefig(f'{EVAL_DIR}/fig_indomain_accuracy_vs_kshot.png', bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, len(K_SHOTS), figsize=FIG_WIDE)
fig.subplots_adjust(wspace=0.7)
for i, k_shot in enumerate(K_SHOTS):
    cm      = confusion_matrices[k_shot]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = axes[i].imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    axes[i].set_xticks(range(N_WAY_MAX))
    axes[i].set_yticks(range(N_WAY_MAX))
    axes[i].set_xticklabels(INDOMAIN_CLASS_NAMES, rotation=45, ha='right', fontsize=7)
    axes[i].set_yticklabels(INDOMAIN_CLASS_NAMES, fontsize=7)
    axes[i].set_title(f'k={k_shot}', fontsize=9)
    for r in range(N_WAY_MAX):
        for c in range(N_WAY_MAX):
            color = 'white' if cm_norm[r, c] > 0.5 else '#222222'
            axes[i].text(c, r, f'{cm_norm[r, c]:.2f}', ha='center', va='center',
                         fontsize=6, color=color)
fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label='Row-normalized rate')
fig.suptitle('In-Domain Test Confusion Matrices', fontsize=9.5, y=1.03)
fig.savefig(f'{EVAL_DIR}/fig_indomain_confusion_matrix.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=FIG_WIDE)
metrics_to_plot = ['precision', 'recall_sensitivity', 'specificity', 'f1']
k_shot_plot     = K_SHOTS[-1]
sub             = per_class_df[per_class_df['k_shot'] == k_shot_plot]
xpos            = np.arange(len(INDOMAIN_CLASS_NAMES))
width           = 0.19
for i, m in enumerate(metrics_to_plot):
    ax.bar(xpos + (i - 1.5) * width, sub[m].values, width, label=m, color=PALETTE[i])
ax.set_xticks(xpos)
ax.set_xticklabels(INDOMAIN_CLASS_NAMES, rotation=30, ha='right',fontsize=6 )
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.set_title(f'Per-Class Metrics, k={k_shot_plot}, N={N_WAY_MAX}', fontsize=9.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=4, fontsize=7.5)
fig.tight_layout()
fig.savefig(f'{EVAL_DIR}/fig_indomain_per_class_metrics.png', bbox_inches='tight')
plt.show()

print('Figures saved to', EVAL_DIR)
