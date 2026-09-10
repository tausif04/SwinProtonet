# -*- coding: utf-8 -*-
"""
### Part 2 — SwinProtoNet Few-Shot Training (Kvasir v2)
"""

"""## Cell 0 — GPU Setup"""

import os, gc, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
gc.collect(); torch.cuda.empty_cache()
print(f'GPU : {torch.cuda.get_device_name(0)}')
free,total = torch.cuda.mem_get_info()
print(f'VRAM: {free/1024**3:.1f}/{total/1024**3:.1f} GiB')

"""## Cell 1 — Install Packages"""

import subprocess, sys
for pkg in ['timm==0.9.16','albumentations>=1.3.0','einops','tqdm','scikit-learn']:
    subprocess.check_call([sys.executable,'-m','pip','install',pkg,'-q'])
print(' packages ready')

"""## Cell 2 — Imports & Seeds"""

import os, random, shutil, warnings, json
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from scipy import stats as sp_stats
from sklearn.metrics import f1_score, classification_report
warnings.filterwarnings('ignore')

SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
set_seed()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device {DEVICE} | timm {timm.__version__} | A {A.__version__}')

# Add after set_seed()
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  

"""## Cell 3 — Configuration"""

CFG = {
    # Paths
    'drive_root'    : '/content/drive/MyDrive/SSL_FYDP',
    'output_dir'    : '/content/drive/MyDrive/SSL_FYDP/Outputs',
    'labeled_root'  : '/content/labeled_data',
    'kvasir_v2_dir' : '/content/drive/MyDrive/SSL_FYDP/kvasir-dataset-v2',
    # SSL checkpoint from Part 1
    'ssl_ckpt'      : '/content/drive/MyDrive/SSL_FYDP/Outputs/swin_ssl_pretrained.pth',
    # ProtoNet checkpoint from earlier training for resumability
    'proto_ckpt'    : '/content/drive/MyDrive/SSL_FYDP/Outputs/protonet_best.pth',
    # Split file generated in Cell 8
    'split_file'    : '/content/drive/MyDrive/SSL_FYDP/Outputs/split_indices.json',
    # Model
    'backbone'      : 'swin_tiny_patch4_window7_224',
    'embed_dim'     : 768,
    'img_size'      : 224,
    'dropout_rate'  : 0.1,
    # Few-shot
    'n_way'         : 8,
    'k_shot_train'  : 5,
    'k_shot_test'   : [1,5],
    'n_query'       : 15,
    'n_ep_train'    : 60,
    'n_ep_val'      : 30,
    'n_ep_test'     : 600,
    # Training
    'encoder_lr'    : 1e-5,
    'projector_lr'  : 1e-4,
    'finetune_lr'   : 1e-6,
    'weight_decay'  : 1e-4,
    'temperature'   : 0.5,
    'label_smooth'  : 0.1,
    'max_per_class' : 300,
    # MC-Dropout
    'mc_passes'     : 20,
}
SIZE = CFG['img_size']
for d in [CFG['output_dir'], CFG['labeled_root']]:
    os.makedirs(d, exist_ok=True)
ssl_ok = Path(CFG['ssl_ckpt']).exists()
print(f'SSL ckpt : {"Found" if ssl_ok else "Run Part 1 first"}')
print(f'Config   : {CFG["n_way"]}-way {CFG["k_shot_train"]}-shot | '
      f'{CFG["n_ep_train"]} train eps | {CFG["n_ep_val"]} val eps')

"""Drive Mount

"""

from google.colab import drive
import os
import shutil

try:
  drive.flush_and_unmount()
  print('Drive unmounted successfully.')
except ValueError:
  print('Drive was not mounted.')
except Exception as e:
  print(f'Error unmounting Drive: {e}')

# Ensure the mountpoint is empty by recursively removing it if it exists
if os.path.exists('/content/drive') and os.path.isdir('/content/drive'):
    try:
        shutil.rmtree('/content/drive')
        print('Removed existing /content/drive directory.')
    except OSError as e:
        print(f'Error removing /content/drive directory: {e}')

drive.mount('/content/drive', force_remount=True)

"""##Cell 4 — Load Kvasir v2 from Driv"""

import zipfile

zip_path = Path('/content/drive/MyDrive/SSL_FYDP/kvasir-dataset-v2.zip')
extract_to = Path('/content/drive/MyDrive/SSL_FYDP/kvasir-dataset-v2')

if zip_path.exists() and not extract_to.exists():
    print(f'Extracting {zip_path.name} → {extract_to} ...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print('Extraction complete.')
elif extract_to.exists():
    print(f'Already extracted: {extract_to}')
else:
    raise FileNotFoundError(f'No zip found at {zip_path} — check Drive path/name.')

import os
from pathlib import Path

# ── Known Kvasir v2 class names (normalized: lowercase, no separators) ────
KVASIR_V2_CLASSES = {
    'dyedliftedpolyps', 'dyedresectionmargins', 'esophagitis',
    'normalcecum', 'normalpylorus', 'normalzline',
    'polyps', 'ulcerativecolitis',
}

# Folders that are known NOT to be the dataset, even if they have >=6 subdirs
EXCLUDE_DIR_NAMES = {'outputs', 'checkpoints', 'logs', 'labeled_data'}

def _norm(name: str) -> str:
    return name.lower().replace('-', '').replace('_', '').replace(' ', '')

def _kvasir_class_hits(candidate: Path) -> int:
    """Count how many of candidate's subdirs match a known Kvasir v2 class name."""
    if not candidate.is_dir():
        return 0
    subdirs = [d for d in candidate.iterdir()
               if d.is_dir() and not d.name.startswith('.')]
    hits = sum(1 for d in subdirs if _norm(d.name) in KVASIR_V2_CLASSES)
    return hits

# ── Step 2 (FIXED): find the Kvasir v2 dataset directory ──────────────────
kv2_root = None
search_roots = [
    Path(CFG.get('kvasir_v2_dir', '')),
    Path('/content/drive/MyDrive/SSL_FYDP/kvasir-dataset-v2'),
    Path('/content/drive/MyDrive/SSL_FYDP/Kvasir-dataset-v2'),
    Path('/content/drive/MyDrive/SSL_FYDP/kvasir_dataset_v2'),
    Path('/content/drive/MyDrive/SSL_FYDP/kvasir-v2'),
    Path('/content/drive/MyDrive/kvasir-dataset-v2'),
    Path('/content/drive/MyDrive/kvasir-v2'),
]

ssl_root_path = Path('/content/drive/MyDrive/SSL_FYDP')
if ssl_root_path.exists() and ssl_root_path.is_dir():
    for sub in ssl_root_path.iterdir():
        if sub.is_dir() and sub.name.lower() not in EXCLUDE_DIR_NAMES:
            search_roots.append(sub)

# Track every candidate checked, for a useful error message if none qualify
checked_log = []

for cand_path in dict.fromkeys(search_roots):  # de-dup, preserve order
    if not cand_path or not cand_path.exists() or not cand_path.is_dir():
        continue
    if cand_path.name.lower() in EXCLUDE_DIR_NAMES:
        continue

    hits = _kvasir_class_hits(cand_path)
    checked_log.append((str(cand_path), hits))
    if hits >= 6:
        kv2_root = cand_path
        break

    # Check one level deeper
    subdirs = [d for d in cand_path.iterdir()
               if d.is_dir() and not d.name.startswith('.')
               and d.name.lower() not in EXCLUDE_DIR_NAMES]
    for sub_sub in subdirs:
        hits2 = _kvasir_class_hits(sub_sub)
        checked_log.append((str(sub_sub), hits2))
        if hits2 >= 6:
            kv2_root = sub_sub
            break
    if kv2_root:
        break

if not kv2_root:
    log_str = '\n'.join(f'  {p}  -> {h}/8 class-name matches' for p, h in checked_log)
    raise AssertionError(
        f'Kvasir v2 dataset directory not found — no candidate had >=6 '
        f'subfolders matching known Kvasir v2 class names.\n\n'
        f'Candidates checked:\n{log_str}\n\n'
        f'This usually means kvasir-dataset-v2 was never extracted into '
        f'Drive, or lives under a different path/name than CFG["kvasir_v2_dir"] '
        f'points to. Upload/extract it to:\n'
        f'  {CFG["drive_root"]}/kvasir-dataset-v2/<8 class folders>\n'
        f'then rerun this cell.'
    )

CFG['kvasir_v2_dir'] = str(kv2_root)
KV2_PATH = Path(CFG['kvasir_v2_dir'])

print(f'Kvasir v2 root : {kv2_root}')
print(f'\n  Class breakdown:')
total, n_classes = 0, 0
for d in sorted(kv2_root.iterdir()):
    if not d.is_dir() or d.name.startswith('.'):
        continue
    n = len(list(d.glob('*.jpg')) + list(d.glob('*.jpeg')) + list(d.glob('*.png')))
    total += n; n_classes += 1
    status = 'okay' if n >= CFG['max_per_class'] else f'  only {n}'
    print(f'  {status} {d.name:<35}: {n} images')

print(f'\n  Classes : {n_classes}')
print(f'  Total   : {total} images')
assert n_classes == 8, f'Expected 8 classes, found {n_classes}'
assert n_classes >= CFG["n_way"], \
    f'Need {CFG["n_way"]} classes for {CFG["n_way"]}-way episodes'

"""## Cell 5 — Build Labeled Dataset (300/class, reproducible)"""

# Rebuild with same seed each time → deterministic file list
rng_dataset = random.Random(SEED)
if Path(CFG['labeled_root']).exists():
    shutil.rmtree(CFG['labeled_root'])

MAX = CFG['max_per_class']
MIN_NEEDED = CFG['k_shot_train'] + CFG['n_query'] + 10
all_classes = []

for cls_dir in sorted(kv2_root.iterdir()):
    if not cls_dir.is_dir(): continue
    imgs = sorted(list(cls_dir.glob('*.jpg')) +
                  list(cls_dir.glob('*.jpeg')) +
                  list(cls_dir.glob('*.png')))
    if len(imgs) < MIN_NEEDED:
        print(f'  Skip {cls_dir.name}: only {len(imgs)} images'); continue
    rng_dataset.shuffle(imgs)
    selected = imgs[:MAX]
    dst = Path(CFG['labeled_root']) / cls_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    for src in selected:
        shutil.copy(src, dst / src.name)
    all_classes.append(cls_dir.name)
    print(f'   {cls_dir.name}: {len(selected)}')

total = sum(len(list((Path(CFG['labeled_root'])/c).glob('*'))) for c in all_classes)
n_way = CFG['n_way']

print("t="+str(n_way))

assert len(all_classes) >= n_way, \
    f'Need {n_way} classes but only found {len(all_classes)}'

print(f'\n {len(all_classes)} classes | {total} images')
print(f'n_way = {n_way} — 8-way episodes during training')
print(f'Each episode randomly samples {n_way} classes from {len(all_classes)} available')

"""## Cell 6 — Domain-Specific Augmentations"""

class SpecularHighlight(A.ImageOnlyTransform):
    def __init__(self, n_spots=(1,3), radius=(5,25), p=0.5):
        super().__init__(p=p)
        self.n_spots = n_spots; self.radius = radius
    def apply(self, img, **kw):
        img = img.copy(); h,w = img.shape[:2]
        for _ in range(random.randint(*self.n_spots)):
            cx,cy = random.randint(0,w), random.randint(0,h)
            rx,ry = random.randint(*self.radius), random.randint(*self.radius)
            mask  = np.zeros((h,w),np.float32)
            cv2.ellipse(mask,(cx,cy),(rx,ry),random.randint(0,180),0,360,1.,-1)
            mask  = cv2.GaussianBlur(mask,(0,0),rx//2+1)
            img   = np.clip(img.astype(np.float32)+mask[:,:,None]*255*
                            random.uniform(0.7,1.),0,255).astype(np.uint8)
        return img
    def get_transform_init_args_names(self): return ('n_spots','radius')

class Vignette(A.ImageOnlyTransform):
    def __init__(self, strength=(0.3,0.6), p=0.4):
        super().__init__(p=p); self.strength = strength
    def apply(self, img, **kw):
        h,w = img.shape[:2]; Y,X = np.ogrid[:h,:w]
        d = np.sqrt((X-w/2)**2+(Y-h/2)**2); d = d/d.max()
        return np.clip(img.astype(np.float32)*(1-random.uniform(*self.strength)*d)
                       [:,:,None],0,255).astype(np.uint8)
    def get_transform_init_args_names(self): return ('strength',)

albu_ver = tuple(int(x) for x in A.__version__.split('.')[:2])
_crop = (A.RandomResizedCrop(size=(SIZE,SIZE),scale=(0.2,1.0))
         if albu_ver>=(1,4) else A.RandomResizedCrop(SIZE,SIZE,scale=(0.2,1.0)))

train_aug = A.Compose([
    A.Resize(SIZE,SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.05,p=0.7),
    A.GaussianBlur(blur_limit=(3,5),p=0.3),
    SpecularHighlight(p=0.3), Vignette(p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)), ToTensorV2(),
])
val_aug = A.Compose([
    A.Resize(SIZE,SIZE),
    A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)), ToTensorV2(),
])
print(f'✅ Augmentations ready (albumentations {A.__version__})')

# crop out device UI chrome — adjust the box to your frame's actual overlay position
def crop_ui(img, top=0, bottom=0, left=0, right=0):
    h, w = img.shape[:2]
    return img[top:h-bottom, left:w-right]

img_resized = crop_ui(img_resized, bottom=40)  # tune based on where the overlay sits

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

# pick one real labeled frame — swap the index/class to choose a nicer-looking example
sample_path = sorted(Path(CFG['labeled_root']).glob('*/*.jpg'))[0]
img = np.array(Image.open(sample_path).convert('RGB'))
def crop_ui(img, top=0, bottom=0, left=0, right=0):
    h, w = img.shape[:2]
    return img[top:h-bottom, left:w-right]

# Define img_resized by applying crop_ui to img
img_resized = crop_ui(img, bottom=40)  # tune based on where the overlay sits
spec_tf = SpecularHighlight(p=1.0)
vig_tf  = Vignette(p=1.0)

random.seed(3)  # pick a seed whose random highlight placement looks good; adjust if needed
img_spec = spec_tf(image=img_resized.copy())['image']
random.seed(7)
img_vig  = vig_tf(image=img_resized.copy())['image']

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), dpi=300)
for ax, im, title in zip(axes, [img_resized, img_spec, img_vig],
        ['(a) Original frame', '(b) + Specular-highlight\n(p=0.3, illustrative)', '(c) + Vignette\n(p=0.4, illustrative)']):
    ax.imshow(im); ax.set_title(title, fontsize=10.5); ax.axis('off')
plt.tight_layout()
plt.savefig(f"{CFG['output_dir']}/fig4_real_augmentation.png", dpi=300, bbox_inches='tight')
plt.show()

"""## Cell 7 — Dataset & Sampler Classes"""

from torch.utils.data import Dataset

class LabeledPolyp(Dataset):
    def __init__(self, root_dirs, transform):
        self.samples=[]; self.transform=transform; self.classes=[]
        for root in root_dirs:
            for d in sorted(Path(root).iterdir()):
                if d.is_dir() and d.name not in self.classes:
                    self.classes.append(d.name)
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        for root in root_dirs:
            for d in sorted(Path(root).iterdir()):
                if d.is_dir():
                    lbl = self.class_to_idx[d.name]
                    for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
                        for p in sorted(d.glob(ext)):
                            self.samples.append((str(p),lbl))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert('RGB'))
        return self.transform(image=img)['image'], label

class SubsetWrapper(Dataset):
    def __init__(self, base, indices):
        self.samples      = [base.samples[i] for i in indices]
        self.classes      = base.classes
        self.class_to_idx = base.class_to_idx
        self.transform    = base.transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert('RGB'))
        return self.transform(image=img)['image'], label

class EpisodeSampler:
    def __init__(self, dataset, n_way, k_shot, n_query, n_episodes):
        self.ds=dataset; self.n_way=n_way; self.k_shot=k_shot
        self.n_query=n_query; self.n_episodes=n_episodes
        self.class_indices = {}
        for i,(_,lbl) in enumerate(dataset.samples):
            self.class_indices.setdefault(lbl,[]).append(i)
        self.available = [k for k,v in self.class_indices.items()
                          if len(v) >= k_shot+n_query]
        assert len(self.available) >= n_way, \
            f'Need ≥{n_way} classes with ≥{k_shot+n_query} imgs, got {len(self.available)}'
        print(f'EpisodeSampler | {n_way}-way {k_shot}-shot | ' \
              f'{n_episodes} episodes | {n_way*(k_shot+n_query)} imgs/ep')
    def __len__(self): return self.n_episodes
    def sample_episode(self):
        classes = random.sample(self.available, self.n_way)
        si,sl,qi,ql = [],[],[],[]
        for local,cls in enumerate(classes):
            chosen = random.sample(self.class_indices[cls], self.k_shot+self.n_query)
            for i in chosen[:self.k_shot]:
                img,_=self.ds[i]; si.append(img); sl.append(local)
            for i in chosen[self.k_shot:]:
                img,_=self.ds[i]; qi.append(img); ql.append(local)
        return (torch.stack(si),torch.tensor(sl),
                torch.stack(qi),torch.tensor(ql))
    def __iter__(self):
        for _ in range(self.n_episodes): yield self.sample_episode()

print(' Dataset classes defined — disk read mode (RAM safe)')

"""## Cell 8 — Build Splits & Save Indices (leakage-free)"""

# Build once with fixed seed 
full_train_ds = LabeledPolyp([CFG['labeled_root']], train_aug)
full_val_ds   = LabeledPolyp([CFG['labeled_root']], val_aug)

n_total = len(full_train_ds)
n_test  = int(n_total * 0.15)
n_val   = int(n_total * 0.15)
n_train = n_total - n_val - n_test

g = torch.Generator().manual_seed(SEED)
train_idx, val_idx, test_idx = random_split(
    range(n_total), [n_train, n_val, n_test], generator=g)
train_idx = list(train_idx)
val_idx   = list(val_idx)
test_idx  = list(test_idx)

# CRITICAL: verify no overlap
assert not set(train_idx) & set(val_idx),  'train/val overlap!'
assert not set(train_idx) & set(test_idx), 'train/test overlap!'
assert not set(val_idx)   & set(test_idx), 'val/test overlap!'

# Save indices to Drive → Part 3 loads these to reconstruct IDENTICAL test set
split_file = f"{CFG['output_dir']}/split_indices.json"
with open(split_file,'w') as f:
    json.dump({'train':train_idx,'val':val_idx,'test':test_idx,
               'seed':SEED,'n_total':n_total}, f)
print(f'Split indices saved → {split_file}')

# Build wrappers — train uses train_aug, val/test use val_aug (no augmentation)
train_wrap = SubsetWrapper(full_train_ds, train_idx)
val_wrap   = SubsetWrapper(full_val_ds,   val_idx)

print(f'Train: {len(train_wrap)} | Val: {len(val_wrap)} | '
      f'Test (held out): {len(test_idx)}')
print(f'Classes: {train_wrap.classes}')

n_way = CFG['n_way']
train_sampler = EpisodeSampler(train_wrap, n_way, CFG['k_shot_train'],
                               CFG['n_query'], CFG['n_ep_train'])
val_sampler   = EpisodeSampler(val_wrap,   n_way, 5,
                               CFG['n_query'], CFG['n_ep_val'])
print(f' Samplers ready')


"""## Cell 9 — SwinProtoNet Model (no stage freezing)"""

import torch, torch.nn as nn, torch.nn.functional as F
import timm
from pathlib import Path
import numpy as np

class SwinProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            CFG['backbone'], pretrained=True,  
            num_classes=0, img_size=CFG['img_size'])

        # Load SSL weights  extract encoder_state key correctly
        ssl_path = CFG['ssl_ckpt']
        if ssl_path and Path(ssl_path).exists():
            ckpt = torch.load(ssl_path, map_location='cpu', weights_only=False)
            enc_state = ckpt['encoder_state']
            missing, unexpected = self.encoder.load_state_dict(
                enc_state, strict=False)
            print(f'  SSL weights loaded | '
                  f'missing:{len(missing)} unexpected:{len(unexpected)}')
            print(f"backbone: {ckpt.get('backbone','swin_tiny_patch4_window7_224')} | "
                 f"embed_dim: {ckpt.get('embed_dim', 768)}")
        else:
            print('  SSL ckpt not found — using ImageNet init only')

        # NO STAGE FREEZING — all encoder params trainable from epoch 1 
        for p in self.encoder.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f'  Encoder fully unfrozen | trainable:{trainable/1e6:.1f}M')

        actual_dim = self.encoder.num_features   # 768 for Swin-Tiny
        D = CFG['embed_dim']
        self.drop = nn.Dropout(p=CFG['dropout_rate'])
        self.proj = nn.Sequential(
            nn.Linear(actual_dim, D*2), nn.LayerNorm(D*2), nn.GELU(),
            nn.Dropout(p=CFG['dropout_rate']),
            nn.Linear(D*2, D),          nn.LayerNorm(D),
        )

    def encode(self, x):
        f = self.encoder.forward_features(x)
        if   f.dim()==4: f = f.mean(dim=[1,2])
        elif f.dim()==3: f = f.mean(dim=1)
        return F.normalize(self.proj(self.drop(f)), dim=-1)

    def compute_prototypes(self, emb, labels, n_way):
        return torch.stack([emb[labels==c].mean(0) for c in range(n_way)])

    def forward(self, sup, sup_lbl, qry, n_way):
        se = self.encode(sup); qe = self.encode(qry)
        p  = self.compute_prototypes(se, sup_lbl, n_way)
        return -torch.cdist(qe.unsqueeze(0), p.unsqueeze(0)).squeeze(0)

    def predict_with_uncertainty(self, sup, sup_lbl, qry, n_way, n_passes=None):
        n_passes = n_passes or CFG['mc_passes']
        self.train()
        all_probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                p = F.softmax(self(sup,sup_lbl,qry,n_way)/CFG['temperature'],dim=-1)
                all_probs.append(p.unsqueeze(0))
        self.eval()
        all_probs = torch.cat(all_probs, 0)
        mean_p    = all_probs.mean(0)
        unc       = all_probs.var(0).mean(-1)
        return mean_p.argmax(-1), mean_p, unc

protonet = SwinProtoNet().to(DEVICE)
total_p  = sum(p.numel() for p in protonet.parameters())/1e6
train_p  = sum(p.numel() for p in protonet.parameters() if p.requires_grad)/1e6
print(f'SwinProtoNet ready | total:{total_p:.1f}M trainable:{train_p:.1f}M')


"""## Cell 10 — Episodic Training (single-phase, 60 epochs, no freeze schedule, RESUMABLE)"""

from tqdm.auto import tqdm
import random as _random
import csv, os, time

def ep_acc(logits, labels):
    return (logits.argmax(-1)==labels.to(DEVICE)).float().mean().item()

# Resume paths separate from the "best" checkpoint 
RESUME_PATH = f"{CFG['output_dir']}/protonet_RESUME.pth"
BEST_PATH   = f"{CFG['output_dir']}/protonet_best.pth"

# Episode-level CSV log — one row PER EPISODE, written in real time 
EPISODE_LOG_PATH = f"{CFG['output_dir']}/episode_log_nofreeze60.csv"
EPISODE_LOG_FIELDS = ['timestamp', 'epoch', 'split', 'episode_idx',
                       'n_way', 'k_shot', 'n_query', 'loss', 'accuracy']

def _ensure_episode_log_header():

    if not Path(EPISODE_LOG_PATH).exists():
        with open(EPISODE_LOG_PATH, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=EPISODE_LOG_FIELDS).writeheader()

def _log_episode(f_handle, writer, epoch, split, episode_idx, k_shot, loss, acc):

    writer.writerow({
        'timestamp'  : time.time(),
        'epoch'      : epoch,
        'split'      : split,
        'episode_idx': episode_idx,
        'n_way'      : n_way,
        'k_shot'     : k_shot,
        'n_query'    : CFG['n_query'],
        'loss'       : '' if loss is None else float(loss),
        'accuracy'   : float(acc),
    })
    f_handle.flush()
    os.fsync(f_handle.fileno())

def count_logged_episodes(epoch, split):

    if not Path(EPISODE_LOG_PATH).exists():
        return 0
    n = 0
    with open(EPISODE_LOG_PATH, 'r', newline='') as f:
        for row in csv.DictReader(f):
            if int(row['epoch']) == epoch and row['split'] == split:
                n += 1
    return n

def _rng_state_dict():
    return {
        'python'      : _random.getstate(),
        'numpy'       : np.random.get_state(),
        'torch'       : torch.get_rng_state(),
        'torch_cuda'  : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def _load_rng_state(state):
    _random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])

def _atomic_torch_save(obj, final_path):

    tmp_path = f'{final_path}.tmp'
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, final_path)  # atomic rename, same filesystem
        return True
    except OSError as e:
        # Covers Drive-quota-exceeded and other disk-write failures
        print(f'  SAVE FAILED for {final_path}: {e}')
        if Path(tmp_path).exists():
            try:
                os.remove(tmp_path)  # don't leave a stray partial temp file
            except OSError:
                pass
        return False

def train_protonet(model, train_wrap, val_wrap, n_way, total_epochs=60):
    start_epoch = 1
    best_val    = 0.
    history     = []

    enc_p  = [p for p in model.encoder.parameters() if p.requires_grad]
    proj_p = list(model.proj.parameters()) + list(model.drop.parameters())
    optimizer = torch.optim.AdamW([
        {'params': enc_p,  'lr': CFG['encoder_lr']},
        {'params': proj_p, 'lr': CFG['projector_lr']},
    ], weight_decay=CFG['weight_decay'])

    warmup = 5
    def lr_fn(ep):
        if ep < warmup: return (ep+1)/warmup
        return 0.5*(1+np.cos(np.pi*(ep-warmup)/max(1,total_epochs-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler    = torch.amp.GradScaler('cuda')

    # Attempt resume 
    if Path(RESUME_PATH).exists():
        ck = torch.load(RESUME_PATH, map_location='cpu', weights_only=False)
        if ck.get('total_epochs') != total_epochs:
            print(f'  RESUME file found but total_epochs mismatch '
                  f'(file={ck.get("total_epochs")}, requested={total_epochs}). '
                  f'Refusing to resume — delete {RESUME_PATH} to start fresh, '
                  f'or call train_protonet with total_epochs={ck.get("total_epochs")}.')
            raise ValueError('total_epochs mismatch with existing resume checkpoint')
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optimizer_state'])
        scheduler.load_state_dict(ck['scheduler_state'])
        scaler.load_state_dict(ck['scaler_state'])
        _load_rng_state(ck['rng_state'])
        start_epoch = ck['epoch'] + 1
        best_val    = ck['best_val']
        history     = ck['history']
        print(f'{"="*62}')
        print(f'  RESUMING from epoch {start_epoch}/{total_epochs} '
              f'(best_val so far: {best_val:.2f}%)')
        print(f'{"="*62}')

        pre_logging_gap_epochs = []
        for prior_epoch in range(1, start_epoch):
            n_tr = count_logged_episodes(prior_epoch, 'train')
            n_va = count_logged_episodes(prior_epoch, 'val')
            if n_tr == 0 and n_va == 0:
                pre_logging_gap_epochs.append(prior_epoch)
                continue
            if n_tr != CFG['n_ep_train'] or n_va != CFG['n_ep_val']:
                raise AssertionError(
                    f'Resume integrity check FAILED: epoch {prior_epoch} is '
                    f'marked complete in the resume checkpoint, and the '
                    f'episode CSV has SOME rows for it ({n_tr}/{CFG["n_ep_train"]} '
                    f'train, {n_va}/{CFG["n_ep_val"]} val) — not zero, so this '
                    f'isn\'t a "logging didn\'t exist yet" gap, it\'s a partial/'
                    f'corrupted write. The checkpoint and the CSV have gone out '
                    f'of sync (likely a mid-write disconnect) — inspect '
                    f'{EPISODE_LOG_PATH} before resuming.')
        if pre_logging_gap_epochs:
            print(f'  NOTE: epoch(s) {pre_logging_gap_epochs} have NO episode-'
                  f'level CSV rows — most likely trained before episode logging '
                  f'was added to this script. Epoch-level history (train_acc/'
                  f'val_acc/loss) for them is still intact via the resume '
                  f'checkpoint; only per-episode granularity is unavailable for '
                  f'these specific epochs. Flag this gap in Limitations if the '
                  f'episode-level CSV is later cited as complete for the full run.')
        verified_epochs = (start_epoch - 1) - len(pre_logging_gap_epochs)
        if verified_epochs > 0:
            print(f'  Episode log cross-check passed for {verified_epochs} '
                  f'fully-logged prior epoch(s) (out of {start_epoch-1} total)')
    else:
        print(f'{"="*62}')
        print(f'  ProtoNet Training — single phase, no freeze schedule (fresh start)')
        print(f'{"="*62}')

    if start_epoch > total_epochs:
        print(f'  Already completed {total_epochs} epochs per resume file — nothing to do.')
        model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=False))
        return history, BEST_PATH

    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    print(f'  Total epochs: {total_epochs} | trainable: {train_p:.1f}M (all unfrozen)')
    print(f'  n_way:{n_way} k_shot:{CFG["k_shot_train"]} n_query:{CFG["n_query"]}')
    print(f'  enc_lr:{CFG["encoder_lr"]:.0e}  proj_lr:{CFG["projector_lr"]:.0e}')

    _ensure_episode_log_header()
    print(f'  Episode-level log → {EPISODE_LOG_PATH} (appending, header preserved)')

    for epoch in range(start_epoch, total_epochs + 1):

        t_sampler = EpisodeSampler(train_wrap, n_way, CFG['k_shot_train'],
                                   CFG['n_query'], CFG['n_ep_train'])
        v_sampler = EpisodeSampler(val_wrap, n_way, 5,
                                   CFG['n_query'], CFG['n_ep_val'])

        # Train episodes — log each one as it actually completes
        model.train(); tl, ta = [], []
        with open(EPISODE_LOG_PATH, 'a', newline='') as log_f:
            writer = csv.DictWriter(log_f, fieldnames=EPISODE_LOG_FIELDS)
            for ep_idx, (si,sl,qi,ql) in enumerate(
                    tqdm(t_sampler, desc=f'E{epoch:03d}[tr]', leave=False)):
                si=si.to(DEVICE); sl=sl.to(DEVICE)
                qi=qi.to(DEVICE); ql=ql.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    logits = model(si,sl,qi,n_way) / CFG['temperature']
                    loss   = F.cross_entropy(logits, ql,
                                 label_smoothing=CFG['label_smooth'])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer); scaler.update()
                loss_val = loss.item()
                acc_val  = ep_acc(logits, ql)
                tl.append(loss_val); ta.append(acc_val)
                _log_episode(log_f, writer, epoch, 'train', ep_idx,
                             CFG['k_shot_train'], loss_val, acc_val)

        # Val episodes — same real-time, per-episode logging 
        model.eval(); va = []
        with torch.no_grad(), open(EPISODE_LOG_PATH, 'a', newline='') as log_f:
            writer = csv.DictWriter(log_f, fieldnames=EPISODE_LOG_FIELDS)
            for ep_idx, (si,sl,qi,ql) in enumerate(
                    tqdm(v_sampler, desc=f'E{epoch:03d}[v]', leave=False)):
                si=si.to(DEVICE); sl=sl.to(DEVICE); qi=qi.to(DEVICE)
                acc_val = ep_acc(model(si,sl,qi,n_way), ql)
                va.append(acc_val)
                _log_episode(log_f, writer, epoch, 'val', ep_idx,
                             5, None, acc_val)
        scheduler.step()

        n_train_logged = count_logged_episodes(epoch, 'train')
        n_val_logged   = count_logged_episodes(epoch, 'val')
        assert n_train_logged == CFG['n_ep_train'], (
            f'Episode log integrity check FAILED at epoch {epoch}: '
            f'expected {CFG["n_ep_train"]} train episode rows, found '
            f'{n_train_logged} in {EPISODE_LOG_PATH}. Do not trust this run\'s '
            f'CSV for this epoch — investigate before continuing.')
        assert n_val_logged == CFG['n_ep_val'], (
            f'Episode log integrity check FAILED at epoch {epoch}: '
            f'expected {CFG["n_ep_val"]} val episode rows, found '
            f'{n_val_logged} in {EPISODE_LOG_PATH}. Do not trust this run\'s '
            f'CSV for this epoch — investigate before continuing.')

        t_a = np.mean(ta)*100
        v_a = np.mean(va)*100
        l   = np.mean(tl)
        lr_now = optimizer.param_groups[0]['lr']
        history.append({'epoch':epoch, 'train_acc':t_a, 'val_acc':v_a, 'loss':l})

        if v_a > best_val:
            best_val = v_a
            _atomic_torch_save(model.state_dict(), BEST_PATH)

        resume_payload = {
            'epoch'           : epoch,
            'total_epochs'    : total_epochs,
            'model_state'     : model.state_dict(),
            'optimizer_state' : optimizer.state_dict(),
            'scheduler_state' : scheduler.state_dict(),
            'scaler_state'    : scaler.state_dict(),
            'rng_state'       : _rng_state_dict(),
            'best_val'        : best_val,
            'history'         : history,
        }
        save_ok = _atomic_torch_save(resume_payload, RESUME_PATH)
        if not save_ok:
            print(f'  WARNING: resume checkpoint save FAILED at epoch {epoch} '
                  f'(likely Drive quota — check the storage warning in Colab). '
                  f'Training continues, but if the runtime disconnects now, '
                  f'resume will restart from the last SUCCESSFUL save, not '
                  f'this epoch. Free up Drive space before the next disconnect.')

        if epoch % 5 == 0 or epoch <= 3 or epoch == start_epoch:
            print(f'  E{epoch:03d} | tr:{t_a:.1f}% val:{v_a:.1f}% '
                  f'loss:{l:.4f} best:{best_val:.1f}% lr:{lr_now:.2e} '
                  f'[resume {"saved" if save_ok else "SAVE FAILED"} | '
                  f'{n_train_logged}+{n_val_logged} episodes logged & verified]')

        if epoch == 15 and best_val < 20.:
            print('  Accuracy very low at ep15 — verify SSL ckpt and data paths')

    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=False))
    print(f'\n{"="*62}')
    print(f'   Training complete | best_val_acc = {best_val:.2f}%')
    print(f'   Best checkpoint   → {BEST_PATH}')
    print(f'   Resume file       → {RESUME_PATH} (kept — delete manually once done)')
    print(f'{"="*62}')
    return history, BEST_PATH


# Run (Resuable)
set_seed(SEED)
history, ckpt_path = train_protonet(
    protonet, train_wrap, val_wrap, n_way,
    total_epochs = 60,
)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

EPISODE_LOG_PATH = f"{CFG['output_dir']}/episode_log_nofreeze60.csv"

assert Path(EPISODE_LOG_PATH).exists(), (
    f'Episode log not found at {EPISODE_LOG_PATH}. This cell only plots '
    f'real logged results — run/resume training first (Cell 10) so the '
    f'log actually exists. Nothing will be estimated or filled in here.'
)

raw = pd.read_csv(EPISODE_LOG_PATH)
assert len(raw) > 0, f'{EPISODE_LOG_PATH} exists but is empty — no episodes logged yet.'

# Integrity check: every epoch present must have the FULL configured
counts = raw.groupby(['epoch', 'split']).size().unstack(fill_value=0)
bad_train = counts.index[counts.get('train', 0) != CFG['n_ep_train']] \
            if 'train' in counts.columns else counts.index
bad_val   = counts.index[counts.get('val', 0) != CFG['n_ep_val']] \
            if 'val' in counts.columns else counts.index
if len(bad_train) > 0 or len(bad_val) > 0:
    raise AssertionError(
        f'Episode log integrity check failed before plotting.\n'
        f'Epochs with wrong train-episode count (expected {CFG["n_ep_train"]}): '
        f'{list(bad_train)}\n'
        f'Epochs with wrong val-episode count (expected {CFG["n_ep_val"]}): '
        f'{list(bad_val)}\n'
        f'Fix or re-run the affected epochs before trusting this plot.'
    )

# Aggregate real per-episode rows into per-epoch means 
train_rows = raw[raw['split'] == 'train']
val_rows   = raw[raw['split'] == 'val']

train_agg = train_rows.groupby('epoch').agg(
    train_acc=('accuracy', lambda s: s.mean() * 100),
    loss=('loss', 'mean'),
).reset_index()

val_agg = val_rows.groupby('epoch').agg(
    val_acc=('accuracy', lambda s: s.mean() * 100),
).reset_index()

df_hist = train_agg.merge(val_agg, on='epoch', how='inner').sort_values('epoch')

n_epochs_logged = df_hist['epoch'].nunique()
print(f'Loaded {len(raw):,} real episode rows from {EPISODE_LOG_PATH}')
print(f'Aggregated into {n_epochs_logged} epoch(s) — all passed the '
      f'per-epoch episode-count integrity check')

# Save the aggregated (but fully traceable-to-source) per-epoch table
df_hist.to_csv(f"{CFG['output_dir']}/training_history.csv", index=False)

#  Plot
fig.suptitle('SwinProtoNet Training — 8-way 5-shot on Kvasir v2 '
             '(no freeze schedule, single phase)',
             fontsize=13, fontweight='bold')

# Accuracy plot
axes[0].plot(df_hist['epoch'], df_hist['train_acc'], 'b-o', ms=5, lw=2, label='Train')
axes[0].plot(df_hist['epoch'], df_hist['val_acc'],   'darkorange', marker='o',
             ms=5, lw=2, ls='--', label='Val')
axes[0].axhline(90, ls=':', color='green', lw=1.5, label='90% line')
axes[0].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy %',
            ylim=(max(0, df_hist[['train_acc','val_acc']].min().min() - 5), 100))
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# Loss plot (train only — val has no loss column, forward-pass-only)
axes[1].plot(df_hist['epoch'], df_hist['loss'], 'b-o', ms=5, lw=2, label='Train loss')
axes[1].set(title='Episode Loss', xlabel='Epoch', ylabel='Loss')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CFG['output_dir']}/training_curves.png", dpi=150)
plt.show()

best_row = df_hist.loc[df_hist['val_acc'].idxmax()]
print(f' Training curves saved → training_curves.png')
print(f' Best val accuracy so far: {best_row["val_acc"]:.1f}% '
      f'(epoch {int(best_row["epoch"])})')
print(f' Checkpoint: {BEST_PATH if "BEST_PATH" in dir() else CFG["output_dir"]+"/protonet_best.pth"}')
