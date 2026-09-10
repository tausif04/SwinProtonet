
""" ## Cell 0 — GPU Check """

import os, gc, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
gc.collect(); torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f'GPU  : {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info()
    print(f'VRAM : {free/1024**3:.1f} / {total/1024**3:.1f} GiB free')
else:
    raise RuntimeError('No GPU — switch to a GPU runtime in Colab.')

"""## Cell 1 — Install Packages"""

import subprocess, sys
for pkg in ['timm==0.9.16', 'albumentations>=1.3.0', 'einops', 'tqdm']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
print('packages ready')

"""## Cell 2 — Imports & Seeds"""

import os, random, math, shutil, copy, warnings, csv
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE    = torch.device('cuda')
IMG_SIZE  = 224
EMBED_DIM = 768
print(f'PyTorch {torch.__version__}  |  timm {timm.__version__}  |  device {DEVICE}')

from google.colab import drive
drive.mount('/content/drive')

"""## Cell 3 — Mount Drive & Directories

Not pre-creating CAPSULE_DIR here anymore — we don't actually know the
real path until Cell 4 downloads the dataset and tells us where it landed.
Making an empty folder at a guessed path ahead of time was basically
setting a trap for ourselves.
"""

DRIVE_ROOT   = '/content/drive/MyDrive/SSL_FYDP'
OUTPUT_DIR   = f'{DRIVE_ROOT}/Outputs'
SSL_CKPT_OUT = f'{OUTPUT_DIR}/swin_ssl_pretrained.pth'
SSL_HISTORY_CSV = f'{OUTPUT_DIR}/ssl_training_history.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'SSL checkpoint will be saved → {SSL_CKPT_OUT}')
print(f'SSL training history will be logged → {SSL_HISTORY_CSV}')

"""## Cell 4 — Download Kvasir-Capsule """

import kagglehub

downloaded_path = kagglehub.dataset_download("mdtausifbinmozid/kvasir-capsule")
print("Path to dataset files:", downloaded_path)

CAPSULE_DIR = downloaded_path

# quick sanity check before we go any further — better to catch an empty
_n_images_found = sum(
    1 for p in Path(CAPSULE_DIR).rglob('*')
    if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
)
assert _n_images_found > 0, (
    f'CAPSULE_DIR resolved to {CAPSULE_DIR!r} but there are zero images in '
    f'it. Either the kagglehub download didn\'t finish or the path is '
    f'wrong — stop here, don\'t let Cell 5 build an empty corpus.'
)
print(f'Verified: {_n_images_found:,} image files found under {CAPSULE_DIR}')

"""## Cell 5 — Filter SSL Corpus


"""

# Classes to EXCLUDE
EXCLUDE = {'pylorus', 'polyp', 'ulcer', 'normal mucosa',
           'normal-mucosa', 'normal_mucosa'}

def is_excluded(path: Path) -> bool:
    names = {p.name.lower().replace('-','').replace('_','').replace(' ','')
             for p in path.parents}
    clean = {e.replace('-','').replace('_','').replace(' ','') for e in EXCLUDE}
    return bool(names & clean)

ssl_frames, skipped = [], set()
for p in Path(CAPSULE_DIR).rglob('*'):
    if p.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
    if is_excluded(p):
        skipped.add(p.parent.name); continue
    ssl_frames.append(p)

# same idea as the check in Cell 4 — if this list is empty something's

assert len(ssl_frames) > 0, (
    'ssl_frames is empty after filtering. Either CAPSULE_DIR wasn\'t what '
    'we thought, or the EXCLUDE keywords are matching everything. Check '
    '`skipped` above before going further.'
)

rng = random.Random(SEED)
rng.shuffle(ssl_frames)

print(f'SSL corpus  : {len(ssl_frames):,} frames  (KEPT)')
print(f'Excluded dirs: {sorted(skipped)}')
print('Kept classes (disjoint from Kvasir v2):')
kept_dirs = {p.parent.name for p in ssl_frames}
for d in sorted(kept_dirs): print(f' {d}')

# 90/10 train/val split — NO image appears in both
split_n      = int(0.9 * len(ssl_frames))
ssl_train_paths = ssl_frames[:split_n]
ssl_val_paths   = ssl_frames[split_n:]
print(f'\nSSL train: {len(ssl_train_paths):,}  |  SSL val: {len(ssl_val_paths):,}')
assert not set(map(str,ssl_train_paths)) & set(map(str,ssl_val_paths)), 'SPLIT OVERLAP!'
print('No overlap between SSL train and val sets')

# Smart reduction — exclude classes with too few images 
MIN_CLASS_IMGS = 100    # drop any class below this threshold
MAX_PER_CLASS  = 500    # cap large classes at this

from collections import defaultdict

reduced_frames = []
class_buckets  = defaultdict(list)
for p in ssl_frames:
    class_buckets[p.parent.name].append(p)

rng2 = random.Random(SEED + 1)
kept_classes, dropped_classes = [], []

for cls, paths in class_buckets.items():
    if len(paths) < MIN_CLASS_IMGS:
        dropped_classes.append((cls, len(paths)))
        continue          # skip tiny classes entirely
    rng2.shuffle(paths)
    take = paths[:MAX_PER_CLASS]
    reduced_frames.extend(take)
    kept_classes.append((cls, len(take)))

rng2.shuffle(reduced_frames)
ssl_frames = reduced_frames

# belt and suspenders — make sure the reduction step didn't wipe everything
assert len(ssl_frames) > 0, (
    'Nothing left after the MIN_CLASS_IMGS filter — every class had fewer '
    'than MIN_CLASS_IMGS images. Check kept_classes/dropped_classes above.'
)

# Rebuild split
split_n         = int(0.9 * len(ssl_frames))
ssl_train_paths = ssl_frames[:split_n]
ssl_val_paths   = ssl_frames[split_n:]

print(f'{"═"*50}')
print(f'  KEPT classes ({len(kept_classes)}):')
for cls, n in sorted(kept_classes, key=lambda x:-x[1]):
    bar = '█' * (n // 50)
    print(f'   {cls:<30}: {n:>4}  {bar}')

print(f'\n  DROPPED classes (too few images):')
for cls, n in dropped_classes:
    print(f'   {cls:<30}: {n:>4}  (below {MIN_CLASS_IMGS} threshold)')

print(f'\n  Total SSL frames : {len(ssl_frames):,}')
print(f'  SSL train        : {len(ssl_train_paths):,}')
print(f'  SSL val          : {len(ssl_val_paths):,}')

assert not set(map(str,ssl_train_paths)) & set(map(str,ssl_val_paths))
print('No overlap between SSL train and val')

"""## Cell 6 — Domain-Specific Augmentation"""

# Two-view augmentation for DINO (student sees two different crops)
class SpecularBlob:
    """Simulate endoscope specular highlight reflection."""
    def __call__(self, img):
        if random.random() > 0.35: return img
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        cx, cy = random.randint(0,w), random.randint(0,h)
        r = random.randint(8, 35)
        Y, X = np.ogrid[:h,:w]
        mask = ((X-cx)**2+(Y-cy)**2 <= r**2).astype(np.float32)
        arr = np.clip(arr + random.uniform(0.6,1.0)*255*mask[...,None], 0,255)
        return Image.fromarray(arr.astype(np.uint8))

class MotionBlur:
    """Simulate colonoscope motion/focus blur."""
    def __call__(self, img):
        if random.random() > 0.25: return img
        from PIL import ImageFilter
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,2.5)))

def make_dino_view(size=IMG_SIZE, scale=(0.25,1.0), global_=True):
    s = (0.5,1.0) if global_ else scale
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=s, interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(0.4,0.4,0.2,0.1),
        transforms.RandomGrayscale(p=0.1),
        SpecularBlob(), MotionBlur(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

global_view1 = make_dino_view(global_=True)
global_view2 = make_dino_view(global_=True)
print('Augmentation pipeline ready')

"""## Cell 7 — Dataset Classes"""

class DINODataset(Dataset):
    """Returns two augmented views of each image for DINO."""
    def __init__(self, paths, view1, view2):
        self.paths = paths
        self.v1, self.v2 = view1, view2
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.v1(img), self.v2(img)

class ValDataset(Dataset):
    """Single-view dataset for SSL validation loss."""
    def __init__(self, paths, tfm):
        self.paths, self.tfm = paths, tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.tfm(Image.open(self.paths[idx]).convert('RGB'))

train_ds = DINODataset(ssl_train_paths, global_view1, global_view2)
val_ds   = ValDataset(ssl_val_paths, val_tfm)

BATCH = 32   # reduce to 16 if OOM
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=0, pin_memory=True)

assert len(train_loader) > 0, (
    f'train_loader has 0 batches (ssl_train_paths={len(ssl_train_paths)}, '
    f'BATCH={BATCH}, drop_last=True). Corpus too small for this batch size?'
)
assert len(val_loader) > 0, (
    f'val_loader has 0 batches (ssl_val_paths={len(ssl_val_paths)}). '
    f'Same deal, fix before training.'
)
print(f'Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}')

"""## Cell 8 — Swin-Tiny Encoder (matches Part 2 exactly)"""

class SwinEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # swin_tiny — SAME model name used in Part 2
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained, num_classes=0, global_pool='avg')
        self.embed_dim = self.backbone.num_features  # 768
    def forward(self, x):
        return self.backbone(x)   # (B, 768)

# Sanity check
_enc = SwinEncoder(pretrained=False).to(DEVICE)
with torch.no_grad():
    _out = _enc(torch.randn(2,3,224,224).to(DEVICE))
assert _out.shape == (2, 768), f'Expected (2,768), got {_out.shape}'
del _enc; print('SwinEncoder → output (B, 768)')

"""## Cell 9 — DINO Model (Student + Teacher + Head) """

class DINOHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=4096, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck))
        self.last = nn.utils.weight_norm(
            nn.Linear(bottleneck, out_dim, bias=False))
        self.last.weight_g.data.fill_(1)
        self.last.weight_g.requires_grad = False
    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1)
        return self.last(x)

class DINOLoss(nn.Module):
    def __init__(self, out_dim=4096, t_temp=0.04, s_temp=0.1, center_m=0.9):
        super().__init__()
        self.s_temp, self.t_temp, self.cm = s_temp, t_temp, center_m
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, s_out, t_out):
        # just the loss now, no side effects on self.center — call
        # update_center() separately once you've got both teacher outputs
        s = F.log_softmax(s_out / self.s_temp, dim=-1)
        t = F.softmax((t_out - self.center) / self.t_temp, dim=-1).detach()
        loss = -(t * s).sum(-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, t1, t2):
        # average both teacher views together before updating, so we're
        # not nudging the center twice per step in a row
        batch_center = torch.cat([t1, t2], dim=0).mean(0, keepdim=True)
        self.center = self.center * self.cm + batch_center * (1 - self.cm)

class DINOModel(nn.Module):
    def __init__(self, out_dim=4096, ema_decay=0.996):
        super().__init__()
        self.student_enc  = SwinEncoder(pretrained=True)
        self.teacher_enc  = SwinEncoder(pretrained=True)
        self.student_head = DINOHead(768, out_dim)
        self.teacher_head = DINOHead(768, out_dim)
        # Teacher is EMA of student — no grad
        for p in self.teacher_enc.parameters():  p.requires_grad = False
        for p in self.teacher_head.parameters(): p.requires_grad = False
        self.ema_decay = ema_decay
    @torch.no_grad()
    def update_teacher(self, decay=None):
        d = decay or self.ema_decay
        for s, t in zip(self.student_enc.parameters(), self.teacher_enc.parameters()):
            t.data = d*t.data + (1-d)*s.data
        for s, t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            t.data = d*t.data + (1-d)*s.data
    def forward(self, v1, v2):
        s1 = self.student_head(self.student_enc(v1))
        s2 = self.student_head(self.student_enc(v2))
        with torch.no_grad():
            t1 = self.teacher_head(self.teacher_enc(v1))
            t2 = self.teacher_head(self.teacher_enc(v2))
        return s1, s2, t1, t2

print('DINOModel defined')

"""## Cell 10 — SSL Validation (cosine similarity)"""

@torch.no_grad()
def validate_ssl(model, loader):
    model.eval()
    sims = []
    for imgs in loader:
        imgs = imgs.to(DEVICE)
        # Two deterministic crops of each image
        f1 = F.normalize(model.teacher_enc(imgs), dim=-1)
        f2 = F.normalize(model.student_enc(imgs), dim=-1)
        sims.append((f1 * f2).sum(-1).mean().item())
    return float(np.mean(sims))

print('Validation function ready')

"""## Cell 11 — Training Loop """

def train_dino(
    epochs    = 100,
    lr        = 3e-4,
    wd        = 0.05,
    warmup_ep = 10,
    out_dim   = 4096,
    save_path = SSL_CKPT_OUT,
    history_csv = SSL_HISTORY_CSV,
):
    model     = DINOModel(out_dim=out_dim).to(DEVICE)
    criterion = DINOLoss(out_dim=out_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(model.student_enc.parameters()) +
        list(model.student_head.parameters()),
        lr=lr, weight_decay=wd)

    def lr_sched(ep):
        if ep < warmup_ep: return (ep+1)/warmup_ep
        prog = (ep-warmup_ep) / max(1, epochs-warmup_ep)
        return 0.5*(1+math.cos(math.pi*prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)
    scaler    = torch.amp.GradScaler('cuda')

    # start a fresh log for this run
    with open(history_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'loss', 'val_sim', 'lr', 'ema_decay'])

    best_sim, history = -1., []
    print(f'DINO training | {epochs} epochs | batch {BATCH} | lr {lr}')
    print(f'   SSL corpus : {len(ssl_train_paths):,} frames ')

    for epoch in range(1, epochs+1):
        model.train(); losses = []
        # Cosine EMA decay schedule
        ema_d = 0.996 + 0.004*((1-math.cos(math.pi*epoch/epochs))/2)

        for v1, v2 in tqdm(train_loader, desc=f'Ep{epoch:03d}', leave=False):
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                s1,s2,t1,t2 = model(v1,v2)
                loss = (criterion(s1,t2) + criterion(s2,t1)) / 2
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.student_enc.parameters()) +
                list(model.student_head.parameters()), 3.0)
            scaler.step(optimizer); scaler.update()
            model.update_teacher(ema_d)
            # one center update per step, using both teacher views together
            criterion.update_center(t1.detach(), t2.detach())
            losses.append(loss.item())

        scheduler.step()
        val_sim = validate_ssl(model, val_loader)
        avg_loss = np.mean(losses)
        lr_now = optimizer.param_groups[0]['lr']
        history.append({'epoch':epoch,'loss':avg_loss,'val_sim':val_sim})

        # write this epoch to disk right away — don't wait until the end
        with open(history_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_loss, val_sim, lr_now, ema_d])

        if val_sim > best_sim:
            best_sim = val_sim
            # Save ONLY the encoder state dict (not head, not teacher)
            torch.save({
                'epoch'        : epoch,
                'encoder_state': model.student_enc.backbone.state_dict(),
                'val_sim'      : val_sim,
                'loss'         : avg_loss,
                'method'       : 'dino',
                'backbone'     : 'swin_tiny_patch4_window7_224',
                'embed_dim'    : 768,
            }, save_path)

        if epoch % 10 == 0 or epoch <= 5:
            print(f'  Ep{epoch:03d} | loss {avg_loss:.4f} | '
                  f'val_sim {val_sim:.4f} | best {best_sim:.4f} | '
                  f'lr {lr_now:.2e} | ema {ema_d:.4f}')

    print(f'\n DINO done | best val_sim={best_sim:.4f}')
    print(f' Checkpoint → {save_path}')
    print(f' Full per-epoch history logged → {history_csv}')
    return history

history = train_dino(epochs=100, lr=3e-4)

"""## Cell 12 — Training Curves & Checkpoint Verification """

import torch, pandas as pd, matplotlib.pyplot as plt

ckpt_path = SSL_CKPT_OUT
assert Path(ckpt_path).exists(), (
    f'No checkpoint at {ckpt_path}. If Cell 11 ran but this is missing, '
    f'val_sim was probably NaN every epoch (empty corpus somewhere '
    f'upstream) so the save-on-improvement check never triggered. Check '
    f'{SSL_HISTORY_CSV} for NaNs first.'
)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
n_params = sum(v.numel() for v in ckpt['encoder_state'].values())

print('='*55)
print('  SSL PRETRAINING COMPLETE')
print('='*55)
print(f'  Backbone   : {ckpt.get("backbone","swin_tiny_patch4_window7_224")}')
print(f'  Embed dim  : {ckpt.get("embed_dim", 768)}')
print(f'  Best epoch : {ckpt.get("epoch")}')
print(f'  Best sim   : {ckpt.get("val_sim"):.4f}')
print(f'  Params     : {n_params/1e6:.1f} M')
print(f'  Saved to   : {ckpt_path}')
print('='*55)

# Load the real history — nothing hand-typed here 
assert Path(SSL_HISTORY_CSV).exists(), (
    f'{SSL_HISTORY_CSV} not found. Run Cell 11 first (or make sure it '
    f'wrote to this path in a previous session) — we\'re not filling in '
    f'placeholder numbers here.'
)
df = pd.read_csv(SSL_HISTORY_CSV)
assert len(df) > 0, f'{SSL_HISTORY_CSV} exists but has nothing in it.'
assert not df['val_sim'].isna().any(), (
    'val_sim has NaN somewhere in the log — almost always means the val '
    'loader was empty at some point. Figure that out before trusting '
    'this plot.'
)
print(f'\nLoaded {len(df)} real logged epochs from {SSL_HISTORY_CSV}')

# Plot 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('DINO SSL Pretraining — Swin-Tiny on Kvasir-Capsule (Filtered)',
             fontsize=13, fontweight='bold')

# Loss curve
axes[0].plot(df['epoch'], df['loss'], 'b-o', lw=2, ms=5)
axes[0].fill_between(df['epoch'], df['loss'], alpha=0.1, color='blue')
axes[0].set(title='Training Loss (↓ better)',
            xlabel='Epoch', ylabel='DINO Loss')
axes[0].grid(alpha=0.3)
final_loss = df['loss'].iloc[-1]
final_epoch = int(df['epoch'].iloc[-1])
axes[0].annotate(f"Final: {final_loss:.4f}",
                 xy=(final_epoch, final_loss),
                 xytext=(max(1, final_epoch*0.7), df['loss'].max()*0.3),
                 arrowprops=dict(arrowstyle='->', color='blue'),
                 fontsize=10, color='blue')

# Val sim curve
axes[1].plot(df['epoch'], df['val_sim'], 'g-o', lw=2, ms=5)
axes[1].fill_between(df['epoch'], df['val_sim'], alpha=0.1, color='green')
axes[1].axhline(0.99, ls='--', color='red', lw=1.5, label='0.99 threshold')
axes[1].set(title='Val Cosine Similarity (↑ better)',
            xlabel='Epoch', ylabel='Cosine Similarity',
            ylim=(min(0.80, df['val_sim'].min()-0.02), 1.02))
axes[1].grid(alpha=0.3)
axes[1].legend()
final_sim = df['val_sim'].iloc[-1]
axes[1].annotate(f"Final: {final_sim:.4f}",
                 xy=(final_epoch, final_sim),
                 xytext=(max(1, final_epoch*0.65), df['val_sim'].min()+0.02),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=10, color='green')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ssl_training_curves.png',
            dpi=150, bbox_inches='tight')
plt.show()
print('Plot saved → ssl_training_curves.png (real logged numbers, nothing typed in)')
print(f'   Use ssl_ckpt: "{ckpt_path}"')
