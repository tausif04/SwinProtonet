# Cell 0 - Initial setup.

                       

import os, gc, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
gc.collect(); torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f'GPU  : {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info()
    print(f'VRAM : {free/1024**3:.1f} / {total/1024**3:.1f} GiB free')
else:
    raise RuntimeError('No GPU — switch to a GPU runtime in Colab.')

# Cell 1 - 1 — Install Packages.

import subprocess, sys
for pkg in ['timm==0.9.16', 'albumentations>=1.3.0', 'einops', 'tqdm', 'scikit-learn', 'scipy']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
print('packages ready')

# Cell 2 - 2 — Imports & Seeds.

import os, random, json, warnings, time, shutil
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from scipy import stats as sp_stats
from sklearn.metrics import f1_score, balanced_accuracy_score
warnings.filterwarnings('ignore')

SEED = 42
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device {DEVICE} | timm {timm.__version__}')

# Cell 3 - 3 — Configuration.

CFG = {
    'drive_root'    : '/content/drive/MyDrive/SSL_FYDP',
    'output_dir'    : '/content/drive/MyDrive/SSL_FYDP/Outputs/',
    'labeled_root'  : '/content/labeled_data',
    'kvasir_v2_dir' : '/content/drive/MyDrive/SSL_FYDP/kvasir-dataset-v2',
    'ssl_ckpt'      : '/content/drive/MyDrive/SSL_FYDP/Outputs/swin_ssl_pretrained.pth',
    'split_file'    : '/content/drive/MyDrive/SSL_FYDP/Outputs/split_indices.json',
    'backbone'      : 'swin_tiny_patch4_window7_224',
    'embed_dim'     : 768,
    'proj_dim'      : 768,                                               
    'img_size'      : 224,
    'max_per_class' : 300,
                                                                         
    'n_way'         : 8,
    'k_shots'       : [1, 5],
    'n_query'       : 15,
    'n_episodes'    : 600,                                  
    'seeds'         : [42, 43, 44, 45, 46],
                                                                      
                                                        
    'meta_epochs'        : 60,
    'meta_episodes_per_ep': 60,
    'meta_k_shot'         : 5,                                                    
    'meta_n_query'        : 15,
    'meta_lr'             : 1e-4,
    'meta_temperature'    : 0.5,                                        
    'meta_label_smooth'   : 0.1,                                         
    'meta_grad_clip'      : 5.0,                                   
    'meta_n_ep_val'       : 30,                                                           
    'dropout_rate'        : 0.1,                                              
    'cls_epochs'          : 60,
    'cls_batch_size'      : 32,
    'cls_lr'              : 1e-4,
    'dist_scale'          : 10.0,                                              
                                                                                            
                 
    'relationnet_hidden'  : 128,
    'relnet_meta_epochs'  : 60,
    'relnet_meta_episodes_per_ep': 60,
    'relnet_meta_lr'      : 1e-3,
                                                                        
                                                                         
                                                                        
    'baselinepp_ft_steps' : 100,
    'baselinepp_ft_lr'    : 0.01,
    'baseline_ft_steps'   : 100,
    'baseline_ft_lr'      : 0.01,
                                                                 
                                                                  
                                                                    
                                                             
    'maml_meta_epochs'        : 60,
    'maml_meta_episodes_per_ep': 60,
    'maml_meta_batch'         : 4,                                            
    'maml_meta_lr'            : 1e-3,
    'maml_inner_lr'           : 0.05,
    'maml_inner_steps'        : 5,                                             
    'maml_inner_steps_eval'   : 10,                                                            
    'maml_first_order'       : False,                                
                                                        
                                                                    
                                                      
    'transformer_meta_epochs'        : 20,
    'transformer_meta_episodes_per_ep': 60,
    'transformer_meta_lr'            : 1e-4,
    'transformer_n_heads'            : 4,
    'transformer_ff_dim'             : 512,
    'transformer_dropout'            : 0.1,
                                                               
                                                                  
    'resnet50_ft_steps'   : 100,
    'resnet50_ft_lr'      : 0.01,
}
SIZE = CFG['img_size']

                           
BASE_EP_CSV       = f"{CFG['output_dir']}/baseline_comparison_episode_metrics.csv"
BASE_SUM_CSV      = f"{CFG['output_dir']}/baseline_comparison_summary.csv"
BASE_SIG_CSV      = f"{CFG['output_dir']}/baseline_comparison_significance.csv"
PROGRESS_JSON     = f"{CFG['output_dir']}/baseline_comparison_progress.json"
PROJ_EPISODIC_CKPT      = f"{CFG['output_dir']}/baseline_proj_episodic.pth"
                                                                 
                                                                     
                                                                     
                                              
PROJ_CLASSIFICATION_CKPT = f"{CFG['output_dir']}/baseline_proj_classification_softmax.pth"
RELATIONNET_CKPT        = f"{CFG['output_dir']}/baseline_relationnet_module.pth"
MAML_CKPT               = f"{CFG['output_dir']}/baseline_maml_head.pth"
TRANSFORMER_CKPT        = f"{CFG['output_dir']}/baseline_transformer_adapter.pth"

os.makedirs(CFG['output_dir'], exist_ok=True)
print('Config ready | n_way:', CFG['n_way'], '| k_shots:', CFG['k_shots'],
      '| seeds:', CFG['seeds'], '| episodes/cell:', CFG['n_episodes'])

# Cell 4 - 4 — Mount Drive & Load Split (robust to shared-drive sync lag).

from google.colab import drive
import os
import shutil                           

                                        
                                                                              
try:
    drive.flush_and_unmount()
except Exception:
    pass                                                  

mount_point = '/content/drive'

                                                                       
                                                                          
                                                                                    
if os.path.exists(mount_point):
    print(f"Directory {mount_point} exists. Checking its contents.")
    if os.path.isdir(mount_point) and len(os.listdir(mount_point)) > 0:
        print(f"Mount point {mount_point} contains files. Attempting to clear it.")
        try:
                                                               
            shutil.rmtree(mount_point)
                                         
            os.makedirs(mount_point)
            print(f"Cleared and recreated empty directory: {mount_point}")
        except OSError as e:
            print(f"Error clearing and recreating {mount_point}: {e}. Proceeding anyway.")
    elif not os.path.isdir(mount_point):
        print(f"Warning: {mount_point} exists but is not a directory. Removing and recreating.")
        try:
            os.remove(mount_point)
            os.makedirs(mount_point)
        except OSError as e:
            print(f"Error removing/recreating non-directory {mount_point}: {e}. Proceeding anyway.")
else:
                                                                              
    os.makedirs(mount_point, exist_ok=True)


                    
drive.mount(mount_point, force_remount=True)
print('Drive mounted')

split_path = Path(CFG['split_file'])
for attempt in range(10):
    if split_path.exists():
        break
    time.sleep(2)
else:
    outputs_dir = Path(CFG['output_dir'])
    print(f'Looking for : {split_path}')
    if outputs_dir.parent.exists():
        print(f'Contents of {outputs_dir.parent}:')
        for p in outputs_dir.parent.iterdir(): print(f'  {p.name}')
    if outputs_dir.exists():
        print(f'Contents of {outputs_dir}:')
        for p in outputs_dir.iterdir(): print(f'  {p.name}')

assert split_path.exists(), (
    f'{split_path} not found after remount + 20s poll. If SSL_FYDP is a '
    f'shared drive/shortcut, open it once at drive.google.com to force it '
    f'to sync into this account, then re-run this cell.'
)
with open(CFG['split_file']) as f:
    split_data = json.load(f)
train_idx = split_data['train']
val_idx   = split_data['val']
test_idx  = split_data['test']
assert split_data['seed'] == SEED
print(f'Split loaded | train:{len(train_idx)} val:{len(val_idx)} test:{len(test_idx)} | seed verified')

def find_output_file(filename, search_root=None):
    root = Path(search_root or CFG['output_dir'])
    matches = list(root.rglob(filename))
    if not matches:
        print(f'  [find_output_file] "{filename}" NOT FOUND under {root}')
        return None
    if len(matches) > 1:
        print(f'  [find_output_file] multiple matches for "{filename}", using first: {matches}')
    print(f'  [find_output_file] "{filename}" -> {matches[0]}')
    return matches[0]

# Cell 5 - 5 — Rebuild Labeled Dataset (identical reconstruction to Part 2).

KV2_PATH = Path(CFG['kvasir_v2_dir'])
assert KV2_PATH.exists(), f'Path not found: {KV2_PATH}'

                                                         
subdirs = [d for d in KV2_PATH.iterdir() if d.is_dir()]
kv2_root = KV2_PATH if len(subdirs) >= 6 else next(
    (s for s in subdirs if len([d for d in s.iterdir() if d.is_dir()]) >= 6), None)

assert kv2_root, f'Could not find class folders inside {KV2_PATH}'
print(f'Kvasir v2 root found at: {kv2_root}')

rng_dataset = random.Random(SEED)
if Path(CFG['labeled_root']).exists():
    shutil.rmtree(CFG['labeled_root'])

MAX = CFG['max_per_class']
MIN_NEEDED = CFG['meta_k_shot'] + CFG['meta_n_query'] + 10
all_classes = []

for cls_dir in sorted(kv2_root.iterdir()):
    if not cls_dir.is_dir(): continue
    imgs = sorted(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.png')))
    if len(imgs) < MIN_NEEDED: continue

    rng_dataset.shuffle(imgs)
    dst = Path(CFG['labeled_root']) / cls_dir.name
    dst.mkdir(parents=True, exist_ok=True)

                                                                       
    for src in imgs[:MAX]:
        shutil.copy(src, dst / src.name)
    all_classes.append(cls_dir.name)

print(f'{len(all_classes)} classes rebuilt (Target: 8)')
assert len(all_classes) == CFG['n_way'], f'Expected {CFG["n_way"]} classes, but found {len(all_classes)}'

# Cell 6 - 6 — Augmentations, Dataset Classes, Samplers.

class SpecularHighlight(A.ImageOnlyTransform):
    def __init__(self, n_spots=(1,3), radius=(5,25), p=0.5):
        super().__init__(p=p); self.n_spots=n_spots; self.radius=radius
    def apply(self, img, **kw):
        img=img.copy(); h,w=img.shape[:2]
        for _ in range(random.randint(*self.n_spots)):
            cx,cy=random.randint(0,w),random.randint(0,h)
            rx,ry=random.randint(*self.radius),random.randint(*self.radius)
            mask=np.zeros((h,w),np.float32)
            cv2.ellipse(mask,(cx,cy),(rx,ry),random.randint(0,180),0,360,1.,-1)
            mask=cv2.GaussianBlur(mask,(0,0),rx//2+1)
            img=np.clip(img.astype(np.float32)+mask[:,:,None]*255*
                        random.uniform(0.7,1.),0,255).astype(np.uint8)
        return img
    def get_transform_init_args_names(self): return ('n_spots','radius')

class Vignette(A.ImageOnlyTransform):
    def __init__(self, strength=(0.3,0.6), p=0.4):
        super().__init__(p=p); self.strength=strength
    def apply(self, img, **kw):
        h,w=img.shape[:2]; Y,X=np.ogrid[:h,:w]
        d=np.sqrt((X-w/2)**2+(Y-h/2)**2); d=d/d.max()
        return np.clip(img.astype(np.float32)*(1-random.uniform(*self.strength)*d)
                       [:,:,None],0,255).astype(np.uint8)
    def get_transform_init_args_names(self): return ('strength',)

train_aug = A.Compose([
    A.Resize(SIZE,SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.05,p=0.7),
    A.GaussianBlur(blur_limit=(3,5),p=0.3),
    SpecularHighlight(p=0.3), Vignette(p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)), ToTensorV2(),
])
val_aug = A.Compose([
    A.Resize(SIZE, SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2(),
])

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
                            self.samples.append((str(p), lbl))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert('RGB'))
        return self.transform(image=img)['image'], label

class SubsetWrapper(Dataset):
    def __init__(self, base, indices, transform):
        self.samples = [base.samples[i] for i in indices]
        self.classes = base.classes; self.class_to_idx = base.class_to_idx
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert('RGB'))
        return self.transform(image=img)['image'], label

class ImageEpisodeSampler:
    def __init__(self, dataset, n_way, k_shot, n_query, n_episodes, seed):
        self.ds = dataset; self.n_way = n_way; self.k_shot = k_shot
        self.n_query = n_query; self.n_episodes = n_episodes
        self.rng = random.Random(seed)
        self.class_indices = {}
        for i, (_, lbl) in enumerate(dataset.samples):
            self.class_indices.setdefault(lbl, []).append(i)
        self.available = [k for k, v in self.class_indices.items() if len(v) >= k_shot + n_query]
        assert len(self.available) >= n_way, f'Need >= {n_way} classes, got {len(self.available)}'
    def __len__(self): return self.n_episodes
    def sample_episode(self):
        classes = self.rng.sample(self.available, self.n_way)
        si, sl, qi, ql = [], [], [], []
        for local, cls in enumerate(classes):
            chosen = self.rng.sample(self.class_indices[cls], self.k_shot + self.n_query)
            for i in chosen[:self.k_shot]:
                img, _ = self.ds[i]; si.append(img); sl.append(local)
            for i in chosen[self.k_shot:]:
                img, _ = self.ds[i]; qi.append(img); ql.append(local)
        return (torch.stack(si), torch.tensor(sl), torch.stack(qi), torch.tensor(ql))
    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self.sample_episode()

train_wrap = SubsetWrapper(LabeledPolyp([CFG['labeled_root']], train_aug), train_idx, train_aug)
val_wrap   = SubsetWrapper(LabeledPolyp([CFG['labeled_root']], val_aug),   val_idx,   val_aug)
test_wrap  = SubsetWrapper(LabeledPolyp([CFG['labeled_root']], val_aug),   test_idx,  val_aug)
print(f'train_wrap: {len(train_wrap)} | val_wrap: {len(val_wrap)} | test_wrap: {len(test_wrap)} '
      f'| classes: {test_wrap.classes}')

# Cell 7 - 7 — Frozen SSL Backbone + Trainable Projection Head.

class FrozenSSLBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG['backbone'], pretrained=True, num_classes=0, img_size=CFG['img_size'])
        ckpt = torch.load(CFG['ssl_ckpt'], map_location='cpu', weights_only=False)
        missing, unexpected = self.backbone.load_state_dict(ckpt['encoder_state'], strict=False)
        print(f'SSL weights loaded | missing:{len(missing)} unexpected:{len(unexpected)}')
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.embed_dim = self.backbone.num_features
    def forward(self, x):
        with torch.no_grad():
            f = self.backbone.forward_features(x)
            if   f.dim() == 4: f = f.mean(dim=[1,2])
            elif f.dim() == 3: f = f.mean(dim=1)
        return f                                 

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.pre_drop = nn.Dropout(p=dropout)
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2), nn.LayerNorm(out_dim * 2), nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim * 2, out_dim), nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.net(self.pre_drop(x))

backbone = FrozenSSLBackbone().to(DEVICE)
backbone.eval()
print(f'Frozen backbone ready | embed_dim={backbone.embed_dim}')

# Cell 8 - 8 — Stage A: Train the Two Projection Heads (resumable, skipped if checkpointed).

def train_episodic_projection():
    proj = ProjectionHead(backbone.embed_dim, CFG['proj_dim'], dropout=CFG['dropout_rate']).to(DEVICE)
    opt = torch.optim.AdamW(proj.parameters(), lr=CFG['meta_lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['meta_epochs'])
    best_val, best_state = -1.0, None
    patience_counter = 0                              
    print(f'Training proj_episodic | {CFG["meta_epochs"]} epochs x '
          f'{CFG["meta_episodes_per_ep"]} episodes | temperature={CFG["meta_temperature"]} '
          f'label_smooth={CFG["meta_label_smooth"]} grad_clip={CFG["meta_grad_clip"]} '
          f'(matching ssl_lp\'s real training recipe)')
    for epoch in range(1, CFG['meta_epochs'] + 1):
        t_sampler = ImageEpisodeSampler(train_wrap, CFG['n_way'], CFG['meta_k_shot'],
                                         CFG['meta_n_query'], CFG['meta_episodes_per_ep'],
                                         seed=1000 + epoch)
        v_sampler = ImageEpisodeSampler(val_wrap, CFG['n_way'], 5,
                                         CFG['meta_n_query'], CFG['meta_n_ep_val'],
                                         seed=5000 + epoch)
        proj.train(); losses, accs = [], []
        for si, sl, qi, ql in tqdm(t_sampler, desc=f'proj_episodic ep{epoch:02d}[tr]', leave=False):
            si, sl, qi, ql = si.to(DEVICE), sl.to(DEVICE), qi.to(DEVICE), ql.to(DEVICE)
            with torch.no_grad():
                s_feat = backbone(si); q_feat = backbone(qi)
            s_emb = F.normalize(proj(s_feat), dim=-1)
            q_emb = F.normalize(proj(q_feat), dim=-1)
            proto = torch.stack([s_emb[sl == c].mean(0) for c in range(CFG['n_way'])])
            logits = -torch.cdist(q_emb.unsqueeze(0), proto.unsqueeze(0)).squeeze(0)
            logits = logits / CFG['meta_temperature']
            loss = F.cross_entropy(logits, ql, label_smoothing=CFG['meta_label_smooth'])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), CFG['meta_grad_clip'])
            opt.step()
            losses.append(loss.item())
            accs.append((logits.argmax(-1) == ql).float().mean().item())
        scheduler.step()

        proj.eval(); vaccs = []
        with torch.no_grad():
            for si, sl, qi, ql in tqdm(v_sampler, desc=f'proj_episodic ep{epoch:02d}[val]', leave=False):
                si, sl, qi, ql = si.to(DEVICE), sl.to(DEVICE), qi.to(DEVICE), ql.to(DEVICE)
                s_feat = backbone(si); q_feat = backbone(qi)
                s_emb = F.normalize(proj(s_feat), dim=-1)
                q_emb = F.normalize(proj(q_feat), dim=-1)
                proto = torch.stack([s_emb[sl == c].mean(0) for c in range(CFG['n_way'])])
                logits = -torch.cdist(q_emb.unsqueeze(0), proto.unsqueeze(0)).squeeze(0)
                vaccs.append((logits.argmax(-1) == ql).float().mean().item())
        val_acc = np.mean(vaccs) * 100
        print(f'  ep{epoch:02d} | train_loss {np.mean(losses):.4f} | '
              f'train_acc {np.mean(accs)*100:.1f}% | val_acc {val_acc:.1f}% | best {best_val:.1f}%')

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in proj.state_dict().items()}
            torch.save(best_state, PROJ_EPISODIC_CKPT)                                               
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f'  Early stopping at epoch {epoch} (no improvement for 20 epochs, best_val={best_val:.1f}%)')
                break

    proj.load_state_dict(best_state)                                            
    print(f'proj_episodic complete | best_val_acc={best_val:.1f}% | Saved -> {PROJ_EPISODIC_CKPT}')
    return proj

def train_classification_projection():
    n_way = CFG['n_way']
    proj = ProjectionHead(backbone.embed_dim, CFG['proj_dim'], dropout=CFG['dropout_rate']).to(DEVICE)
    classifier = nn.Linear(CFG['proj_dim'], n_way).to(DEVICE)
    opt = torch.optim.AdamW(list(proj.parameters()) + list(classifier.parameters()),
                             lr=CFG['cls_lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['cls_epochs'])
    train_loader = DataLoader(train_wrap, batch_size=CFG['cls_batch_size'], shuffle=True,
                               num_workers=0, drop_last=True)
    val_loader = DataLoader(val_wrap, batch_size=CFG['cls_batch_size'], shuffle=False, num_workers=0)
    best_val, best_state = -1.0, None
    patience_counter = 0                              
    print(f'Training proj_classification | {CFG["cls_epochs"]} epochs, plain softmax CE, '
          f'grad_clip={CFG["meta_grad_clip"]}, best-val checkpoint selection')
    for epoch in range(1, CFG['cls_epochs'] + 1):
        proj.train(); classifier.train(); losses, accs = [], []
        for x, y in tqdm(train_loader, desc=f'proj_classification ep{epoch:02d}[tr]', leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                feat = backbone(x)
            emb = proj(feat)
            logits = classifier(emb)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(proj.parameters()) + list(classifier.parameters()),
                                            CFG['meta_grad_clip'])
            opt.step()
            losses.append(loss.item())
            accs.append((logits.argmax(-1) == y).float().mean().item())
        scheduler.step()

        proj.eval(); classifier.eval(); vaccs = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                emb = proj(backbone(x))
                logits = classifier(emb)
                vaccs.append((logits.argmax(-1) == y).float().mean().item())
        val_acc = np.mean(vaccs) * 100
        print(f'  ep{epoch:02d} | train_loss {np.mean(losses):.4f} | '
              f'train_acc {np.mean(accs)*100:.1f}% | val_acc {val_acc:.1f}% | best {best_val:.1f}%')

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in proj.state_dict().items()}
            torch.save(best_state, PROJ_CLASSIFICATION_CKPT)
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f'  Early stopping at epoch {epoch} (no improvement for 20 epochs, best_val={best_val:.1f}%)')
                break

    proj.load_state_dict(best_state)
    print(f'proj_classification complete | best_val_acc={best_val:.1f}% | Saved -> {PROJ_CLASSIFICATION_CKPT}')
    return proj

                                                                
proj_episodic = ProjectionHead(backbone.embed_dim, CFG['proj_dim'], dropout=CFG['dropout_rate']).to(DEVICE)
if Path(PROJ_EPISODIC_CKPT).exists():
    proj_episodic.load_state_dict(torch.load(PROJ_EPISODIC_CKPT, map_location=DEVICE))
    print(f'Loaded existing proj_episodic checkpoint -> {PROJ_EPISODIC_CKPT}')
else:
    proj_episodic = train_episodic_projection()
proj_episodic.eval()
for p in proj_episodic.parameters(): p.requires_grad = False

proj_classification = ProjectionHead(backbone.embed_dim, CFG['proj_dim'], dropout=CFG['dropout_rate']).to(DEVICE)
if Path(PROJ_CLASSIFICATION_CKPT).exists():
    proj_classification.load_state_dict(torch.load(PROJ_CLASSIFICATION_CKPT, map_location=DEVICE))
    print(f'Loaded existing proj_classification checkpoint -> {PROJ_CLASSIFICATION_CKPT}')
else:
    proj_classification = train_classification_projection()
proj_classification.eval()
for p in proj_classification.parameters(): p.requires_grad = False

print('Both projection heads ready (frozen).')

# Cell 9 - 9 — Stage B: Meta-Train RelationNet's Relation Module (resumable).

class RelationModule(nn.Module):
    def __init__(self, embed_dim, hidden=128):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid())
    def forward(self, query_rep, proto_rep):
                                                                         
        x = torch.cat([query_rep, proto_rep], dim=-1)
        return self.g(x).squeeze(-1)                 

def meta_train_relationnet():
    n_way = CFG['n_way']
    relnet = RelationModule(CFG['proj_dim'], CFG['relationnet_hidden']).to(DEVICE)
    opt = torch.optim.Adam(relnet.parameters(), lr=CFG['relnet_meta_lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['relnet_meta_epochs'])
    best_val, best_state = -1.0, None
    patience_counter = 0
    print(f'Meta-training RelationNet module | {CFG["relnet_meta_epochs"]} epochs x '
          f'{CFG["relnet_meta_episodes_per_ep"]} episodes | grad_clip={CFG["meta_grad_clip"]}, '
          f'best-val checkpoint selection')
    for epoch in range(1, CFG['relnet_meta_epochs'] + 1):
        t_sampler = ImageEpisodeSampler(train_wrap, n_way, CFG['meta_k_shot'],
                                         CFG['meta_n_query'], CFG['relnet_meta_episodes_per_ep'],
                                         seed=2000 + epoch)
        v_sampler = ImageEpisodeSampler(val_wrap, n_way, 5,
                                         CFG['meta_n_query'], CFG['meta_n_ep_val'],
                                         seed=6000 + epoch)
        relnet.train(); losses, accs = [], []
        for si, sl, qi, ql in tqdm(t_sampler, desc=f'relationnet ep{epoch:02d}[tr]', leave=False):
            si, sl, qi, ql = si.to(DEVICE), sl.to(DEVICE), qi.to(DEVICE), ql.to(DEVICE)
            with torch.no_grad():
                s_emb = proj_episodic(backbone(si))
                q_emb = proj_episodic(backbone(qi))
            proto = torch.stack([s_emb[sl == c].mean(0) for c in range(n_way)])
            n_q = q_emb.size(0)
            q_rep = q_emb.unsqueeze(1).expand(-1, n_way, -1)
            p_rep = proto.unsqueeze(0).expand(n_q, -1, -1)
            scores = relnet(q_rep, p_rep)
            target = F.one_hot(ql, n_way).float()
            loss = F.mse_loss(scores, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(relnet.parameters(), CFG['meta_grad_clip'])
            opt.step()
            losses.append(loss.item())
            accs.append((scores.argmax(-1) == ql).float().mean().item())
        scheduler.step()

        relnet.eval(); vaccs = []
        with torch.no_grad():
            for si, sl, qi, ql in tqdm(v_sampler, desc=f'relationnet ep{epoch:02d}[val]', leave=False):
                si, sl, qi, ql = si.to(DEVICE), sl.to(DEVICE), qi.to(DEVICE), ql.to(DEVICE)
                s_emb = proj_episodic(backbone(si)); q_emb = proj_episodic(backbone(qi))
                proto = torch.stack([s_emb[sl == c].mean(0) for c in range(n_way)])
                n_q = q_emb.size(0)
                q_rep = q_emb.unsqueeze(1).expand(-1, n_way, -1)
                p_rep = proto.unsqueeze(0).expand(n_q, -1, -1)
                scores = relnet(q_rep, p_rep)
                vaccs.append((scores.argmax(-1) == ql).float().mean().item())
        val_acc = np.mean(vaccs) * 100
        print(f'  ep{epoch:02d} | train_loss {np.mean(losses):.4f} | '
              f'train_acc {np.mean(accs)*100:.1f}% | val_acc {val_acc:.1f}% | best {best_val:.1f}%')
        patience_counter = 0
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in relnet.state_dict().items()}
            torch.save(best_state, RELATIONNET_CKPT)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f'  Early stopping at epoch {epoch} (no improvement for 20 epochs, best_val={best_val:.1f}%)')
                break

    relnet.load_state_dict(best_state)
    print(f'RelationNet meta-training complete | best_val_acc={best_val:.1f}% | Saved -> {RELATIONNET_CKPT}')
    return relnet

relnet_module = RelationModule(CFG['proj_dim'], CFG['relationnet_hidden']).to(DEVICE)
if Path(RELATIONNET_CKPT).exists():
    relnet_module.load_state_dict(torch.load(RELATIONNET_CKPT, map_location=DEVICE))
    print(f'Loaded existing RelationNet checkpoint -> {RELATIONNET_CKPT}')
else:
    relnet_module = meta_train_relationnet()
relnet_module.eval()
for p in relnet_module.parameters(): p.requires_grad = False
print('RelationNet module ready (frozen).')

# Cell 10 - 9b — Meta-Train MAML Classifier Head (resumable).

class MAMLHead(nn.Module):
    def __init__(self, embed_dim, n_way):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_way, embed_dim))
        self.bias = nn.Parameter(torch.zeros(n_way))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    @staticmethod
    def forward_with_params(x, w, b):
        return F.linear(x, w, b)

    def adapt(self, s_emb, s_lbl, inner_lr, inner_steps, first_order=False):
        w, b = self.weight, self.bias
        for _ in range(inner_steps):
            logits = self.forward_with_params(s_emb, w, b)
            loss = F.cross_entropy(logits, s_lbl)
            grads = torch.autograd.grad(loss, [w, b], create_graph=not first_order)
            w = w - inner_lr * grads[0]
            b = b - inner_lr * grads[1]
        return w, b


def meta_train_maml():
    n_way = CFG['n_way']
    maml_head = MAMLHead(CFG['proj_dim'], n_way).to(DEVICE)
    meta_opt = torch.optim.Adam(maml_head.parameters(), lr=CFG['maml_meta_lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=CFG['maml_meta_epochs'])
    meta_batch = CFG['maml_meta_batch']
    best_val, best_state = -1.0, None
    print(f'Meta-training MAML head | {CFG["maml_meta_epochs"]} epochs x '
          f'{CFG["maml_meta_episodes_per_ep"]} episodes | inner_steps={CFG["maml_inner_steps"]} '
          f'inner_lr={CFG["maml_inner_lr"]} meta_batch={meta_batch} '
          f'first_order={CFG["maml_first_order"]}')
    for epoch in range(1, CFG['maml_meta_epochs'] + 1):
        t_sampler = ImageEpisodeSampler(train_wrap, n_way, CFG['meta_k_shot'],
                                         CFG['meta_n_query'], CFG['maml_meta_episodes_per_ep'],
                                         seed=3000 + epoch)
        v_sampler = ImageEpisodeSampler(val_wrap, n_way, 5,
                                         CFG['meta_n_query'], CFG['meta_n_ep_val'],
                                         seed=8000 + epoch)
        maml_head.train(); losses, accs = [], []
        meta_opt.zero_grad()
        for i, (si, sl, qi, ql) in enumerate(
                tqdm(t_sampler, desc=f'maml ep{epoch:02d}[tr]', leave=False)):
            si, sl, qi, ql = si.to(DEVICE), sl.to(DEVICE), qi.to(DEVICE), ql.to(DEVICE)
            with torch.no_grad():
                s_emb = proj_episodic(backbone(si)); q_emb = proj_episodic(backbone(qi))
            w, b = maml_head.adapt(s_emb, sl, CFG['maml_inner_lr'], CFG['maml_inner_steps'],
                                    first_order=CFG['maml_first_order'])
            q_logits = maml_head.forward_with_params(q_emb, w, b)
            outer_loss = F.cross_entropy(q_logits, ql) / meta_batch
            outer_loss.backward()
            losses.append(outer_loss.item() * meta_batch)
            accs.append((q_logits.argmax(-1) == ql).float().mean().item())
            if (i + 1) % meta_batch == 0:
                torch.nn.utils.clip_grad_norm_(maml_head.parameters(), CFG['meta_grad_clip'])
                meta_opt.step(); meta_opt.zero_grad()
        scheduler.step()

        maml_head.eval(); vaccs = []
        for si, sl, qi, ql in tqdm(v_sampler, desc=f'maml ep{epoch:02d}[val]', leave=False):
            si, sl, qi, ql = si.to(DEVICE), sl.to(DEVICE), qi.to(DEVICE), ql.to(DEVICE)
            with torch.no_grad():
                s_emb = proj_episodic(backbone(si)); q_emb = proj_episodic(backbone(qi))
            w, b = maml_head.adapt(s_emb, sl, CFG['maml_inner_lr'], CFG['maml_inner_steps_eval'],
                                    first_order=True)
            with torch.no_grad():
                q_logits = maml_head.forward_with_params(q_emb, w, b)
            vaccs.append((q_logits.argmax(-1) == ql).float().mean().item())
        val_acc = np.mean(vaccs) * 100
        print(f'  ep{epoch:02d} | train_loss {np.mean(losses):.4f} | '
              f'train_acc {np.mean(accs)*100:.1f}% | val_acc {val_acc:.1f}% | best {best_val:.1f}%')
        patience_counter = 0
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in maml_head.state_dict().items()}
            torch.save(best_state, MAML_CKPT)
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f'  Early stopping at epoch {epoch} (no improvement for 20 epochs, best_val={best_val:.1f}%)')
                break

    maml_head.load_state_dict(best_state)
    print(f'MAML meta-training complete | best_val_acc={best_val:.1f}% | Saved -> {MAML_CKPT}')
    return maml_head


maml_head = MAMLHead(CFG['proj_dim'], CFG['n_way']).to(DEVICE)
if Path(MAML_CKPT).exists():
    maml_head.load_state_dict(torch.load(MAML_CKPT, map_location=DEVICE))
    print(f'Loaded existing MAML checkpoint -> {MAML_CKPT}')
else:
    maml_head = meta_train_maml()
print('MAML head ready (meta-parameters frozen; still inner-loop-adapted per episode at eval).')

# Cell 11 - 9c — Meta-Train Transformer (FEAT-style) Prototype Adapter (resumable).

class SetTransformerAdapter(nn.Module):
    def __init__(self, dim, n_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, proto):
                                                                             
                                                                     
        return self.encoder(proto.unsqueeze(0)).squeeze(0)


def meta_train_transformer():
    n_way = CFG['n_way']
    adapter = SetTransformerAdapter(CFG['proj_dim'], CFG['transformer_n_heads'],
                                     CFG['transformer_ff_dim'], CFG['transformer_dropout']).to(DEVICE)
    opt = torch.optim.AdamW(adapter.parameters(), lr=CFG['transformer_meta_lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['transformer_meta_epochs'])
    best_val, best_state = -1.0, None
    print(f'Meta-training Transformer (FEAT) adapter | {CFG["transformer_meta_epochs"]} epochs x '
          f'{CFG["transformer_meta_episodes_per_ep"]} episodes | heads={CFG["transformer_n_heads"]}')
    for epoch in range(1, CFG['transformer_meta_epochs'] + 1):
        t_sampler = ImageEpisodeSampler(train_wrap, n_way, CFG['meta_k_shot'],
                                         CFG['meta_n_query'], CFG['transformer_meta_episodes_per_ep'],
                                         seed=4000 + epoch)
        v_sampler = ImageEpisodeSampler(val_wrap, n_way, 5,
                                         CFG['meta_n_query'], CFG['meta_n_ep_val'],
                                         seed=9000 + epoch)
        adapter.train(); losses, accs = [], []
        for si, sl, qi, ql in tqdm(t_sampler, desc=f'transformer ep{epoch:02d}[tr]', leave=False):
            si, sl, qi, ql = si.to(DEVICE), sl.to(DEVICE), qi.to(DEVICE), ql.to(DEVICE)
            with torch.no_grad():
                s_emb = proj_episodic(backbone(si)); q_emb = proj_episodic(backbone(qi))
            proto = torch.stack([s_emb[sl == c].mean(0) for c in range(n_way)])
            adapted_proto = adapter(proto)
            logits = -torch.cdist(q_emb.unsqueeze(0), adapted_proto.unsqueeze(0)).squeeze(0)
            loss = F.cross_entropy(logits, ql, label_smoothing=CFG['meta_label_smooth'])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), CFG['meta_grad_clip'])
            opt.step()
            losses.append(loss.item())
            accs.append((logits.argmax(-1) == ql).float().mean().item())
        scheduler.step()

        adapter.eval(); vaccs = []
        with torch.no_grad():
            for si, sl, qi, ql in tqdm(v_sampler, desc=f'transformer ep{epoch:02d}[val]', leave=False):
                si, sl, qi, ql = si.to(DEVICE), sl.to(DEVICE), qi.to(DEVICE), ql.to(DEVICE)
                s_emb = proj_episodic(backbone(si)); q_emb = proj_episodic(backbone(qi))
                proto = torch.stack([s_emb[sl == c].mean(0) for c in range(n_way)])
                adapted_proto = adapter(proto)
                logits = -torch.cdist(q_emb.unsqueeze(0), adapted_proto.unsqueeze(0)).squeeze(0)
                vaccs.append((logits.argmax(-1) == ql).float().mean().item())
        val_acc = np.mean(vaccs) * 100
        print(f'  ep{epoch:02d} | train_loss {np.mean(losses):.4f} | '
              f'train_acc {np.mean(accs)*100:.1f}% | val_acc {val_acc:.1f}% | best {best_val:.1f}%')
        patience_counter = 0
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
            torch.save(best_state, TRANSFORMER_CKPT)
        else:
             patience_counter += 1
             if patience_counter >= 20:
                print(f'  Early stopping at epoch {epoch} (no improvement for 20 epochs, best_val={best_val:.1f}%)')
                break


    adapter.load_state_dict(best_state)
    print(f'Transformer adapter meta-training complete | best_val_acc={best_val:.1f}% | Saved -> {TRANSFORMER_CKPT}')
    return adapter


transformer_adapter = SetTransformerAdapter(
    CFG['proj_dim'], CFG['transformer_n_heads'], CFG['transformer_ff_dim'],
    CFG['transformer_dropout']).to(DEVICE)
if Path(TRANSFORMER_CKPT).exists():
    transformer_adapter.load_state_dict(torch.load(TRANSFORMER_CKPT, map_location=DEVICE))
    print(f'Loaded existing Transformer checkpoint -> {TRANSFORMER_CKPT}')
else:
    transformer_adapter = meta_train_transformer()
transformer_adapter.eval()
for p in transformer_adapter.parameters(): p.requires_grad = False
print('Transformer (FEAT) adapter ready (frozen).')

# Cell 12 - 10 — Precompute & Cache Test-Set Embeddings.

@torch.no_grad()
def cache_embeddings(dataset, proj_head, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embs, labels = [], []
    for x, y in tqdm(loader, desc='caching embeddings', leave=False):
        x = x.to(DEVICE)
        feat = backbone(x)
        emb = proj_head(feat) if proj_head is not None else feat
        embs.append(emb.cpu()); labels.append(y)
    return torch.cat(embs, 0), torch.cat(labels, 0)

test_emb_episodic, test_labels = cache_embeddings(test_wrap, proj_episodic)
test_emb_classification, _     = cache_embeddings(test_wrap, proj_classification)
print(f'Cached test embeddings | episodic: {test_emb_episodic.shape} | '
      f'classification: {test_emb_classification.shape}')

class EmbeddingEpisodeSampler:
    def __init__(self, embeddings, labels, n_way, k_shot, n_query, n_episodes, seed):
        self.emb = embeddings; self.lbl = labels.numpy()
        self.n_way = n_way; self.k_shot = k_shot; self.n_query = n_query
        self.n_episodes = n_episodes
        self.rng = random.Random(seed)
        self.class_indices = {}
        for i, l in enumerate(self.lbl):
            self.class_indices.setdefault(int(l), []).append(i)
        self.available = [k for k, v in self.class_indices.items()
                           if len(v) >= k_shot + n_query]
        assert len(self.available) >= n_way, f'Need >= {n_way} classes, got {len(self.available)}'
    def __len__(self): return self.n_episodes
    def sample_episode(self):
        classes = self.rng.sample(self.available, self.n_way)
        si, sl, qi, ql = [], [], [], []
        for local, cls in enumerate(classes):
            chosen = self.rng.sample(self.class_indices[cls], self.k_shot + self.n_query)
            for i in chosen[:self.k_shot]:
                si.append(self.emb[i]); sl.append(local)
            for i in chosen[self.k_shot:]:
                qi.append(self.emb[i]); ql.append(local)
        return (torch.stack(si), torch.tensor(sl), torch.stack(qi), torch.tensor(ql))
    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self.sample_episode()

print('EmbeddingEpisodeSampler ready — eval loop now runs on cached tensors.')

class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(net.children())[:-1])           
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()
        self.embed_dim = 2048
    def forward(self, x):
        with torch.no_grad():
            f = self.features(x)
        return f.flatten(1)

resnet50_backbone = ResNet50Backbone().to(DEVICE)
resnet50_backbone.eval()

@torch.no_grad()
def cache_embeddings_with_backbone(dataset, bb, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embs, labels = [], []
    for x, y in tqdm(loader, desc='caching resnet50 embeddings', leave=False):
        x = x.to(DEVICE)
        embs.append(bb(x).cpu()); labels.append(y)
    return torch.cat(embs, 0), torch.cat(labels, 0)

test_emb_resnet50, _ = cache_embeddings_with_backbone(test_wrap, resnet50_backbone)
print(f'Cached ResNet50 test embeddings: {test_emb_resnet50.shape}')

# Cell 13 - 11 — Evaluation-Time Classification Heads.

class MatchingNetHead(nn.Module):
    def forward(self, s_emb, s_lbl, q_emb, n_way):
        s = F.normalize(s_emb, dim=-1); q = F.normalize(q_emb, dim=-1)
        sims = q @ s.t()
        attn = F.softmax(sims * 10.0, dim=-1)
        one_hot = F.one_hot(s_lbl, n_way).float().to(q.device)
        probs = attn @ one_hot
        return torch.log(probs.clamp_min(1e-8))

class BaselinePlusPlusHead(nn.Module):
    def __init__(self, embed_dim, n_way):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_way, embed_dim) * 0.01)
        self.scale = 10.0
    def fit_support(self, s_emb, s_lbl, steps, lr, device):
        opt = torch.optim.Adam([self.weight], lr=lr)
        for _ in range(steps):
            w = F.normalize(self.weight, dim=-1); x = F.normalize(s_emb, dim=-1)
            loss = F.cross_entropy(self.scale * (x @ w.t()), s_lbl)
            opt.zero_grad(); loss.backward(); opt.step()
    def forward(self, q_emb):
        w = F.normalize(self.weight, dim=-1); x = F.normalize(q_emb, dim=-1)
        return self.scale * (x @ w.t())

class BaselineHead(nn.Module):
    def __init__(self, embed_dim, n_way):
        super().__init__()
        self.linear = nn.Linear(embed_dim, n_way)
    def fit_support(self, s_emb, s_lbl, steps, lr, device):
        opt = torch.optim.Adam(self.linear.parameters(), lr=lr)
        for _ in range(steps):
            loss = F.cross_entropy(self.linear(s_emb), s_lbl)
            opt.zero_grad(); loss.backward(); opt.step()
    def forward(self, q_emb):
        return self.linear(q_emb)

def ep_metrics(logits, labels_np):
    preds = logits.argmax(-1).detach().cpu().numpy()
    acc = (preds == labels_np).mean()
    f1  = f1_score(labels_np, preds, average='macro', zero_division=0)
    bacc = balanced_accuracy_score(labels_np, preds)
    return acc, f1, bacc

def run_episode(variant, s_emb, s_lbl, q_emb, q_lbl, n_way, device):
    s_emb, q_emb = s_emb.to(device), q_emb.to(device)
    s_lbl_t, q_lbl_np = s_lbl.to(device), q_lbl.numpy()

    if variant == 'matchingnet':
        head = MatchingNetHead().to(device)
        logits = head(s_emb, s_lbl_t, q_emb, n_way)

    elif variant == 'relationnet':
        n_q = q_emb.size(0)
        proto = torch.stack([s_emb[s_lbl_t == c].mean(0) for c in range(n_way)])
        q_rep = q_emb.unsqueeze(1).expand(-1, n_way, -1)
        p_rep = proto.unsqueeze(0).expand(n_q, -1, -1)
        with torch.no_grad():
            logits = relnet_module(q_rep, p_rep)                                   

    elif variant == 'baseline++':
        head = BaselinePlusPlusHead(CFG['proj_dim'], n_way).to(device)
        head.fit_support(s_emb, s_lbl_t, steps=CFG['baselinepp_ft_steps'],
                          lr=CFG['baselinepp_ft_lr'], device=device)
        with torch.no_grad():
            logits = head(q_emb)

    elif variant == 'baseline':
        head = BaselineHead(CFG['proj_dim'], n_way).to(device)
        head.fit_support(s_emb, s_lbl_t, steps=CFG['baseline_ft_steps'],
                          lr=CFG['baseline_ft_lr'], device=device)
        with torch.no_grad():
            logits = head(q_emb)

    elif variant == 'maml':
                                                                           
                                                                        
                                                                         
        w, b = maml_head.adapt(s_emb, s_lbl_t, CFG['maml_inner_lr'],
                                CFG['maml_inner_steps_eval'], first_order=True)
        with torch.no_grad():
            logits = maml_head.forward_with_params(q_emb, w, b)

    elif variant == 'transformer':
                                                                         
                                                                              
        proto = torch.stack([s_emb[s_lbl_t == c].mean(0) for c in range(n_way)])
        with torch.no_grad():
            adapted_proto = transformer_adapter(proto)
            logits = -torch.cdist(q_emb.unsqueeze(0), adapted_proto.unsqueeze(0)).squeeze(0)

    elif variant == 'resnet50':
                                                                          
                                                                        
                                                                   
        head = BaselineHead(resnet50_backbone.embed_dim, n_way).to(device)
        head.fit_support(s_emb, s_lbl_t, steps=CFG['resnet50_ft_steps'],
                          lr=CFG['resnet50_ft_lr'], device=device)
        with torch.no_grad():
            logits = head(q_emb)
    else:
        raise ValueError(variant)

    return ep_metrics(logits, q_lbl_np)

print('Evaluation heads ready')

# Cell 14 - 12 — Resumable Main Evaluation Loop.

VARIANTS = ['matchingnet', 'relationnet', 'baseline++', 'baseline',
            'maml', 'transformer', 'resnet50']

                                                
EMBEDDING_FOR = {
    'matchingnet': test_emb_episodic,
    'relationnet': test_emb_episodic,
    'baseline++':  test_emb_classification,
    'baseline':    test_emb_classification,
    'maml':        test_emb_episodic,                                       
    'transformer': test_emb_episodic,                                       
    'resnet50':    test_emb_resnet50,                                              
}

def load_progress():
    if Path(PROGRESS_JSON).exists():
        with open(PROGRESS_JSON) as f:
            return json.load(f)
    return {'completed': []}

def save_progress(progress):
    with open(PROGRESS_JSON, 'w') as f:
        json.dump(progress, f)

def append_rows(rows):
    df = pd.DataFrame(rows)
    header = not Path(BASE_EP_CSV).exists()
    df.to_csv(BASE_EP_CSV, mode='a', header=header, index=False)

progress = load_progress()
done_set = set(progress['completed'])
print(f'Resuming — {len(done_set)} (variant,k_shot,seed) triples already complete')

total_combos = len(VARIANTS) * len(CFG['k_shots']) * len(CFG['seeds'])
pbar = tqdm(total=total_combos, desc='Baseline comparison (variant x k_shot x seed)')
pbar.update(len(done_set))

for variant in VARIANTS:
    emb = EMBEDDING_FOR[variant]
    for k_shot in CFG['k_shots']:
        for seed in CFG['seeds']:
            key = f'{variant}|{k_shot}|{seed}'
            if key in done_set:
                continue
            set_seed(seed)
            sampler = EmbeddingEpisodeSampler(emb, test_labels, CFG['n_way'], k_shot,
                                               CFG['n_query'], CFG['n_episodes'], seed=seed)
            rows = []
            for ep_idx, (si, sl, qi, ql) in enumerate(
                    tqdm(sampler, desc=f'{variant} k={k_shot} seed={seed}', leave=False)):
                acc, f1, bacc = run_episode(variant, si, sl, qi, ql, CFG['n_way'], DEVICE)
                rows.append({'variant': variant, 'k_shot': k_shot, 'seed': seed,
                             'episode_idx': ep_idx, 'accuracy': acc,
                             'macro_f1': f1, 'balanced_acc': bacc})
                if (ep_idx + 1) % 50 == 0:
                    append_rows(rows); rows = []
            if rows:
                append_rows(rows)
            done_set.add(key)
            progress['completed'] = sorted(done_set)
            save_progress(progress)
            pbar.update(1)
            print(f'  done: {key}')

pbar.close()
print(f'\nAll combinations complete. Episode-level results -> {BASE_EP_CSV}')

# Cell 15 - 13 — Aggregate Summary, Significance, and 9-Row Comparison Table.

ep_df = pd.read_csv(BASE_EP_CSV)

summary_rows = []
for variant in VARIANTS:
    for k_shot in CFG['k_shots']:
        sub = ep_df[(ep_df.variant == variant) & (ep_df.k_shot == k_shot)]
        n = len(sub)
        mean_acc = sub['accuracy'].mean()
        ci95 = 1.96 * sub['accuracy'].std(ddof=1) / np.sqrt(n)
        summary_rows.append({'variant': variant, 'k_shot': k_shot, 'n_episodes': n,
                              'mean_accuracy': mean_acc, 'ci95': ci95,
                              'mean_macro_f1': sub['macro_f1'].mean(),
                              'mean_balanced_acc': sub['balanced_acc'].mean()})
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(BASE_SUM_CSV, index=False)
print(summary_df)

ladder_path = find_output_file('ladder_test_results.csv')
sig_rows = []
if ladder_path is not None:
    ladder_df = pd.read_csv(ladder_path)
    for k_shot in CFG['k_shots']:
        ssl_full_acc = ladder_df[(ladder_df.variant == 'ssl_full') &
                                  (ladder_df.k_shot == k_shot)]['accuracy'].values
        for variant in VARIANTS:
            base_acc = ep_df[(ep_df.variant == variant) & (ep_df.k_shot == k_shot)]['accuracy'].values
            if len(ssl_full_acc) == 0 or len(base_acc) == 0: continue
            u, p = sp_stats.mannwhitneyu(ssl_full_acc, base_acc, alternative='greater')
            sig_rows.append({'k_shot': k_shot, 'comparison': f'ssl_full > {variant}',
                              'mann_whitney_u': u, 'p_value': p})
    pd.DataFrame(sig_rows).to_csv(BASE_SIG_CSV, index=False)
    print(pd.DataFrame(sig_rows))
else:
    print('WARNING: ladder_test_results.csv not found anywhere under Outputs/ — '
          'significance tests skipped. Confirm Part 3c has been run and synced to Drive.')

import matplotlib.pyplot as plt

rows_for_table = []
indomain_path = find_output_file('indomain_summary.csv')
if indomain_path is not None:
    idf = pd.read_csv(indomain_path)
    for k_shot in CFG['k_shots']:
        r = idf[(idf.n_way == CFG['n_way']) & (idf.k_shot == k_shot)].iloc[0]
        rows_for_table.append({'Model': 'SwinProtoNet (ours)', 'k_shot': k_shot,
                                'Accuracy': r['mean_accuracy'], 'CI95': r['ci95'],
                                'Macro-F1': r['mean_macro_f1']})
else:
    print('WARNING: indomain_summary.csv not found anywhere under Outputs/ — '
          '"SwinProtoNet (ours)" row will be missing from the table.')

ladder_summary_path = find_output_file('ladder_summary.csv')
if ladder_summary_path is not None:
    ldf = pd.read_csv(ladder_summary_path)
    for k_shot in CFG['k_shots']:
        rows_imgnet = ldf[(ldf.variant == 'imagenet') & (ldf.k_shot == k_shot)]
        if len(rows_imgnet):
            r = rows_imgnet.iloc[0]
            rows_for_table.append({'Model': 'ImageNet-init', 'k_shot': k_shot,
                                    'Accuracy': r['mean_accuracy'], 'CI95': r['ci95'],
                                    'Macro-F1': np.nan})
else:
    print('WARNING: ladder_summary.csv not found anywhere under Outputs/ — '
          'ImageNet-init row will be missing.')

for _, r in summary_df.iterrows():
    rows_for_table.append({'Model': r['variant'], 'k_shot': r['k_shot'],
                            'Accuracy': r['mean_accuracy'], 'CI95': r['ci95'],
                            'Macro-F1': r['mean_macro_f1']})

table_df = pd.DataFrame(rows_for_table).sort_values(['k_shot','Accuracy'], ascending=[True, False])
print(table_df.to_string(index=False))
table_df.to_csv(f"{CFG['output_dir']}/baseline_comparison_table.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, k_shot in zip(axes, CFG['k_shots']):
    sub = table_df[table_df.k_shot == k_shot].sort_values('Accuracy', ascending=True)
    colors = ['darkorange' if m == 'SwinProtoNet (ours)' else 'steelblue' for m in sub['Model']]
    ax.barh(sub['Model'], sub['Accuracy']*100, xerr=sub['CI95']*100, color=colors)
    ax.set(title=f'K={k_shot}', xlabel='Accuracy (%)', xlim=(0,100))
    ax.grid(alpha=0.3, axis='x')
fig.suptitle('SwinProtoNet vs. Literature-Standard Few-Shot Baselines (In-Domain, N=8)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CFG['output_dir']}/baseline_comparison_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print(f'\nSaved -> {CFG["output_dir"]}/baseline_comparison_plot.png')
print('Part 3h (v2, corrected) complete.')

# Cell 16 - Final execution/provenance check.

                                         
                                                                                
                                                                   

from datetime import datetime, timezone
from pathlib import Path
import json, hashlib, numpy as np, pandas as pd

run_manifest = {
    "run_id_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    "notebook": "part3h_baseline_model_comparison_q1.ipynb",
    "source_script": "part3h_baseline_model_comparison (2).py",
    "seed_config": CFG["seeds"],
    "n_way": CFG["n_way"],
    "k_shots": CFG["k_shots"],
    "n_query": CFG["n_query"],
    "episodes_per_seed": CFG["n_episodes"],
    "projection_meta_epochs": CFG["meta_epochs"],
    "relationnet_meta_epochs": CFG["relnet_meta_epochs"],
    "test_split_sha256": hashlib.sha256(
        json.dumps(sorted(test_idx)).encode()
    ).hexdigest(),
    "artifacts": {
        "episode_metrics": BASE_EP_CSV,
        "summary": BASE_SUM_CSV,
        "significance": BASE_SIG_CSV,
        "comparison_table": f'{CFG["output_dir"]}/baseline_comparison_table.csv',
        "plot": f'{CFG["output_dir"]}/baseline_comparison_plot.png',
        "proj_episodic": PROJ_EPISODIC_CKPT,
        "proj_classification": PROJ_CLASSIFICATION_CKPT,
        "relationnet": RELATIONNET_CKPT,
        "maml": MAML_CKPT,
        "transformer": TRANSFORMER_CKPT,
    },
}

manifest_path = Path(CFG["output_dir"]) / "baseline_comparison_run_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(run_manifest, f, indent=2)

print("Run manifest saved:", manifest_path)

                                              
if Path(BASE_EP_CSV).exists():
    eval_df = pd.read_csv(BASE_EP_CSV)
    print("\nPrediction/metric sanity checks:")
    for variant in sorted(eval_df["variant"].unique()):
        for k in sorted(eval_df["k_shot"].unique()):
            sub = eval_df[(eval_df.variant == variant) & (eval_df.k_shot == k)]
            if len(sub) == 0:
                continue
            vals = sub["accuracy"].to_numpy()
            print(
                f"{variant:12s} K={k}: n={len(vals):4d}, "
                f"mean={vals.mean():.4f}, std={vals.std(ddof=1):.4f}, "
                f"min={vals.min():.4f}, max={vals.max():.4f}"
            )
            if np.allclose(vals, vals[0]):
                raise RuntimeError(
                    f"Degenerate constant accuracy detected for {variant}, K={k}."
                )
else:
    print("Episode CSV not found yet; run the evaluation cells first.")
