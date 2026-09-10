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
for pkg in ['timm==0.9.16', 'albumentations>=1.3.0', 'einops', 'tqdm', 'scikit-learn', 'scipy']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
print('packages ready')

# Cell 2 - Import dependencies, set seeds, and configure evaluation.
import random, json, hashlib, csv, shutil, warnings, time
import numpy as np
from pathlib import Path
from PIL import Image
import torch.nn as nn, torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, balanced_accuracy_score
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda')

CFG = {
    'output_dir'    : '/content/drive/MyDrive/SSL_FYDP/Outputs',
    'labeled_root'  : '/content/labeled_data',
    'kvasir_v2_dir' : '/content/drive/MyDrive/SSL_FYDP/kvasir-dataset-v2',
    'ssl_ckpt'      : '/content/drive/MyDrive/SSL_FYDP/Outputs/swin_ssl_pretrained.pth',
    'proto_ckpt'    : '/content/drive/MyDrive/SSL_FYDP/Outputs/protonet_best.pth',
    'split_file'    : '/content/drive/MyDrive/SSL_FYDP/Outputs/split_indices.json',
    'backbone'      : 'swin_tiny_patch4_window7_224',
    'embed_dim'     : 768,
    'img_size'      : 224,
    'dropout_rate'  : 0.1,
    'temperature'   : 0.5,
    'mc_passes'     : 20,
    'max_per_class' : 300,
                                                                            
    'n_way'         : 8,
    'k_shot_train'  : 5,
    'n_query'       : 15,
    'n_ep_train'    : 100,
    'n_ep_val'      : 30,
    'train_epochs'  : 60,
    'projector_lr'  : 1e-4,
    'weight_decay'  : 0.05,
    'label_smooth'  : 0.1,
    'grad_clip'     : 5.0,
                                                                             
                                                                
    'eval_n_way'    : 8,
    'eval_k_shots'  : [1, 5],
    'eval_n_query'  : 15,
    'eval_seeds'    : [42, 43, 44, 45, 46],
    'eval_episodes' : 600,
}
SIZE = CFG['img_size']
os.makedirs(CFG['output_dir'], exist_ok=True)
print('Config ready | n_way:', CFG['n_way'], '| eval_k_shots:', CFG['eval_k_shots'])

\
\
\
\

# Cell 3 - Rebuild the labeled dataset and load the held-out split.
def find_kv2_root(base):
    base = Path(base)
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if len(subdirs) >= 6: return base
    for sub in subdirs:
        if len([d for d in sub.iterdir() if d.is_dir()]) >= 6: return sub
    raise RuntimeError(f'Could not locate Kvasir v2 class folders under {base}')

kv2_root = find_kv2_root(CFG['kvasir_v2_dir'])
rng_dataset = random.Random(SEED)
if Path(CFG['labeled_root']).exists(): shutil.rmtree(CFG['labeled_root'])
MAX = CFG['max_per_class']
MIN_NEEDED = CFG['k_shot_train'] + CFG['n_query'] + 10
for cls_dir in sorted(kv2_root.iterdir()):
    if not cls_dir.is_dir(): continue
    imgs = sorted(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.png')))
    if len(imgs) < MIN_NEEDED: continue
    rng_dataset.shuffle(imgs)
    dst = Path(CFG['labeled_root']) / cls_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    for src in imgs[:MAX]: shutil.copy(src, dst / src.name)

class LabeledPolyp(torch.utils.data.Dataset):
    def __init__(self, root_dirs):
        self.samples, self.classes = [], []
        for root in root_dirs:
            for d in sorted(Path(root).iterdir()):
                if d.is_dir() and d.name not in self.classes: self.classes.append(d.name)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for root in root_dirs:
            for d in sorted(Path(root).iterdir()):
                if d.is_dir():
                    lbl = self.class_to_idx[d.name]
                    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                        for p in sorted(d.glob(ext)): self.samples.append((str(p), lbl))
    def __len__(self): return len(self.samples)

full_ds = LabeledPolyp([CFG['labeled_root']])
with open(CFG['split_file']) as f: split_data = json.load(f)
assert split_data['seed'] == SEED
assert len(full_ds) == split_data['n_total'], 'Rebuilt dataset size mismatch vs. split_indices.json'
train_idx, val_idx, test_idx = split_data['train'], split_data['val'], split_data['test']
train_samples = [full_ds.samples[i] for i in train_idx]
val_samples   = [full_ds.samples[i] for i in val_idx]
test_samples  = [full_ds.samples[i] for i in test_idx]
print(f'Split verified | train:{len(train_samples)} val:{len(val_samples)} test:{len(test_samples)}')

# Cell 4 - Define image augmentations and the episodic sampler.
train_tfm = A.Compose([
    A.Resize(SIZE, SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.7),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(),
])
val_tfm = A.Compose([
    A.Resize(SIZE, SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(),
])

def load_image(path, tfm):
    img = np.array(Image.open(path).convert('RGB'))
    return tfm(image=img)['image']

class EpisodeSampler:
    def __init__(self, samples, n_way, k_shot, n_query, n_episodes, tfm, seed):
        self.samples, self.n_way, self.k_shot = samples, n_way, k_shot
        self.n_query, self.n_episodes, self.tfm = n_query, n_episodes, tfm
        self.rng = random.Random(seed)
        self.class_indices = {}
        for i, (_, lbl) in enumerate(samples):
            self.class_indices.setdefault(lbl, []).append(i)
        self.available = [k for k, v in self.class_indices.items() if len(v) >= k_shot + n_query]
        assert len(self.available) >= n_way
    def __len__(self): return self.n_episodes
    def sample_episode(self):
        classes = self.rng.sample(self.available, self.n_way)
        si, sl, qi, ql = [], [], [], []
        for local, cls in enumerate(classes):
            chosen = self.rng.sample(self.class_indices[cls], self.k_shot + self.n_query)
            for i in chosen[:self.k_shot]:
                si.append(load_image(self.samples[i][0], self.tfm)); sl.append(local)
            for i in chosen[self.k_shot:]:
                qi.append(load_image(self.samples[i][0], self.tfm)); ql.append(local)
        return (torch.stack(si), torch.tensor(sl), torch.stack(qi), torch.tensor(ql))
    def __iter__(self):
        for _ in range(self.n_episodes): yield self.sample_episode()

print('EpisodeSampler ready')

\
\
\
\
\
\
\
\

# Cell 5 - Define the SwinProtoNet ablation model variants.
class SwinProtoNetAblation(nn.Module):
    def __init__(self, variant):
        super().__init__()
        assert variant in ('random_init', 'ssl_lp')
        self.variant = variant
        self.encoder = timm.create_model(
            CFG['backbone'], pretrained=False,                                           
                                                                                        
                                                                                       
                                                                
            num_classes=0, img_size=CFG['img_size'])

        if variant == 'ssl_lp':
            ckpt = torch.load(CFG['ssl_ckpt'], map_location='cpu', weights_only=False)
            missing, unexpected = self.encoder.load_state_dict(ckpt['encoder_state'], strict=False)
            print(f'  [ssl_lp] SSL weights loaded | missing:{len(missing)} unexpected:{len(unexpected)}')
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        else:
            print('  [random_init] TRUE random init — no ImageNet, no DINO-SSL checkpoint loaded')
            for p in self.encoder.parameters():
                p.requires_grad = True

        actual_dim = self.encoder.num_features
        D = CFG['embed_dim']
        self.drop = nn.Dropout(p=CFG['dropout_rate'])
        self.proj = nn.Sequential(
            nn.Linear(actual_dim, D * 2), nn.LayerNorm(D * 2), nn.GELU(),
            nn.Dropout(p=CFG['dropout_rate']),
            nn.Linear(D * 2, D), nn.LayerNorm(D),
        )
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        print(f'  variant={variant} | trainable params: {trainable:.1f}M')

    def encode(self, x):
        if self.variant == 'ssl_lp':
            with torch.no_grad():
                f = self.encoder.forward_features(x)
        else:
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

print('SwinProtoNetAblation defined')

# Cell 6 - Define the resumable training loop for the ablation arms.
def _rng_state_dict():
    return {
        'python'     : random.getstate(),
        'numpy'      : np.random.get_state(),
        'torch'      : torch.get_rng_state(),
        'torch_cuda' : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def _load_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])

def _atomic_torch_save(obj, final_path):
\
\
                                                     
    tmp_path = f'{final_path}.tmp'
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, final_path)
        return True
    except OSError as e:
        print(f'  SAVE FAILED for {final_path}: {e}')
        if Path(tmp_path).exists():
            try: os.remove(tmp_path)
            except OSError: pass
        return False


def train_arm(variant, patience=20, min_delta=1e-4):
\
\
\
\
\
\
\
\
\
\
\
\
       
    print(f'\n{"="*62}\n  Training ablation arm: {variant}\n{"="*62}')

    RESUME_PATH = f"{CFG['output_dir']}/ablation_{variant}_RESUME.pth"
    BEST_PATH   = f"{CFG['output_dir']}/ablation_{variant}_best.pth"
    LOG_PATH    = f"{CFG['output_dir']}/ablation_{variant}_train_log.csv"

    model = SwinProtoNetAblation(variant).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=CFG['projector_lr'], weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['train_epochs'])
    scaler = torch.amp.GradScaler('cuda')

    start_epoch, best_val, history, no_improve_epochs = 1, -1., [], 0

    if Path(RESUME_PATH).exists():
        ck = torch.load(RESUME_PATH, map_location='cpu', weights_only=False)
        if ck.get('variant') != variant or ck.get('total_epochs') != CFG['train_epochs']:
            print(f'  RESUME file found but variant/total_epochs mismatch — '
                  f'refusing to resume. Delete {RESUME_PATH} to start fresh.')
            raise ValueError('Resume checkpoint mismatch')
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optimizer_state'])
        scheduler.load_state_dict(ck['scheduler_state'])
        scaler.load_state_dict(ck['scaler_state'])
        _load_rng_state(ck['rng_state'])
        start_epoch = ck['epoch'] + 1
        best_val = ck['best_val']
        history = ck['history']
        no_improve_epochs = ck['no_improve_epochs']
        print(f'  RESUMING [{variant}] from epoch {start_epoch}/{CFG["train_epochs"]} '
              f'(best_val={best_val:.2f}%, no_improve={no_improve_epochs}/{patience})')

                                                                                 
                                                                                
                                                                                
                                                                                  
                                                                       
        if no_improve_epochs >= patience:
            print(f'  [{variant}] RESUMED state already at/over patience '
                  f'({no_improve_epochs}/{patience}) -- stopping now WITHOUT running '
                  f'another epoch.')
            model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=False))
            print(f'  [{variant}] final model = best-val checkpoint -> {BEST_PATH}')
            return model
    else:
        print(f'  Fresh start [{variant}]')
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_acc', 'no_improve_epochs'])

    if start_epoch > CFG['train_epochs']:
        print(f'  [{variant}] already completed {CFG["train_epochs"]} epochs per resume file.')
        model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=False))
        return model

    stopped_early = False
    for epoch in range(start_epoch, CFG['train_epochs'] + 1):
        t_sampler = EpisodeSampler(train_samples, CFG['n_way'], CFG['k_shot_train'],
                                    CFG['n_query'], CFG['n_ep_train'], train_tfm, seed=1000 + epoch)
        v_sampler = EpisodeSampler(val_samples, CFG['n_way'], 5,
                                    CFG['n_query'], CFG['n_ep_val'], val_tfm, seed=5000 + epoch)
        model.train(); tl, ta = [], []
        for si, sl, qi, ql in t_sampler:
            si, sl, qi, ql = si.to(DEVICE), sl.to(DEVICE), qi.to(DEVICE), ql.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(si, sl, qi, CFG['n_way']) / CFG['temperature']
                loss = F.cross_entropy(logits, ql, label_smoothing=CFG['label_smooth'])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, CFG['grad_clip'])
            scaler.step(optimizer); scaler.update()
            tl.append(loss.item()); ta.append((logits.argmax(-1) == ql).float().mean().item())
        scheduler.step()

        model.eval(); va = []
        with torch.no_grad():
            for si, sl, qi, ql in v_sampler:
                si, sl, qi, ql = si.to(DEVICE), sl.to(DEVICE), qi.to(DEVICE), ql.to(DEVICE)
                logits = model(si, sl, qi, CFG['n_way'])
                va.append((logits.argmax(-1) == ql).float().mean().item())

        t_l, t_a, v_a = np.mean(tl), np.mean(ta) * 100, np.mean(va) * 100
        history.append({'epoch': epoch, 'train_loss': t_l, 'train_acc': t_a, 'val_acc': v_a})

        if v_a > best_val + min_delta:
            best_val = v_a
            no_improve_epochs = 0
            _atomic_torch_save(model.state_dict(), BEST_PATH)
        else:
            no_improve_epochs += 1

        with open(LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, t_l, t_a, v_a, no_improve_epochs])

        resume_payload = {
            'variant': variant, 'epoch': epoch, 'total_epochs': CFG['train_epochs'],
            'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(), 'scaler_state': scaler.state_dict(),
            'rng_state': _rng_state_dict(), 'best_val': best_val, 'history': history,
            'no_improve_epochs': no_improve_epochs,
        }
        save_ok = _atomic_torch_save(resume_payload, RESUME_PATH)

        if epoch % 5 == 0 or epoch <= 3 or no_improve_epochs == 0:
            print(f'  [{variant}] ep{epoch:03d} | loss {t_l:.4f} | train {t_a:.1f}% | '
                  f'val {v_a:.1f}% | best {best_val:.1f}% | no_improve {no_improve_epochs}/{patience}'
                  f'{"" if save_ok else " | RESUME SAVE FAILED"}')

        if no_improve_epochs >= patience:
            print(f'  [{variant}] EARLY STOP at epoch {epoch} — val_acc unchanged '
                  f'(within {min_delta}) for {patience} consecutive epochs. '
                  f'best_val={best_val:.2f}% (epoch {epoch - patience}).')
            stopped_early = True
            break

    if not stopped_early:
        print(f'  [{variant}] completed all {CFG["train_epochs"]} epochs | best_val={best_val:.2f}%')

    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=False))
    print(f'  [{variant}] final model = best-val checkpoint -> {BEST_PATH}')
    return model

print('train_arm() ready — resumable + early stopping (patience=20 epochs)')

\
\
\
\
\

# Cell 7 - Define the resumable evaluation loop and rebuild summaries.
def _sample_episode_eval(samples, class_indices, n_way, k_shot, n_query,
                          eval_n_way, seed, episode_idx, tfm):
\
\
\
                                                                         
    ep_seed = seed * 1_000_000 + eval_n_way * 100_000 + k_shot * 10_000 + episode_idx
    rng = random.Random(ep_seed)
    classes = rng.sample(list(class_indices.keys()), n_way)
    si, sl, qi, ql = [], [], [], []
    for local, cls in enumerate(classes):
        chosen = rng.sample(class_indices[cls], k_shot + n_query)
        for i in chosen[:k_shot]:
            si.append(load_image(samples[i][0], tfm)); sl.append(local)
        for i in chosen[k_shot:]:
            qi.append(load_image(samples[i][0], tfm)); ql.append(local)
    return (torch.stack(si), torch.tensor(sl), torch.stack(qi), torch.tensor(ql))


def _completed_eval_episodes(ep_csv, variant):
\
\
                                                                 
    done = set()
    if not Path(ep_csv).exists():
        return done
    with open(ep_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row['variant'] != variant:
                continue
            done.add((int(row['k_shot']), int(row['seed']), int(row['episode_idx'])))
    return done


def evaluate_arm(model, variant):
\
\
                                                                               
    ep_csv = f"{CFG['output_dir']}/ablation_indomain_episode_metrics.csv"
    write_header = not Path(ep_csv).exists()
    model.eval()

    class_indices = {}
    for i, (_, lbl) in enumerate(test_samples):
        class_indices.setdefault(lbl, []).append(i)

    done = _completed_eval_episodes(ep_csv, variant)
    if done:
        print(f'  [{variant}] {len(done)} episode(s) already logged on disk -- resuming, skipping those.')

    with open(ep_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['variant', 'n_way', 'k_shot', 'seed', 'episode_idx',
                              'accuracy', 'macro_f1', 'balanced_acc'])
        for k_shot in CFG['eval_k_shots']:
            for seed in CFG['eval_seeds']:
                n_skipped = 0
                for episode_idx in range(CFG['eval_episodes']):
                    if (k_shot, seed, episode_idx) in done:
                        n_skipped += 1
                        continue
                    si, sl, qi, ql = _sample_episode_eval(
                        test_samples, class_indices, CFG['eval_n_way'], k_shot,
                        CFG['eval_n_query'], CFG['eval_n_way'], seed, episode_idx, val_tfm)
                    si, sl, qi = si.to(DEVICE), sl.to(DEVICE), qi.to(DEVICE)
                    with torch.no_grad():
                        logits = model(si, sl, qi, CFG['eval_n_way'])
                    y_true = ql.numpy(); y_pred = logits.argmax(-1).cpu().numpy()
                    acc = float((y_pred == y_true).mean())
                    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
                    bacc = balanced_accuracy_score(y_true, y_pred)
                    writer.writerow([variant, CFG['eval_n_way'], k_shot, seed, episode_idx, acc, f1, bacc])
                    f.flush(); os.fsync(f.fileno())
                print(f'  [{variant}] eval k={k_shot} seed={seed} done'
                      f'{f" ({n_skipped} already-logged, skipped)" if n_skipped else ""}')

                                                                                     
    n_logged = len(_completed_eval_episodes(ep_csv, variant))
    expected = len(CFG['eval_k_shots']) * len(CFG['eval_seeds']) * CFG['eval_episodes']
    assert n_logged == expected, (
        f'[{variant}] evaluation integrity check FAILED: expected {expected} episode '
        f'rows, found {n_logged} in {ep_csv}. Do not trust this variant\'s numbers -- '
        f'investigate before continuing.')
    print(f'  [{variant}] evaluation complete and verified ({n_logged}/{expected} episodes) -> {ep_csv}')


def rebuild_ablation_summary():
    import pandas as pd
    ep_csv = f"{CFG['output_dir']}/ablation_indomain_episode_metrics.csv"
    if not Path(ep_csv).exists():
        print('No ablation episode metrics yet.'); return
    df = pd.read_csv(ep_csv)
    rows = []
    for variant in df['variant'].unique():
        for k_shot in CFG['eval_k_shots']:
            sub = df[(df.variant == variant) & (df.k_shot == k_shot)]
            if len(sub) == 0: continue
            acc = sub['accuracy'].values
            ci95 = 1.96 * acc.std(ddof=1) / np.sqrt(len(acc))
            rows.append({'variant': variant, 'k_shot': k_shot, 'n_episodes': len(acc),
                         'mean_accuracy': acc.mean(), 'ci95': ci95,
                         'mean_macro_f1': sub['macro_f1'].mean(),
                         'mean_balanced_acc': sub['balanced_acc'].mean()})
    pd.DataFrame(rows).to_csv(f"{CFG['output_dir']}/ablation_indomain_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

print('evaluate_arm() [now episode-resumable] and rebuild_ablation_summary() ready')

# Cell 8 - Train and evaluate the random-init and SSL linear-probe arms.
def _variant_fully_evaluated(variant):
\
\
\
                                                                 
    ep_csv = f"{CFG['output_dir']}/ablation_indomain_episode_metrics.csv"
    if not Path(ep_csv).exists():
        return False
    expected = len(CFG['eval_k_shots']) * len(CFG['eval_seeds']) * CFG['eval_episodes']
    df = pd.read_csv(ep_csv)
    n_rows = len(df[df.variant == variant])
    return n_rows == expected


import pandas as pd

for variant in ['random_init', 'ssl_lp']:
    if _variant_fully_evaluated(variant):
        print(f'[{variant}] already fully evaluated ({CFG["eval_k_shots"]} x '
              f'{CFG["eval_seeds"]} x {CFG["eval_episodes"]} episodes found) — skipping.')
        continue

                                                                             
                                                                            
                                                                              
                                                                         
                                                                       
    gc.collect()
    torch.cuda.empty_cache()

    m = None
    try:
        m = train_arm(variant, patience=20)                                                  
        evaluate_arm(m, variant)
    except KeyboardInterrupt:
        print(f'\n[{variant}] training interrupted by user. The RESUME checkpoint '
              f'holds progress through the last FULLY COMPLETED epoch -- rerunning '
              f'this cell will continue from there (Cell 7b will NOT delete it; '
              f'it only cleans up once, guarded by its sentinel).')
        raise
    except torch.cuda.OutOfMemoryError as e:
        print(f'\n[{variant}] CUDA OOM: {e}')
        print('This usually means CUDA memory from a PREVIOUS interrupted run is')
        print('still fragmented/orphaned and empty_cache() alone cannot reclaim it.')
        print('Fix: Runtime -> Restart runtime, remount Drive, rerun Cells 0-2, then')
        print('rerun Cell 7b (safe -- sentinel prevents re-deleting progress) and')
        print('this cell again. It will resume from the last saved RESUME checkpoint,')
        print('not restart from epoch 1, as long as Cell 7b already ran once before.')
        raise
    finally:
                                                                             
                                                                          
        if m is not None:
            del m
        gc.collect()
        torch.cuda.empty_cache()

rebuild_ablation_summary()
print('\nBoth ablation arms complete -> ablation_indomain_summary.csv')

\
\
\
\
\
\

# Cell 9 - Run the paired MC-Dropout ablation and save its summary.
class SwinProtoNetFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(CFG['backbone'], pretrained=False,
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
    def predict_mc(self, sup, sup_lbl, qry, n_way, n_passes):
        self.train()
        probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                p = F.softmax(self(sup, sup_lbl, qry, n_way) / CFG['temperature'], dim=-1)
                probs.append(p.unsqueeze(0))
        self.eval()
        return torch.cat(probs, 0).mean(0)
    def predict_deterministic(self, sup, sup_lbl, qry, n_way):
        self.eval()
        with torch.no_grad():
            return F.softmax(self(sup, sup_lbl, qry, n_way) / CFG['temperature'], dim=-1)

protonet_full = SwinProtoNetFull().to(DEVICE)
state = torch.load(CFG['proto_ckpt'], map_location=DEVICE, weights_only=False)
protonet_full.load_state_dict(state)
protonet_full.eval()
print('Full Model checkpoint loaded for MC-Dropout ablation:', CFG['proto_ckpt'])

class_indices_test = {}
for i, (_, lbl) in enumerate(test_samples):
    class_indices_test.setdefault(lbl, []).append(i)

def sample_episode_mc(n_way, k_shot, n_query, seed):
    rng = random.Random(seed)
    classes = rng.sample(list(class_indices_test.keys()), n_way)
    si, sl, qi, ql = [], [], [], []
    for local, cls in enumerate(classes):
        chosen = rng.sample(class_indices_test[cls], k_shot + n_query)
        for i in chosen[:k_shot]:
            si.append(load_image(test_samples[i][0], val_tfm)); sl.append(local)
        for i in chosen[k_shot:]:
            qi.append(load_image(test_samples[i][0], val_tfm)); ql.append(local)
    return (torch.stack(si), torch.tensor(sl), torch.stack(qi), torch.tensor(ql))

mc_csv = f"{CFG['output_dir']}/ablation_mc_dropout_episode_metrics.csv"
mc_fields = ['k_shot', 'seed', 'episode_idx',
             'acc_mc', 'f1_mc', 'bacc_mc',
             'acc_deterministic', 'f1_deterministic', 'bacc_deterministic']

def _completed_mc_episodes():
\
\
\
\
                             
    done = set()
    if not Path(mc_csv).exists():
        return done
    with open(mc_csv, newline='') as f:
        for row in csv.DictReader(f):
            done.add((int(row['k_shot']), int(row['seed']), int(row['episode_idx'])))
    return done

write_header = not Path(mc_csv).exists()
done = _completed_mc_episodes()
if done:
    print(f'{len(done)} MC-Dropout episode(s) already logged on disk -- resuming, skipping those.')

with open(mc_csv, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(mc_fields)
    for k_shot in CFG['eval_k_shots']:
        for seed in CFG['eval_seeds']:
            n_skipped = 0
            for episode_idx in range(CFG['eval_episodes']):
                if (k_shot, seed, episode_idx) in done:
                    n_skipped += 1
                    continue
                ep_seed = seed * 1_000_000 + CFG['eval_n_way'] * 100_000 + k_shot * 10_000 + episode_idx
                sup, sup_lbl, qry, qry_lbl = sample_episode_mc(
                    CFG['eval_n_way'], k_shot, CFG['eval_n_query'], ep_seed)
                sup, sup_lbl = sup.to(DEVICE), sup_lbl.to(DEVICE)
                qry, qry_lbl = qry.to(DEVICE), qry_lbl.to(DEVICE)
                y_true = qry_lbl.cpu().numpy()

                p_mc = protonet_full.predict_mc(sup, sup_lbl, qry, CFG['eval_n_way'], CFG['mc_passes'])
                y_pred_mc = p_mc.argmax(-1).cpu().numpy()
                p_det = protonet_full.predict_deterministic(sup, sup_lbl, qry, CFG['eval_n_way'])
                y_pred_det = p_det.argmax(-1).cpu().numpy()

                writer.writerow([
                    k_shot, seed, episode_idx,
                    float((y_pred_mc == y_true).mean()),
                    f1_score(y_true, y_pred_mc, average='macro', zero_division=0),
                    balanced_accuracy_score(y_true, y_pred_mc),
                    float((y_pred_det == y_true).mean()),
                    f1_score(y_true, y_pred_det, average='macro', zero_division=0),
                    balanced_accuracy_score(y_true, y_pred_det),
                ])
                f.flush(); os.fsync(f.fileno())
            print(f'  k={k_shot} seed={seed} done'
                  f'{f" ({n_skipped} already-logged, skipped)" if n_skipped else ""}')

n_logged = len(_completed_mc_episodes())
n_expected = len(CFG['eval_k_shots']) * len(CFG['eval_seeds']) * CFG['eval_episodes']
assert n_logged == n_expected, (
    f'MC-Dropout episode log integrity check FAILED: expected {n_expected}, '
    f'found {n_logged} in {mc_csv}. Do not trust this for the Wilcoxon test below -- '
    f'investigate before continuing.')
print(f'Paired MC-Dropout ablation episodes complete and verified '
      f'({n_logged}/{n_expected}) -> {mc_csv}')

import pandas as pd
mc_df = pd.read_csv(mc_csv)

                                                                        
                                                                    
                                                                 
n_before = len(mc_df)
mc_df = mc_df.drop_duplicates(subset=['k_shot', 'seed', 'episode_idx'], keep='last')
if len(mc_df) < n_before:
    print(f'NOTE: dropped {n_before - len(mc_df)} duplicate episode row(s) before aggregating.')

expected_per_kshot = len(CFG['eval_seeds']) * CFG['eval_episodes']
mc_rows = []
for k_shot in CFG['eval_k_shots']:
    sub = mc_df[mc_df.k_shot == k_shot]
    assert len(sub) == expected_per_kshot, (
        f'k_shot={k_shot}: expected {expected_per_kshot} episodes, found {len(sub)}. '
        f'The MC-Dropout sweep (previous cell) is not yet complete -- rerun it before '
        f'trusting this Wilcoxon test.')
    stat, p = wilcoxon(sub['acc_mc'], sub['acc_deterministic'])
    mc_rows.append({
        'k_shot': k_shot, 'n_episodes': len(sub),
        'mean_acc_mc': sub['acc_mc'].mean(), 'mean_acc_deterministic': sub['acc_deterministic'].mean(),
        'mean_f1_mc': sub['f1_mc'].mean(), 'mean_f1_deterministic': sub['f1_deterministic'].mean(),
        'wilcoxon_stat': stat, 'wilcoxon_p_value': p,
    })
mc_summary_df = pd.DataFrame(mc_rows)
mc_summary_df.to_csv(f"{CFG['output_dir']}/ablation_mc_dropout_summary.csv", index=False)
print(mc_summary_df.to_string(index=False))

\
\
\
\
\
\
\
\

# Cell 10 - Load logged results and generate the final ablation figure and table.
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.linewidth': 0.8, 'axes.edgecolor': '#3A3A3A',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'legend.frameon': False, 'savefig.dpi': 300, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
K_SHOTS = [1, 5]

def try_read(path):
    p = Path(CFG['output_dir']) / path
    return pd.read_csv(p) if p.exists() else None

indomain_df = try_read('indomain_test/indomain_summary.csv')
baseline_df = try_read('baseline_comparison_summary.csv')
ablation_df = try_read('ablation_indomain_summary.csv')
mcdrop_df   = try_read('ablation_mc_dropout_summary.csv')

def get_full_model(k_shot):
    if indomain_df is None: return None
    row = indomain_df[(indomain_df.n_way == 8) & (indomain_df.k_shot == k_shot)]
    if row.empty: return None
    r = row.iloc[0]; return r['mean_accuracy'], r['ci95']

def get_wo_ssl(k_shot):
    if ablation_df is None: return None
    row = ablation_df[(ablation_df.variant == 'random_init') & (ablation_df.k_shot == k_shot)]
    if row.empty: return None
    r = row.iloc[0]; return r['mean_accuracy'], r['ci95']

def get_wo_finetune(k_shot):
    if ablation_df is None: return None
    row = ablation_df[(ablation_df.variant == 'ssl_lp') & (ablation_df.k_shot == k_shot)]
    if row.empty: return None
    r = row.iloc[0]; return r['mean_accuracy'], r['ci95']

def get_wo_prototype(k_shot):
    if baseline_df is None: return None
    sub = baseline_df[baseline_df.k_shot == k_shot]
    if sub.empty: return None
    best = sub.loc[sub['mean_accuracy'].idxmax()]
    return best['mean_accuracy'], best['ci95'], best['variant']

def get_wo_mcdropout(k_shot):
    if mcdrop_df is None: return None
    row = mcdrop_df[mcdrop_df.k_shot == k_shot]
    if row.empty: return None
    r = row.iloc[0]; return r['mean_acc_deterministic'], None

ARM_GETTERS = [
    ('Full Model',            get_full_model),
    ('w/o SSL pretraining',   get_wo_ssl),
    ('w/o full fine-tune',    get_wo_finetune),
    ('w/o Prototype head',    get_wo_prototype),
    ('w/o MC-Dropout',        get_wo_mcdropout),
]

fig, axes = plt.subplots(1, len(K_SHOTS), figsize=(9, 4.2), sharey=True)
for ax, k_shot in zip(axes, K_SHOTS):
    labels, heights, errs, colors, hatches, notes = [], [], [], [], [], []
    for i, (name, getter) in enumerate(ARM_GETTERS):
        result = getter(k_shot)
        if result is None:
            labels.append(name); heights.append(0.0); errs.append(0.0)
            colors.append('#DDDDDD'); hatches.append('//'); notes.append('PENDING')
        else:
            acc = result[0]; ci = result[1] if len(result) > 1 and result[1] is not None else 0.0
            extra = f' ({result[2]})' if len(result) > 2 else ''
            labels.append(name + extra); heights.append(acc * 100); errs.append(ci * 100)
            colors.append(PALETTE[i % len(PALETTE)]); hatches.append(None); notes.append(None)

    x = np.arange(len(labels))
    bars = ax.bar(x, heights, yerr=errs, color=colors, capsize=3, edgecolor='#333333', linewidth=0.6)
    for bar, hatch, note in zip(bars, hatches, notes):
        if hatch: bar.set_hatch(hatch)
        if note:
            ax.text(bar.get_x() + bar.get_width() / 2, 2, note, ha='center', va='bottom',
                     fontsize=7, rotation=90, color='#555555')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=7)
    ax.set_title(f'K={k_shot}', fontsize=9.5)
    ax.set_ylim(0, 100)
    if k_shot == K_SHOTS[0]:
        ax.set_ylabel('Accuracy (%)')

fig.suptitle('Ablation Study — In-Domain (N=8, Kvasir v2 Test)\n')
fig.tight_layout()
fig.savefig(f"{CFG['output_dir']}/fig_ablation_grouped_bars.png", bbox_inches='tight')
plt.show()

n_pending = sum(1 for k in K_SHOTS for name, getter in ARM_GETTERS if getter(k) is None)
if n_pending:
    print(f'\n[FIGURE PENDING — {n_pending} arm x k_shot cells awaiting completed evaluation]')
    print('Do not report this figure in the manuscript Results section until all bars are solid.')
else:
    print('\nAll 5 arms present for both k_shot values — figure is ready for final review.')

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.linewidth': 0.8, 'axes.edgecolor': '#3A3A3A',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'legend.frameon': False, 'savefig.dpi': 300, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
K_SHOTS = [1, 5]

def try_read(path):
    p = Path(CFG['output_dir']) / path
    return pd.read_csv(p) if p.exists() else None

indomain_df = try_read('indomain_test/indomain_summary.csv')
baseline_df = try_read('baseline_comparison_summary.csv')
ablation_df = try_read('ablation_indomain_summary.csv')
mcdrop_df   = try_read('ablation_mc_dropout_summary.csv')

                                                                             
                                                                         
                                                                         
                                                                        
                                                                          
                                                      
                                                                             

def get_full_model(k_shot):
    if indomain_df is None:
        return None
    row = indomain_df[(indomain_df.n_way == 8) & (indomain_df.k_shot == k_shot)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        'accuracy':      r['mean_accuracy'],
        'ci95':          r['ci95'],
        'macro_f1':      r.get('mean_macro_f1', np.nan),
        'balanced_acc':  r.get('mean_balanced_acc', np.nan),
        'n_episodes':    r.get('n_episodes', np.nan),
        'variant_label': None,
        'source_csv':    'indomain_summary.csv',
    }

def get_wo_ssl(k_shot):
    if ablation_df is None:
        return None
    row = ablation_df[(ablation_df.variant == 'random_init') & (ablation_df.k_shot == k_shot)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        'accuracy':      r['mean_accuracy'],
        'ci95':          r['ci95'],
        'macro_f1':      r.get('mean_macro_f1', np.nan),
        'balanced_acc':  r.get('mean_balanced_acc', np.nan),
        'n_episodes':    r.get('n_episodes', np.nan),
        'variant_label': 'random_init',
        'source_csv':    'ablation_indomain_summary.csv',
    }

def get_wo_finetune(k_shot):
    if ablation_df is None:
        return None
    row = ablation_df[(ablation_df.variant == 'ssl_lp') & (ablation_df.k_shot == k_shot)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        'accuracy':      r['mean_accuracy'],
        'ci95':          r['ci95'],
        'macro_f1':      r.get('mean_macro_f1', np.nan),
        'balanced_acc':  r.get('mean_balanced_acc', np.nan),
        'n_episodes':    r.get('n_episodes', np.nan),
        'variant_label': 'ssl_lp',
        'source_csv':    'ablation_indomain_summary.csv',
    }

def get_wo_prototype(k_shot):
    if baseline_df is None:
        return None
    sub = baseline_df[baseline_df.k_shot == k_shot]
    if sub.empty:
        return None
    best = sub.loc[sub['mean_accuracy'].idxmax()]
    return {
        'accuracy':      best['mean_accuracy'],
        'ci95':          best['ci95'],
        'macro_f1':      best.get('mean_macro_f1', np.nan),
        'balanced_acc':  best.get('mean_balanced_acc', np.nan),
        'n_episodes':    best.get('n_episodes', np.nan),
        'variant_label': best['variant'],                         
        'source_csv':    'baseline_comparison_summary.csv',
    }

def get_wo_mcdropout(k_shot):
    if mcdrop_df is None:
        return None
    row = mcdrop_df[mcdrop_df.k_shot == k_shot]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
                                                                          
        'accuracy':      r['mean_acc_deterministic'],
                                                                                    
        'ci95':          np.nan,
        'macro_f1':      r.get('mean_f1_deterministic', np.nan),
        'balanced_acc':  np.nan,                            
        'n_episodes':    r.get('n_episodes', np.nan),
        'variant_label': None,
        'source_csv':    'ablation_mc_dropout_summary.csv',
                                                                
        'mean_acc_mc':          r.get('mean_acc_mc', np.nan),
        'mean_f1_mc':           r.get('mean_f1_mc', np.nan),
        'wilcoxon_stat':        r.get('wilcoxon_stat', np.nan),
        'wilcoxon_p_value':     r.get('wilcoxon_p_value', np.nan),
    }

ARM_GETTERS = [
    ('Full Model',            get_full_model),
    ('w/o SSL pretraining',   get_wo_ssl),
    ('w/o full fine-tune',    get_wo_finetune),
    ('w/o Prototype head',    get_wo_prototype),
    ('w/o MC-Dropout',        get_wo_mcdropout),
]

                                                                             
                                                                         
                                                                       
                                                                        
                                                                        
                                                                             

table_rows = []
for k_shot in K_SHOTS:
    for arm_name, getter in ARM_GETTERS:
        result = getter(k_shot)
        if result is None:
            table_rows.append({
                'arm': arm_name, 'k_shot': k_shot, 'variant_label': None,
                'accuracy': np.nan, 'ci95': np.nan,
                'macro_f1': np.nan, 'balanced_acc': np.nan,
                'n_episodes': np.nan, 'source_csv': None,
                'status': 'PENDING -- not yet logged',
            })
        else:
            row = {
                'arm': arm_name, 'k_shot': k_shot,
                'variant_label': result.get('variant_label'),
                'accuracy': result.get('accuracy', np.nan),
                'ci95': result.get('ci95', np.nan),
                'macro_f1': result.get('macro_f1', np.nan),
                'balanced_acc': result.get('balanced_acc', np.nan),
                'n_episodes': result.get('n_episodes', np.nan),
                'source_csv': result.get('source_csv'),
                'status': 'logged',
            }
                                                                        
                                                
            for extra_key in ('mean_acc_mc', 'mean_f1_mc',
                               'wilcoxon_stat', 'wilcoxon_p_value'):
                if extra_key in result:
                    row[extra_key] = result[extra_key]
            table_rows.append(row)

arm_summary_df = pd.DataFrame(table_rows)

                                                                          
                                                                          
                          
arm_summary_df['accuracy_pct'] = arm_summary_df['accuracy'] * 100
arm_summary_df['ci95_pct']     = arm_summary_df['ci95'] * 100

display_cols = ['arm', 'k_shot', 'variant_label', 'accuracy_pct', 'ci95_pct',
                 'macro_f1', 'balanced_acc', 'n_episodes',
                 'mean_acc_mc', 'mean_f1_mc', 'wilcoxon_stat', 'wilcoxon_p_value',
                 'source_csv', 'status']
display_cols = [c for c in display_cols if c in arm_summary_df.columns]

print('=' * 100)
print('ABLATION ARM SUMMARY -- all values traced to the CSVs named in `source_csv`')
print('=' * 100)
print(arm_summary_df[display_cols].to_string(index=False))

                                                                          
arm_summary_csv_path = f"{CFG['output_dir']}/ablation_arm_summary_table.csv"
arm_summary_df.to_csv(arm_summary_csv_path, index=False)
print(f'\nSaved -> {arm_summary_csv_path}')

                                                                             
                                                                            
                                                                             

fig, axes = plt.subplots(1, len(K_SHOTS), figsize=(9, 4.2), sharey=True)
for ax, k_shot in zip(axes, K_SHOTS):
    labels, heights, errs, colors, hatches, notes = [], [], [], [], [], []
    for i, (name, getter) in enumerate(ARM_GETTERS):
        result = getter(k_shot)
        if result is None:
            labels.append(name); heights.append(0.0); errs.append(0.0)
            colors.append('#DDDDDD'); hatches.append('//'); notes.append('PENDING')
        else:
            acc = result['accuracy']
            ci = result['ci95'] if pd.notna(result.get('ci95')) else 0.0
            extra = f" ({result['variant_label']})" if result.get('variant_label') else ''
            labels.append(name + extra); heights.append(acc * 100); errs.append(ci * 100)
            colors.append(PALETTE[i % len(PALETTE)]); hatches.append(None); notes.append(None)

    x = np.arange(len(labels))
    bars = ax.bar(x, heights, yerr=errs, color=colors, capsize=3, edgecolor='#333333', linewidth=0.6)
    for bar, hatch, note in zip(bars, hatches, notes):
        if hatch:
            bar.set_hatch(hatch)
        if note:
            ax.text(bar.get_x() + bar.get_width() / 2, 2, note, ha='center', va='bottom',
                     fontsize=7, rotation=90, color='#555555')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=7)
    ax.set_title(f'K={k_shot}', fontsize=9.5)
    ax.set_ylim(0, 100)
    if k_shot == K_SHOTS[0]:
        ax.set_ylabel('Accuracy (%)')

fig.suptitle('Ablation Study — In-Domain (N=8, Kvasir v2 Test)\n')
fig.tight_layout()
fig.savefig(f"{CFG['output_dir']}/fig_ablation_grouped_bars.png", bbox_inches='tight')
plt.show()

n_pending = int(arm_summary_df['status'].eq('PENDING -- not yet logged').sum())
if n_pending:
    print(f'\n[FIGURE PENDING — {n_pending} arm x k_shot cells awaiting completed evaluation]')
    print('Do not report this figure in the manuscript Results section until all bars are solid.')
else:
    print('\nAll 5 arms present for both k_shot values — figure is ready for final review.')
