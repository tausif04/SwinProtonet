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

from google.colab import drive
drive.mount('/content/drive')


import subprocess, sys
for pkg in ['timm==0.9.16', 'albumentations>=1.3.0', 'einops', 'tqdm', 'scikit-learn', 'scipy']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
print('packages ready')

# Cell 1 - ### 0.3 — Imports, Base Seed, Config.

import zipfile, random, json, hashlib, csv, shutil, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import scipy.stats as st
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
K_SHOTS = [1, 5]
N_QUERY = 15

print('CFG ready. K_SHOTS =', K_SHOTS)


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

MAX = CFG['max_per_class']
MIN_NEEDED = CFG['k_shot_train'] + CFG['n_query'] + 10
all_classes = []
for cls_dir in sorted(kv2_root.iterdir()):
    if not cls_dir.is_dir():
        continue
    imgs = sorted(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.png')))
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
print(f'test_idx sha256 : {test_hash}  (must match the split-generation notebook)')

assert len(full_ds) == split_data['n_total'], (
    f'Rebuilt dataset size {len(full_ds)} != split n_total {split_data["n_total"]}. '
    f'Dataset rebuild does not match the saved split — stop and investigate before '
    f'trusting any downstream result.')

test_samples = [full_ds.samples[i] for i in test_idx]
test_by_class = {}
for path, lbl in test_samples:
    test_by_class.setdefault(lbl, []).append(path)
idx_to_class = {v: k for k, v in full_ds.class_to_idx.items()}
print('Test images per class:')
for lbl, paths in sorted(test_by_class.items()):
    print(f'  {idx_to_class[lbl]}: {len(paths)}')


class SwinProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            CFG['backbone'], pretrained=False, num_classes=0, img_size=CFG['img_size'])
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
        probs = torch.cat(probs, 0)
        mean_p = probs.mean(0)
        unc = probs.var(0).mean(-1)
        return mean_p.argmax(-1), mean_p, unc

protonet = SwinProtoNet().to(DEVICE)
state = torch.load(CFG['proto_ckpt'], map_location=DEVICE, weights_only=False)
protonet.load_state_dict(state)
protonet.eval()
print('SwinProtoNet loaded from', CFG['proto_ckpt'])


MAX_SHOT = max(K_SHOTS)
MIN_NEEDED_EP = MAX_SHOT + N_QUERY

qualifying = {lbl: paths for lbl, paths in test_by_class.items() if len(paths) >= MIN_NEEDED_EP}
dropped = {lbl: len(paths) for lbl, paths in test_by_class.items() if len(paths) < MIN_NEEDED_EP}
if dropped:
    print(f'Classes dropped from in-domain episodes (below {MIN_NEEDED_EP} test images):')
    for lbl, n in dropped.items():
        print(f'  {idx_to_class[lbl]}: {n}')

N_WAY_MAX = len(qualifying)
assert N_WAY_MAX >= 2, 'Not enough in-domain test classes with sufficient images'
INDOMAIN_LABELS = sorted(qualifying.keys())
INDOMAIN_LABEL_TO_LOCAL = {lbl: i for i, lbl in enumerate(INDOMAIN_LABELS)}
INDOMAIN_CLASS_NAMES = [idx_to_class[lbl] for lbl in INDOMAIN_LABELS]
INDOMAIN_SAMPLES = {INDOMAIN_LABEL_TO_LOCAL[lbl]: sorted(qualifying[lbl]) for lbl in INDOMAIN_LABELS}
print(f'N_WAY_MAX = {N_WAY_MAX} classes: {INDOMAIN_CLASS_NAMES}')

val_tfm = A.Compose([
    A.Resize(SIZE, SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def load_image(path):
    img = np.array(Image.open(path).convert('RGB'))
    return val_tfm(image=img)['image']

def mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float('nan'), float('nan'), 0
    if n == 1:
        return float(values[0]), float('nan'), 1
    m = float(values.mean())
    se = values.std(ddof=1) / np.sqrt(n)
    h = float(se * st.t.ppf((1 + confidence) / 2., n - 1))
    return m, h, n

print('Shared helpers ready (load_image, mean_ci).')

# Cell 2 - ## PART 1 — Grad-CAM + Uncertainty Analysis.

GRADCAM_DIR = f"{CFG['output_dir']}/gradcam_uncertainty"
os.makedirs(GRADCAM_DIR, exist_ok=True)

CFG.setdefault('mc_passes_cam', 10)                                                        
CFG.setdefault('cam_n_way', N_WAY_MAX)                                
CFG.setdefault('cam_k_shot', 5)                                                      
CFG.setdefault('cam_n_query_per_class', 4)                                        
                                                                         
                                                                            
                                                                               
                                                                  
CFG.setdefault('cam_seeds', [42, 43, 44])

GRADCAM_LOG = f'{GRADCAM_DIR}/gradcam_uncertainty_log.csv'
GRADCAM_FIELDS = ['timestamp', 'run_id', 'seed', 'query_global_idx',
                   'true_class', 'pred_class', 'correct',
                   'predictive_variance', 'predictive_entropy', 'uncertainty_pct',
                   'mean_uncertainty_cam', 'max_uncertainty_cam', 'figure_path']

print(f'Grad-CAM outputs -> {GRADCAM_DIR}')
print(f'Log               -> {GRADCAM_LOG}')
print(f'cam_seeds         -> {CFG["cam_seeds"]}')


class SwinGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self._fwd.remove(); self._bwd.remove()

    def _to_spatial(self, t):
        if t.dim() == 4:
            return t
        if t.dim() == 3:
            b, n, c = t.shape
            s = int(round(n ** 0.5))
            assert s * s == n, f'Non-square token grid (N={n}) -- cannot reshape for CAM'
            return t.reshape(b, s, s, c)
        raise ValueError(f'Unexpected activation shape {tuple(t.shape)}')

    def generate(self, score: torch.Tensor, out_size=224):
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        acts = self._to_spatial(self.activations.detach())
        grads = self._to_spatial(self.gradients.detach())
        weights = grads.mean(dim=(1, 2))
        cam = torch.einsum('bhwc,bc->bhw', acts, weights)
        cam = F.relu(cam)
        cam = cam[0]
        cam = cam - cam.min()
        denom = cam.max()
        cam = cam / denom if denom > 1e-8 else cam
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(out_size, out_size), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

def get_target_layer(model):
    return model.encoder.layers[-1]


def _prototypes_from_support(model, sup, sup_lbl, n_way):
    model.eval()
    with torch.no_grad():
        s_emb = model.encode(sup)
        protos = model.compute_prototypes(s_emb, sup_lbl, n_way)
    return protos

def single_query_cam(model, query_img, protos, target_class, cam_engine, dropout_active: bool):
    model.train() if dropout_active else model.eval()
    q = query_img.unsqueeze(0).clone().requires_grad_(False)
    qe = model.encode(q)
    logits = -torch.cdist(qe.unsqueeze(0), protos.unsqueeze(0)).squeeze(0)
    score = logits[0, target_class]
    cam = cam_engine.generate(score)
    model.eval()
    return cam, logits.detach()

def compute_query_uncertainty_and_cams(model, sup, sup_lbl, query_img, n_way,
                                        mc_passes, mc_passes_cam):
    protos = _prototypes_from_support(model, sup, sup_lbl, n_way)

    model.train()
    probs_stack = []
    with torch.no_grad():
        for _ in range(mc_passes):
            qe = model.encode(query_img.unsqueeze(0))
            logits = -torch.cdist(qe.unsqueeze(0), protos.unsqueeze(0)).squeeze(0)
            p = F.softmax(logits / CFG['temperature'], dim=-1)
            probs_stack.append(p)
    model.eval()
    probs_stack = torch.cat(probs_stack, 0)
    mean_p = probs_stack.mean(0)
    pred_class = int(mean_p.argmax().item())
    predictive_variance = float(probs_stack.var(0).mean().item())
    predictive_entropy = float(-(mean_p * (mean_p.clamp_min(1e-8)).log()).sum().item())
    uncertainty_pct = float(100.0 * predictive_entropy / np.log(n_way))

    target_layer = get_target_layer(model)
    cam_engine = SwinGradCAM(model, target_layer)
    try:
        point_cam, _ = single_query_cam(model, query_img, protos, pred_class, cam_engine, dropout_active=False)
        stoch_cams = []
        for _ in range(mc_passes_cam):
            cam_i, _ = single_query_cam(model, query_img, protos, pred_class, cam_engine, dropout_active=True)
            stoch_cams.append(cam_i)
        stoch_cams = np.stack(stoch_cams, 0)
        uncertainty_cam = stoch_cams.var(0)
        uncertainty_cam = uncertainty_cam / (uncertainty_cam.max() + 1e-8)
    finally:
        cam_engine.remove()

    return {
        'pred_class': pred_class,
        'predictive_variance': predictive_variance,
        'predictive_entropy': predictive_entropy,
        'uncertainty_pct': uncertainty_pct,
        'point_cam': point_cam,
        'uncertainty_cam': uncertainty_cam,
    }


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

import matplotlib.pyplot as plt
import matplotlib as mpl

def tensor_to_rgb(img_tensor):
    arr = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr, 0, 1)

def overlay_heatmap(rgb_img, cam, cmap='jet', alpha=0.45):
                                                                    
                                                   
    heat = mpl.colormaps[cmap](cam)[..., :3]
    return np.clip((1 - alpha) * rgb_img + alpha * heat, 0, 1)

def make_exemplar_figure(rgb_img, point_cam, unc_cam, true_name, pred_name,
                          correct, pred_var, pred_ent, unc_pct, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    axes[0].imshow(rgb_img); axes[0].set_title('Input frame', fontsize=9)
    axes[1].imshow(overlay_heatmap(rgb_img, point_cam))
    axes[1].set_title('Grad-CAM (decision)', fontsize=9)
    axes[2].imshow(overlay_heatmap(rgb_img, unc_cam, cmap='magma'))
    axes[2].set_title('Uncertainty-CAM\n(MC-dropout CAM variance)', fontsize=9)
    axes[3].axis('off')
    status = 'CORRECT' if correct else 'INCORRECT'
    color = '#2E7D32' if correct else '#C62828'
    axes[3].text(0.02, 0.88, f'True:  {true_name}', fontsize=10, transform=axes[3].transAxes)
    axes[3].text(0.02, 0.74, f'Pred:  {pred_name}', fontsize=10, transform=axes[3].transAxes)
    axes[3].text(0.02, 0.60, status, fontsize=11, fontweight='bold', color=color, transform=axes[3].transAxes)
    axes[3].text(0.02, 0.42, f'Uncertainty: {unc_pct:.1f}%', fontsize=12, fontweight='bold',
                 color='#B8860B', transform=axes[3].transAxes)
    axes[3].text(0.02, 0.26, f'Predictive variance: {pred_var:.4f}', fontsize=8.5, transform=axes[3].transAxes)
    axes[3].text(0.02, 0.14, f'Predictive entropy:  {pred_ent:.4f}', fontsize=8.5, transform=axes[3].transAxes)
    for ax in axes[:3]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    tmp_path = f'{save_path}.tmp.png'
    fig.savefig(tmp_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    os.replace(tmp_path, save_path)

# Cell 3 - ### 1.5 — Multi-Seed, Resumable Grad-CAM Gallery.

def _ensure_log_header():
    if not Path(GRADCAM_LOG).exists():
        with open(GRADCAM_LOG, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=GRADCAM_FIELDS).writeheader()

def _already_logged():
    done = set()
    if Path(GRADCAM_LOG).exists():
        with open(GRADCAM_LOG) as f:
            for row in csv.DictReader(f):
                done.add(row['query_global_idx'])
    return done

def build_gradcam_gallery_one_seed(model, seed, n_way=None, k_shot=None, n_query_per_class=None):
    n_way = n_way or CFG['cam_n_way']
    k_shot = k_shot or CFG['cam_k_shot']
    n_query_per_class = n_query_per_class or CFG['cam_n_query_per_class']
    run_id = f'gallery_seed{seed}'

    rng = np.random.RandomState(seed)
    class_ids = list(range(n_way))

    sup_imgs, sup_lbls = [], []
    query_records = []
    for local, cls in enumerate(class_ids):
        pool = INDOMAIN_SAMPLES[cls]
        chosen = rng.choice(len(pool), size=k_shot + n_query_per_class, replace=False)
        chosen_paths = [pool[i] for i in chosen]
        for p in chosen_paths[:k_shot]:
            sup_imgs.append(load_image(p)); sup_lbls.append(local)
        for qi, p in enumerate(chosen_paths[k_shot:]):
            img_t = load_image(p)
            gid = f'seed{seed}_cls{cls}_q{qi}'
            query_records.append((img_t, local, gid, INDOMAIN_CLASS_NAMES[local]))

    sup = torch.stack(sup_imgs).to(DEVICE)
    sup_lbl = torch.tensor(sup_lbls).to(DEVICE)

    _ensure_log_header()
    done = _already_logged()
    n_written = 0

    with open(GRADCAM_LOG, 'a', newline='') as log_f:
        writer = csv.DictWriter(log_f, fieldnames=GRADCAM_FIELDS)
        for img_t, true_local, gid, true_name in tqdm(query_records, desc=f'seed{seed}[gradcam]'):
            if gid in done:
                continue
            img_dev = img_t.to(DEVICE)
            result = compute_query_uncertainty_and_cams(
                model, sup, sup_lbl, img_dev, n_way,
                mc_passes=CFG['mc_passes'], mc_passes_cam=CFG['mc_passes_cam'])

            correct = int(result['pred_class'] == true_local)
            pred_name = INDOMAIN_CLASS_NAMES[result['pred_class']]
            fig_path = f'{GRADCAM_DIR}/{gid}.png'
            make_exemplar_figure(
                tensor_to_rgb(img_t), result['point_cam'], result['uncertainty_cam'],
                true_name, pred_name, correct,
                result['predictive_variance'], result['predictive_entropy'],
                result['uncertainty_pct'], fig_path)

            row = {
                'timestamp': time.time(), 'run_id': run_id, 'seed': seed,
                'query_global_idx': gid, 'true_class': true_name,
                'pred_class': pred_name, 'correct': correct,
                'predictive_variance': result['predictive_variance'],
                'predictive_entropy': result['predictive_entropy'],
                'uncertainty_pct': result['uncertainty_pct'],
                'mean_uncertainty_cam': float(result['uncertainty_cam'].mean()),
                'max_uncertainty_cam': float(result['uncertainty_cam'].max()),
                'figure_path': fig_path,
            }
            writer.writerow(row)
            log_f.flush(); os.fsync(log_f.fileno())
            n_written += 1
            done.add(gid)

    expected_total = len(query_records)
    logged_now = sum(1 for g in [r[2] for r in query_records] if g in _already_logged())
    print(f'  seed {seed}: {n_written} new rows written | {logged_now}/{expected_total} rows verified on disk')
    return n_written

def build_gradcam_gallery_multiseed(model, seeds=None):
    seeds = seeds or CFG['cam_seeds']
    total_new = 0
    for seed in seeds:
        total_new += build_gradcam_gallery_one_seed(model, seed)
    print(f'Grad-CAM multi-seed gallery done | seeds={seeds} | {total_new} new rows this run '
          f'| log -> {GRADCAM_LOG}')
    return total_new


def summarize_uncertainty_pct(log_path=None):
    log_path = log_path or GRADCAM_LOG
    assert Path(log_path).exists(), f'{log_path} not found -- run 1.5 first.'
    df = pd.read_csv(log_path)
    assert len(df) > 0, f'{log_path} is empty -- run 1.5 first.'

    print(f'Loaded {len(df)} logged queries across seeds {sorted(df.seed.unique().tolist())}')
    print()
    print('--- POOLED (all seeds combined) ---')
    pooled = df.groupby('correct')['uncertainty_pct'].agg(['mean', 'std', 'count'])
    print(pooled)
    if df.correct.nunique() > 1:
        u_stat, p_val = st.mannwhitneyu(
            df.loc[df.correct == 0, 'uncertainty_pct'],
            df.loc[df.correct == 1, 'uncertainty_pct'],
            alternative='greater')
        print(f'\nMann-Whitney U (incorrect > correct, pooled queries): U={u_stat:.1f}, p={p_val:.4g}')
    else:
        print('\nOnly one correctness class present in the pooled log -- cannot run the pooled test yet.')

    print()
    print('--- PER-SEED (mean uncertainty %, by correctness) ---')
    per_seed = df.groupby(['seed', 'correct'])['uncertainty_pct'].mean().unstack('correct')
    per_seed.columns = [f'correct={c}' for c in per_seed.columns]
    print(per_seed)

    if per_seed.shape[1] == 2 and per_seed.shape[0] >= 2:
        diffs = (per_seed.iloc[:, 0] - per_seed.iloc[:, 1]).values                                 
        if len(diffs) >= 6:
            w_stat, w_p = st.wilcoxon(diffs, alternative='greater')
            print(f'\nWilcoxon signed-rank across seeds (incorrect-mean > correct-mean): '
                  f'W={w_stat:.2f}, p={w_p:.4g}, n_seeds={len(diffs)}')
        else:
            print(f'\nOnly {len(diffs)} seed(s) with both classes present -- Wilcoxon needs >=6 '
                  f'paired differences for an asymptotically valid p-value. Report the per-seed '
                  f'table above descriptively and flag this as a seed-count limitation rather '
                  f'than computing an underpowered test.')
    else:
        print('\nNot enough seeds/classes yet for a per-seed paired comparison.')

    m, h, n = mean_ci(df.groupby('seed')['uncertainty_pct'].mean().values)
    if n >= 2:
        print(f'\nMean Uncertainty(%) across seeds: {m:.2f}% +/- {h:.2f}% (95% CI, n_seeds={n})')
    else:
        print(f'\nOnly {n} seed logged -- cannot report a cross-seed CI yet '
              f'(add more seeds to CFG["cam_seeds"] and re-run 1.5).')
    return df

def build_uncertainty_corner_panel(df=None, top_k=3):
    df = df if df is not None else pd.read_csv(GRADCAM_LOG)
    if len(df) == 0:
        print('Log is empty -- run 1.5 first.')
        return
    corners = {
        'High uncertainty, INCORRECT': df[df.correct == 0].nlargest(top_k, 'uncertainty_pct'),
        'High uncertainty, correct': df[df.correct == 1].nlargest(top_k, 'uncertainty_pct'),
        'Low uncertainty, correct': df[df.correct == 1].nsmallest(top_k, 'uncertainty_pct'),
        'Low uncertainty, INCORRECT': df[df.correct == 0].nsmallest(top_k, 'uncertainty_pct'),
    }
    fig, axes = plt.subplots(4, top_k, figsize=(3.2 * top_k, 12))
    for row_i, (title, sub) in enumerate(corners.items()):
        for col_i in range(top_k):
            ax = axes[row_i, col_i]
            if col_i < len(sub):
                r = sub.iloc[col_i]
                img = Image.open(r['figure_path'])
                ax.imshow(img)
            ax.axis('off')
        axes[row_i, 0].text(-0.05, 0.5, title, fontsize=10, fontweight='bold', rotation=90,
                             va='center', ha='center', transform=axes[row_i, 0].transAxes)
    fig.suptitle(f'Grad-CAM / Uncertainty-CAM -- corner-case exemplars by Uncertainty(%) '
                 f'(n={len(df)} logged queries, seeds={sorted(df.seed.unique().tolist())})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out_path = f'{GRADCAM_DIR}/uncertainty_corner_panel.png'
    tmp = f'{out_path}.tmp.png'
    fig.savefig(tmp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    os.replace(tmp, out_path)
    print(f'Saved -> {out_path}')
    return corners


from tqdm.auto import tqdm

print(f'[1] Multi-seed Grad-CAM gallery | seeds={CFG["cam_seeds"]} | n_way={CFG["cam_n_way"]} | '
      f'k_shot={CFG["cam_k_shot"]} | n_query_per_class={CFG["cam_n_query_per_class"]}')
build_gradcam_gallery_multiseed(protonet)

print()
gradcam_df = summarize_uncertainty_pct()
build_uncertainty_corner_panel(gradcam_df)

# Cell 4 - ## PART 2 — Uncertainty-Aware Clinical Decision Support System.

import random as _rand

assert 'split_data' in globals() and 'full_ds' in globals(), (
    'split_data / full_ds not found -- run Part 0 first.')

CFG.setdefault('referral_target_risk', 0.05)                                
CFG.setdefault('referral_target_coverage', 0.80)                                 
CFG.setdefault('tier1_target_risk', 0.02)                                                  

                                                                      
                                                                          
                                                                        
                                                                       
                                                                             
CFG.setdefault('referral_cal_seeds', [42, 43, 44])
CFG.setdefault('referral_cal_episodes', 100)
CFG.setdefault('referral_test_seeds', [42, 43, 44, 45, 46])
CFG.setdefault('referral_test_episodes', 100)

val_idx_local = sorted(split_data['val'])
val_samples_all = [full_ds.samples[i] for i in val_idx_local]
val_by_global_class = {}
for path, lbl in val_samples_all:
    val_by_global_class.setdefault(lbl, []).append(path)

MIN_NEEDED_VAL = max(K_SHOTS) + N_QUERY
qualifying_local = [c for c in range(N_WAY_MAX)
                     if len(val_by_global_class.get(INDOMAIN_LABELS[c], [])) >= MIN_NEEDED_VAL]
if len(qualifying_local) < N_WAY_MAX:
    dropped_cls = [INDOMAIN_CLASS_NAMES[c] for c in range(N_WAY_MAX) if c not in qualifying_local]
    print(f'  NOTE: {len(dropped_cls)} in-domain class(es) have too few VAL images and are '
          f'excluded from the referral val/test pools: {dropped_cls}. Flag in Limitations if '
          f'REFERRAL_N_WAY < N_WAY_MAX.')

REFERRAL_N_WAY = len(qualifying_local)
assert REFERRAL_N_WAY >= 2, 'Not enough classes with sufficient val+test images for referral evaluation'
_remap = {old: new for new, old in enumerate(qualifying_local)}
REFERRAL_CLASS_NAMES = [INDOMAIN_CLASS_NAMES[old] for old in qualifying_local]
VAL_SAMPLES_BY_LOCAL = {_remap[old]: sorted(val_by_global_class[INDOMAIN_LABELS[old]]) for old in qualifying_local}
TEST_SAMPLES_BY_LOCAL = {_remap[old]: INDOMAIN_SAMPLES[old] for old in qualifying_local}

print(f'Referral evaluation classes: {REFERRAL_N_WAY} (N_WAY_MAX={N_WAY_MAX}) -> {REFERRAL_CLASS_NAMES}')
print(f'cal_seeds={CFG["referral_cal_seeds"]} x {CFG["referral_cal_episodes"]} episodes/seed')
print(f'test_seeds={CFG["referral_test_seeds"]} x {CFG["referral_test_episodes"]} episodes/seed')


def sample_referral_episode(samples_by_local, n_way, k_shot, n_query, seed, episode_idx):
    rng = _rand.Random(seed * 1_000_000 + n_way * 100_000 + k_shot * 10_000 + episode_idx)
    available = list(samples_by_local.keys())
    class_ids = available if n_way >= len(available) else rng.sample(available, n_way)
    sup_imgs, sup_lbls, qry_imgs, qry_lbls, qry_class_names = [], [], [], [], []
    for local, cls in enumerate(class_ids):
        pool = samples_by_local[cls]
        chosen = rng.sample(pool, k_shot + n_query)
        for p in chosen[:k_shot]:
            sup_imgs.append(load_image(p)); sup_lbls.append(local)
        for p in chosen[k_shot:]:
            qry_imgs.append(load_image(p)); qry_lbls.append(local)
            qry_class_names.append(REFERRAL_CLASS_NAMES[cls])
    return (torch.stack(sup_imgs), torch.tensor(sup_lbls),
            torch.stack(qry_imgs), torch.tensor(qry_lbls), qry_class_names, class_ids)
                                                                         
                                                                        
                                                                           

def evaluate_referral_episode(model, samples_by_local, n_way, k_shot, n_query, seed, episode_idx):
    sup, sup_lbl, qry, qry_lbl, qry_class_names, class_ids = sample_referral_episode(
        samples_by_local, n_way, k_shot, n_query, seed, episode_idx)
    sup, sup_lbl = sup.to(DEVICE), sup_lbl.to(DEVICE)
    qry, qry_lbl = qry.to(DEVICE), qry_lbl.to(DEVICE)

    pred, mean_p, unc = model.predict_with_uncertainty(sup, sup_lbl, qry, n_way, n_passes=CFG['mc_passes'])
    entropy = -(mean_p * mean_p.clamp_min(1e-8).log()).sum(-1)
    confidence = mean_p.max(-1).values
    uncertainty_pct = 100.0 * entropy / np.log(n_way)

    model.eval()
    with torch.no_grad():
        det_logits = model(sup, sup_lbl, qry, n_way)
        top2 = det_logits.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]

    correct = (pred == qry_lbl).long()
    pred_names = [REFERRAL_CLASS_NAMES[class_ids[p]] for p in pred.tolist()]

    return {
        'true_class': qry_class_names,
        'pred_class': pred_names,
        'correct': correct.cpu().numpy(),
        'predictive_variance': unc.detach().cpu().numpy(),
        'predictive_entropy': entropy.detach().cpu().numpy(),
        'uncertainty_pct': uncertainty_pct.detach().cpu().numpy(),
        'confidence': confidence.detach().cpu().numpy(),
        'margin': margin.detach().cpu().numpy(),
    }

# Cell 5 - ### 2.3 — Resumable, Multi-Seed Logging Loop.

REFERRAL_VAL_LOG = f'{GRADCAM_DIR}/referral_val_log.csv'
REFERRAL_TEST_LOG = f'{GRADCAM_DIR}/referral_test_log.csv'
REFERRAL_FIELDS = ['split', 'n_way', 'k_shot', 'seed', 'episode_idx', 'query_idx',
                    'true_class', 'pred_class', 'correct', 'predictive_variance',
                    'predictive_entropy', 'uncertainty_pct', 'confidence', 'margin', 'mc_passes']

def _referral_progress_path(split):
    return f'{GRADCAM_DIR}/referral_progress_{split}.json'

def _save_referral_progress(split, done_set):
    tmp = f'{_referral_progress_path(split)}.tmp'
    with open(tmp, 'w') as f:
        json.dump(sorted(done_set), f)
    os.replace(tmp, _referral_progress_path(split))

def _scan_completed_episodes(log_path, n_way, n_query):
    from collections import Counter
    counts = Counter()
    if not Path(log_path).exists():
        return set()
    with open(log_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (int(row['k_shot']), int(row['seed']), int(row['episode_idx']))
            counts[key] += 1
    complete = {key for key, c in counts.items() if c == n_way * n_query}
    incomplete = {key: c for key, c in counts.items() if 0 < c < n_way * n_query}
    if incomplete:
        preview = sorted(incomplete)[:5]
        print(f'  WARNING: {len(incomplete)} episode(s) in {log_path} have PARTIAL rows on disk '
              f'(a mid-write disconnect) -- treated as NOT done and will be RE-RUN, which '
              f'duplicates their existing partial rows. Deduplicate {log_path} for these '
              f'(k_shot, seed, episode_idx) keys before trusting it for reporting: {preview}'
              f'{" ..." if len(incomplete) > 5 else ""}')
    return complete

def run_referral_sweep(model, samples_by_local, split, log_path, seeds, n_episodes, k_shots, n_way):
    write_header = not Path(log_path).exists()
    n_query = N_QUERY
    complete_keys = _scan_completed_episodes(log_path, n_way, n_query)
    _save_referral_progress(split, {f'{n_way}|{k}|{s}|{e}' for (k, s, e) in complete_keys})
    print(f'  {split}: {len(complete_keys)} episode(s) already complete on disk -- will be skipped')

    with open(log_path, 'a', newline='') as log_f:
        writer = csv.DictWriter(log_f, fieldnames=REFERRAL_FIELDS)
        if write_header:
            writer.writeheader()
        for k_shot in k_shots:
            for seed in seeds:
                for episode_idx in tqdm(range(n_episodes), desc=f'{split}[k={k_shot} seed={seed}]'):
                    key = (k_shot, seed, episode_idx)
                    if key in complete_keys:
                        continue
                    result = evaluate_referral_episode(model, samples_by_local, n_way, k_shot, n_query, seed, episode_idx)
                    n_rows = len(result['correct'])
                    for qi in range(n_rows):
                        writer.writerow({
                            'split': split, 'n_way': n_way, 'k_shot': k_shot, 'seed': seed,
                            'episode_idx': episode_idx, 'query_idx': qi,
                            'true_class': result['true_class'][qi], 'pred_class': result['pred_class'][qi],
                            'correct': int(result['correct'][qi]),
                            'predictive_variance': float(result['predictive_variance'][qi]),
                            'predictive_entropy': float(result['predictive_entropy'][qi]),
                            'uncertainty_pct': float(result['uncertainty_pct'][qi]),
                            'confidence': float(result['confidence'][qi]),
                            'margin': float(result['margin'][qi]),
                            'mc_passes': CFG['mc_passes'],
                        })
                    log_f.flush(); os.fsync(log_f.fileno())
                    assert n_rows == n_way * n_query, (
                        f'Referral log integrity check FAILED for {split} {key}: '
                        f'expected {n_way * n_query} rows, wrote {n_rows}.')
                    complete_keys.add(key)
                    _save_referral_progress(split, {f'{n_way}|{k}|{s}|{e}' for (k, s, e) in complete_keys})
    print(f'{split} sweep complete | rows on disk verified per-episode | log -> {log_path}')

def _load_seed_filtered(log_path, seeds, label):
    df = pd.read_csv(log_path)
    n_before = len(df)
    df = df[df['seed'].isin(seeds)].reset_index(drop=True)
    n_after = len(df)
    if n_after < n_before:
        print(f'  NOTE: {label} contains {n_before - n_after} row(s) from seed(s) other than '
              f'{seeds} -- EXCLUDED from this analysis. Raw rows untouched on disk.')
    if n_after == 0:
        print(f'  WARNING: {label} has ZERO rows for seed(s) {seeds}. Run 2.3 for this seed first.')
    return df


REFERRAL_TAU_JSON = f'{GRADCAM_DIR}/referral_calibrated_tau.json'
REFERRAL_TAU_PERSEED_JSON = f'{GRADCAM_DIR}/referral_calibrated_tau_perseed.json'

def _risk_coverage_from_scores(scores, correct, ascending_is_more_confident=True):
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order_key = scores if ascending_is_more_confident else -scores
    order = np.argsort(order_key)
    correct_sorted = correct[order]
    n = len(correct_sorted)
    cum_correct = np.cumsum(correct_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - cum_correct / np.arange(1, n + 1)
    sorted_scores = scores[order]
    return coverage, risk, sorted_scores

def aurc_from_curve(coverage, risk):
    return float(np.trapz(risk, coverage))

def calibrate_tau(val_df, score_col, ascending_is_more_confident, target_risk, target_coverage):
    coverage, risk, sorted_scores = _risk_coverage_from_scores(
        val_df[score_col].values, val_df['correct'].values, ascending_is_more_confident)
    feasible = np.where(risk <= target_risk)[0]
    if len(feasible) > 0:
        idx = feasible[-1]
        tau_by_risk = float(sorted_scores[idx]); coverage_at_risk_tau = float(coverage[idx]); achieved_risk = float(risk[idx])
    else:
        tau_by_risk, coverage_at_risk_tau, achieved_risk = None, 0.0, None
        print(f'  WARNING: no threshold on VAL achieves risk <= {target_risk} for {score_col}.')
    idx_cov = min(int(round(target_coverage * len(sorted_scores))) - 1, len(sorted_scores) - 1)
    idx_cov = max(idx_cov, 0)
    tau_by_coverage = float(sorted_scores[idx_cov]); risk_at_coverage_tau = float(risk[idx_cov])
    return {
        'score_col': score_col,
        'target_risk_rule': {'target_risk': target_risk, 'tau': tau_by_risk,
                              'achieved_coverage': coverage_at_risk_tau, 'achieved_risk': achieved_risk},
        'target_coverage_rule': {'target_coverage': target_coverage, 'tau': tau_by_coverage,
                                  'achieved_coverage': float(coverage[idx_cov]), 'achieved_risk': risk_at_coverage_tau},
    }

def _calibrate_all_scores(val_df):
    out = {}
    for k_shot in sorted(val_df['k_shot'].unique()):
        sub = val_df[val_df.k_shot == k_shot]
        out[str(k_shot)] = {
            'predictive_variance': calibrate_tau(sub, 'predictive_variance', True, CFG['referral_target_risk'], CFG['referral_target_coverage']),
            'predictive_entropy': calibrate_tau(sub, 'predictive_entropy', True, CFG['referral_target_risk'], CFG['referral_target_coverage']),
            'margin': calibrate_tau(sub, 'margin', False, CFG['referral_target_risk'], CFG['referral_target_coverage']),
        }
    return out

def run_calibration_pooled():
    val_df = _load_seed_filtered(REFERRAL_VAL_LOG, CFG['referral_cal_seeds'], 'REFERRAL_VAL_LOG')
    calibration = {'timestamp': time.time(), 'val_log': REFERRAL_VAL_LOG,
                    'val_n_rows': len(val_df), 'seeds_pooled': CFG['referral_cal_seeds'],
                    'by_k_shot': _calibrate_all_scores(val_df)}
    tmp = f'{REFERRAL_TAU_JSON}.tmp'
    with open(tmp, 'w') as f: json.dump(calibration, f, indent=2)
    os.replace(tmp, REFERRAL_TAU_JSON)
    print(f'Pooled calibration (DEPLOYED tau) -> {REFERRAL_TAU_JSON}')
    for k_shot, d in calibration['by_k_shot'].items():
        for score_col, res in d.items():
            r = res['target_risk_rule']
            print(f'  k={k_shot} {score_col:>20s} | target-risk tau={r["tau"]!s:>10} '
                  f'cov={r["achieved_coverage"]:.2f} risk={r["achieved_risk"]}')
    return calibration

def run_calibration_perseed():
    val_df_all = pd.read_csv(REFERRAL_VAL_LOG)
    per_seed = {}
    for seed in CFG['referral_cal_seeds']:
        sub = val_df_all[val_df_all.seed == seed]
        if len(sub) == 0:
            print(f'  seed {seed}: no VAL rows logged -- skipping.')
            continue
        per_seed[str(seed)] = _calibrate_all_scores(sub)
    tmp = f'{REFERRAL_TAU_PERSEED_JSON}.tmp'
    with open(tmp, 'w') as f: json.dump(per_seed, f, indent=2)
    os.replace(tmp, REFERRAL_TAU_PERSEED_JSON)
    print(f'Per-seed calibration (for CI reporting only) -> {REFERRAL_TAU_PERSEED_JSON}')

    rows = []
    for seed, by_k in per_seed.items():
        for k_shot, by_score in by_k.items():
            for score_col, res in by_score.items():
                r = res['target_risk_rule']
                rows.append({'seed': int(seed), 'k_shot': int(k_shot), 'score': score_col,
                             'tau': r['tau'], 'achieved_coverage': r['achieved_coverage'], 'achieved_risk': r['achieved_risk']})
    perseed_df = pd.DataFrame(rows)
    print('\nSeed-to-seed variability of the target-risk operating point:')
    for (k_shot, score_col), g in perseed_df.groupby(['k_shot', 'score']):
        cov_m, cov_h, n = mean_ci(g['achieved_coverage'].dropna().values)
        if n >= 2:
            print(f'  k={k_shot} {score_col:>20s} | achieved_coverage = {cov_m:.3f} +/- {cov_h:.3f} (95% CI, n_seeds={n})')
        else:
            print(f'  k={k_shot} {score_col:>20s} | only {n} seed(s) with a feasible tau -- no CI yet')
    return perseed_df


def evaluate_test_risk_coverage_pooled():
    assert Path(REFERRAL_TAU_JSON).exists(), 'Run 2.4 (run_calibration_pooled) before this cell.'
    with open(REFERRAL_TAU_JSON) as f:
        calibration = json.load(f)
    test_df = _load_seed_filtered(REFERRAL_TEST_LOG, CFG['referral_test_seeds'], 'REFERRAL_TEST_LOG')

    curves, headline_rows = {}, []
    for k_shot in sorted(test_df['k_shot'].unique()):
        sub = test_df[test_df.k_shot == k_shot]
        curves[k_shot] = {}
        for score_col, ascending in [('predictive_variance', True), ('predictive_entropy', True), ('margin', False)]:
            coverage, risk, sorted_scores = _risk_coverage_from_scores(sub[score_col].values, sub['correct'].values, ascending)
            aurc = aurc_from_curve(coverage, risk)
            curves[k_shot][score_col] = {'coverage': coverage, 'risk': risk, 'aurc': aurc}
            cal = calibration['by_k_shot'].get(str(k_shot), {}).get(score_col)
            for rule_name, rule_key in [('target_risk_rule', 'target_risk_rule'), ('target_coverage_rule', 'target_coverage_rule')]:
                r = cal[rule_key]
                tau = r['tau']
                if tau is None:
                    continue
                accept = (sub[score_col] <= tau) if ascending else (sub[score_col] >= tau)
                cov = float(accept.mean())
                risk_at = float(1 - sub.loc[accept, 'correct'].mean()) if accept.any() else float('nan')
                headline_rows.append({
                    'k_shot': k_shot, 'score': score_col, 'rule': rule_name, 'tau_from_val': tau,
                    'test_coverage': cov, 'test_risk': risk_at, 'test_referral_rate': 1 - cov,
                    'accuracy_on_referred_cases': float(sub.loc[~accept, 'correct'].mean()) if (~accept).any() else float('nan'),
                    'aurc': aurc,
                })
    headline_df = pd.DataFrame(headline_rows)
    out_csv = f'{GRADCAM_DIR}/referral_test_headline_metrics.csv'
    headline_df.to_csv(out_csv, index=False)
    print(f'Pooled headline operating-point metrics -> {out_csv}')
    print(headline_df.to_string(index=False))
    return curves, headline_df

def evaluate_test_risk_coverage_perseed():
    assert Path(REFERRAL_TAU_JSON).exists(), 'Run 2.4 first.'
    test_df_all = pd.read_csv(REFERRAL_TEST_LOG)
    rows = []
    for seed in CFG['referral_test_seeds']:
        sub_seed = test_df_all[test_df_all.seed == seed]
        if len(sub_seed) == 0:
            continue
        for k_shot in sorted(sub_seed['k_shot'].unique()):
            sub = sub_seed[sub_seed.k_shot == k_shot]
            for score_col, ascending in [('predictive_variance', True), ('predictive_entropy', True), ('margin', False)]:
                coverage, risk, _ = _risk_coverage_from_scores(sub[score_col].values, sub['correct'].values, ascending)
                rows.append({'seed': seed, 'k_shot': k_shot, 'score': score_col, 'aurc': aurc_from_curve(coverage, risk)})
    perseed_df = pd.DataFrame(rows)
    out_csv = f'{GRADCAM_DIR}/referral_test_aurc_perseed.csv'
    perseed_df.to_csv(out_csv, index=False)
    print(f'Per-seed AURC -> {out_csv}')

    print('\nAURC, mean +/- 95% CI across seeds:')
    for (k_shot, score_col), g in perseed_df.groupby(['k_shot', 'score']):
        m, h, n = mean_ci(g['aurc'].values)
        if n >= 2:
            print(f'  k={k_shot} {score_col:>20s} | AURC = {m:.4f} +/- {h:.4f} (95% CI, n_seeds={n})')
        else:
            print(f'  k={k_shot} {score_col:>20s} | only {n} seed -- no CI yet')

    print('\nPaired seed-level significance: does one signal give consistently lower AURC?')
    for k_shot in sorted(perseed_df['k_shot'].unique()):
        piv = perseed_df[perseed_df.k_shot == k_shot].pivot(index='seed', columns='score', values='aurc')
        pairs = [('margin', 'predictive_variance'), ('margin', 'predictive_entropy'), ('predictive_variance', 'predictive_entropy')]
        for a, b in pairs:
            if a not in piv.columns or b not in piv.columns:
                continue
            paired = piv[[a, b]].dropna()
            if len(paired) >= 6:
                stat, p = st.wilcoxon(paired[a], paired[b])
                print(f'  k={k_shot} {a} vs {b}: Wilcoxon signed-rank stat={stat:.2f}, p={p:.4g}, n_seeds={len(paired)}')
            else:
                print(f'  k={k_shot} {a} vs {b}: only {len(paired)} paired seed(s) -- add more seeds '
                      f'to CFG["referral_test_seeds"] before this comparison has enough power '
                      f'to support a "dominates" claim in the manuscript.')
    return perseed_df


def compute_ece(confidence, correct, n_bins=15):
    confidence = np.asarray(confidence); correct = np.asarray(correct)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, bin_stats = 0.0, []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            bin_stats.append({'bin_lo': lo, 'bin_hi': hi, 'n': 0, 'acc': np.nan, 'conf': np.nan}); continue
        acc_bin = correct[mask].mean(); conf_bin = confidence[mask].mean(); weight = mask.sum() / len(confidence)
        ece += weight * abs(acc_bin - conf_bin)
        bin_stats.append({'bin_lo': lo, 'bin_hi': hi, 'n': int(mask.sum()), 'acc': float(acc_bin), 'conf': float(conf_bin)})
    return float(ece), pd.DataFrame(bin_stats)

def compute_brier_score(confidence, correct):
    confidence = np.asarray(confidence, dtype=float); correct = np.asarray(correct, dtype=float)
    return float(np.mean((confidence - correct) ** 2))

from sklearn.isotonic import IsotonicRegression
import pickle
CONFIDENCE_CALIBRATOR_PATH = f'{GRADCAM_DIR}/confidence_isotonic_calibrators.pkl'

def fit_confidence_calibrators():
    val_df = _load_seed_filtered(REFERRAL_VAL_LOG, CFG['referral_cal_seeds'], 'REFERRAL_VAL_LOG')
    calibrators = {}
    for k_shot in sorted(val_df['k_shot'].unique()):
        sub = val_df[val_df.k_shot == k_shot]
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(sub['confidence'].values, sub['correct'].values.astype(float))
        calibrators[int(k_shot)] = iso
        raw_ece, _ = compute_ece(sub['confidence'].values, sub['correct'].values)
        cal_conf = iso.predict(sub['confidence'].values)
        cal_ece, _ = compute_ece(cal_conf, sub['correct'].values)
        print(f'  k={k_shot} | VAL ECE raw={raw_ece:.4f} -> calibrated={cal_ece:.4f}')
    tmp = f'{CONFIDENCE_CALIBRATOR_PATH}.tmp'
    with open(tmp, 'wb') as f: pickle.dump(calibrators, f)
    os.replace(tmp, CONFIDENCE_CALIBRATOR_PATH)
    print(f'Confidence calibrators (VAL-only) -> {CONFIDENCE_CALIBRATOR_PATH}')
    return calibrators

def apply_confidence_calibration_and_report_pooled(calibrators):
    test_df = _load_seed_filtered(REFERRAL_TEST_LOG, CFG['referral_test_seeds'], 'REFERRAL_TEST_LOG')
    rows = []
    for k_shot in sorted(test_df['k_shot'].unique()):
        sub = test_df[test_df.k_shot == k_shot]
        iso = calibrators.get(int(k_shot))
        if iso is None:
            continue
        raw_ece, _ = compute_ece(sub['confidence'].values, sub['correct'].values)
        raw_brier = compute_brier_score(sub['confidence'].values, sub['correct'].values)
        cal_conf = iso.predict(sub['confidence'].values)
        cal_ece, _ = compute_ece(cal_conf, sub['correct'].values)
        cal_brier = compute_brier_score(cal_conf, sub['correct'].values)
        rows.append({'k_shot': k_shot, 'ece_raw': raw_ece, 'ece_calibrated': cal_ece,
                     'brier_raw': raw_brier, 'brier_calibrated': cal_brier, 'n_test_queries': len(sub)})
    out_df = pd.DataFrame(rows)
    out_csv = f'{GRADCAM_DIR}/confidence_calibration_ece_summary.csv'
    out_df.to_csv(out_csv, index=False)
    print(f'Pooled TEST calibration quality -> {out_csv}')
    print(out_df.to_string(index=False))
    return out_df

def ece_perseed(calibrators):
    test_df_all = pd.read_csv(REFERRAL_TEST_LOG)
    rows = []
    for seed in CFG['referral_test_seeds']:
        sub_seed = test_df_all[test_df_all.seed == seed]
        for k_shot in sorted(sub_seed['k_shot'].unique()) if len(sub_seed) else []:
            sub = sub_seed[sub_seed.k_shot == k_shot]
            iso = calibrators.get(int(k_shot))
            if iso is None or len(sub) == 0:
                continue
            raw_ece, _ = compute_ece(sub['confidence'].values, sub['correct'].values)
            cal_conf = iso.predict(sub['confidence'].values)
            cal_ece, _ = compute_ece(cal_conf, sub['correct'].values)
            rows.append({'seed': seed, 'k_shot': k_shot, 'ece_raw': raw_ece, 'ece_calibrated': cal_ece})
    perseed_df = pd.DataFrame(rows)
    print('ECE (calibrated), mean +/- 95% CI across seeds:')
    for k_shot, g in perseed_df.groupby('k_shot'):
        m, h, n = mean_ci(g['ece_calibrated'].values)
        print(f'  k={k_shot} | ECE_cal = {m:.4f} +/- {h:.4f} (95% CI, n_seeds={n})' if n >= 2
              else f'  k={k_shot} | only {n} seed -- no CI yet')
    return perseed_df


def calibrate_tier_thresholds():
    assert Path(REFERRAL_TAU_JSON).exists(), 'Run 2.4 (run_calibration_pooled) before this cell -- tau_2 reuses its output.'
    with open(REFERRAL_TAU_JSON) as f:
        existing_cal = json.load(f)
    val_df = _load_seed_filtered(REFERRAL_VAL_LOG, CFG['referral_cal_seeds'], 'REFERRAL_VAL_LOG')
    tiers = {}
    for k_shot in sorted(val_df['k_shot'].unique()):
        sub = val_df[val_df.k_shot == k_shot]
        tiers[str(k_shot)] = {}
        for score_col, ascending in [('predictive_variance', True), ('predictive_entropy', True), ('margin', False)]:
            tau1_result = calibrate_tau(sub, score_col, ascending, CFG['tier1_target_risk'], CFG['referral_target_coverage'])
            tau_1 = tau1_result['target_risk_rule']['tau']
            existing = existing_cal['by_k_shot'].get(str(k_shot), {}).get(score_col)
            tau_2 = existing['target_risk_rule']['tau'] if existing else None
            ok = None
            if tau_1 is not None and tau_2 is not None:
                ok = (tau_1 <= tau_2) if ascending else (tau_1 >= tau_2)
                if not ok:
                    print(f'  WARNING: k={k_shot} {score_col} -- tau_1 ({tau_1:.4f}) not stricter than '
                          f'tau_2 ({tau_2:.4f}); tier boundaries may be degenerate here.')
            tiers[str(k_shot)][score_col] = {
                'ascending_is_more_confident': ascending, 'tier1_target_risk': CFG['tier1_target_risk'],
                'tau_1': tau_1, 'tier2_target_risk': CFG['referral_target_risk'], 'tau_2': tau_2, 'ordering_ok': ok,
            }
    out_path = f'{GRADCAM_DIR}/decision_tier_thresholds.json'
    tmp = f'{out_path}.tmp'
    with open(tmp, 'w') as f: json.dump(tiers, f, indent=2)
    os.replace(tmp, out_path)
    print(f'Tier thresholds (VAL-only, pooled) -> {out_path}')
    return tiers

def apply_decision_tiers_and_report(tiers, primary_score='predictive_variance', file_suffix=None):
    suffix = file_suffix if file_suffix is not None else ''
    test_df = _load_seed_filtered(REFERRAL_TEST_LOG, CFG['referral_test_seeds'], 'REFERRAL_TEST_LOG')
    per_k_frames, tier_summary_rows = [], []
    tier_order = ['Level 1 - Confident', 'Level 2 - Uncertain', 'Level 3 - Highly uncertain']
    for k_shot in sorted(test_df['k_shot'].unique()):
        sub = test_df[test_df.k_shot == k_shot].copy()
        t = tiers.get(str(k_shot), {}).get(primary_score)
        if t is None or t['tau_1'] is None or t['tau_2'] is None:
            print(f'  k={k_shot}: tau_1/tau_2 unavailable for {primary_score} -- skipping.'); continue
        tau_1, tau_2, ascending = t['tau_1'], t['tau_2'], t['ascending_is_more_confident']
        def _tier(u, tau_1=tau_1, tau_2=tau_2, ascending=ascending):
            if ascending:
                return tier_order[0] if u < tau_1 else (tier_order[1] if u < tau_2 else tier_order[2])
            return tier_order[0] if u > tau_1 else (tier_order[1] if u > tau_2 else tier_order[2])
        sub['decision_tier'] = sub[primary_score].apply(_tier)
        per_k_frames.append(sub)
        for tier_name in tier_order:
            tsub = sub[sub.decision_tier == tier_name]
            n = len(tsub)
            tier_summary_rows.append({'k_shot': k_shot, 'tier': tier_name, 'n_queries': n,
                                       'pct_of_total': (n / len(sub) * 100) if len(sub) else float('nan'),
                                       'accuracy_in_tier': tsub['correct'].mean() if n else float('nan')})
    full_df = pd.concat(per_k_frames, ignore_index=True) if per_k_frames else pd.DataFrame()
    tier_summary_df = pd.DataFrame(tier_summary_rows)
    out_csv = f'{GRADCAM_DIR}/decision_tier_test_queries{suffix}.csv'
    summary_csv = f'{GRADCAM_DIR}/decision_tier_summary{suffix}.csv'
    full_df.to_csv(out_csv, index=False); tier_summary_df.to_csv(summary_csv, index=False)
    print(f'[{primary_score}] Per-query tier assignments -> {out_csv}')
    print(f'[{primary_score}] Tier summary               -> {summary_csv}')
    print(tier_summary_df.to_string(index=False))
    return full_df, tier_summary_df

def tier_composition_perseed(tiers, primary_score='predictive_variance'):
    test_df_all = pd.read_csv(REFERRAL_TEST_LOG)
    rows = []
    tier_order = ['Level 1 - Confident', 'Level 2 - Uncertain', 'Level 3 - Highly uncertain']
    for seed in CFG['referral_test_seeds']:
        sub_seed = test_df_all[test_df_all.seed == seed]
        for k_shot in sorted(sub_seed['k_shot'].unique()) if len(sub_seed) else []:
            sub = sub_seed[sub_seed.k_shot == k_shot].copy()
            t = tiers.get(str(k_shot), {}).get(primary_score)
            if t is None or t['tau_1'] is None or t['tau_2'] is None:
                continue
            tau_1, tau_2, ascending = t['tau_1'], t['tau_2'], t['ascending_is_more_confident']
            def _tier(u, tau_1=tau_1, tau_2=tau_2, ascending=ascending):
                if ascending:
                    return tier_order[0] if u < tau_1 else (tier_order[1] if u < tau_2 else tier_order[2])
                return tier_order[0] if u > tau_1 else (tier_order[1] if u > tau_2 else tier_order[2])
            sub['decision_tier'] = sub[primary_score].apply(_tier)
            for tier_name in tier_order:
                tsub = sub[sub.decision_tier == tier_name]
                n = len(tsub)
                rows.append({'seed': seed, 'k_shot': k_shot, 'tier': tier_name,
                             'pct_of_total': (n / len(sub) * 100) if len(sub) else float('nan'),
                             'accuracy_in_tier': tsub['correct'].mean() if n else float('nan')})
    perseed_df = pd.DataFrame(rows)
    print(f'Tier composition stability across seeds ({primary_score}):')
    for (k_shot, tier), g in perseed_df.groupby(['k_shot', 'tier']):
        pm, ph, pn = mean_ci(g['pct_of_total'].dropna().values)
        am, ah, an = mean_ci(g['accuracy_in_tier'].dropna().values)
        pct_str = f'{pm:.1f}% +/- {ph:.1f}%' if pn >= 2 else f'{pm:.1f}% (n_seeds={pn}, no CI)'
        acc_str = f'{am:.3f} +/- {ah:.3f}' if an >= 2 else f'{am:.3f} (n_seeds={an}, no CI)'
        print(f'  k={k_shot} {tier:<28s} | share={pct_str} | accuracy={acc_str}')
    return perseed_df


def referral_rate_by_class(test_df, score_col, tau, ascending=True):
    accept = (test_df[score_col] <= tau) if ascending else (test_df[score_col] >= tau)
    tmp = test_df.assign(accepted=accept)
    grp = tmp.groupby('true_class').agg(
        n=('correct', 'size'), referral_rate=('accepted', lambda s: 1 - s.mean()),
        accuracy_if_accepted=('correct', lambda s: s[tmp.loc[s.index, 'accepted']].mean() if tmp.loc[s.index, 'accepted'].any() else float('nan')),
    ).reset_index()
    return grp

def accepted_referred_confusion(test_df, score_col, tau, ascending=True, class_names=None):
    class_names = class_names if class_names is not None else REFERRAL_CLASS_NAMES
    accept = (test_df[score_col] <= tau) if ascending else (test_df[score_col] >= tau)
    accepted_df = test_df[accept]; referred_df = test_df[~accept]
    cm_accepted = pd.crosstab(accepted_df['true_class'], accepted_df['pred_class'])
    cm_referred = pd.crosstab(referred_df['true_class'], referred_df['pred_class'])
    cm_accepted = cm_accepted.reindex(index=class_names, columns=class_names, fill_value=0)
    cm_referred = cm_referred.reindex(index=class_names, columns=class_names, fill_value=0)
    return cm_accepted, cm_referred

def plot_referral_results(curves, headline_df):
    with open(REFERRAL_TAU_JSON) as f:
        calibration = json.load(f)
    test_df = _load_seed_filtered(REFERRAL_TEST_LOG, CFG['referral_test_seeds'], 'REFERRAL_TEST_LOG')
    for k_shot, per_score in curves.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        for score_col in ['predictive_variance', 'predictive_entropy', 'margin']:
            c = per_score[score_col]
            axes[0].plot(c['coverage'], c['risk'] * 100, lw=1.6, label=f"{score_col} (AURC={c['aurc']:.4f})")
        axes[0].set(title=f'Risk-Coverage Curve (k={k_shot}, pooled seeds={CFG["referral_test_seeds"]})',
                    xlabel='Coverage', ylabel='Risk (%) on accepted cases')
        axes[0].legend(fontsize=7.5); axes[0].grid(alpha=0.3)

        sub = test_df[test_df.k_shot == k_shot]
        ece, bin_df = compute_ece(sub['confidence'].values, sub['correct'].values)
        valid = bin_df.dropna()
        axes[1].bar(valid['bin_lo'], valid['acc'], width=1/15, align='edge', alpha=0.6, label='Accuracy', color='steelblue')
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
        axes[1].set(title=f'Reliability Diagram (k={k_shot}) | ECE={ece:.4f}', xlabel='Confidence', ylabel='Accuracy', xlim=(0, 1), ylim=(0, 1))
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

        cal = calibration['by_k_shot'].get(str(k_shot), {}).get('predictive_variance')
        tau = cal['target_risk_rule']['tau'] if cal else None
        if tau is not None:
            rr = referral_rate_by_class(sub, 'predictive_variance', tau, ascending=True).sort_values('referral_rate', ascending=False)
            axes[2].barh(rr['true_class'], rr['referral_rate'] * 100, color='indianred')
            axes[2].set(title=f'Referral Rate by Class (k={k_shot})', xlabel='Referral rate (%)')
            axes[2].grid(alpha=0.3, axis='x')
        else:
            axes[2].text(0.5, 0.5, 'No feasible tau at target_risk', ha='center', va='center', transform=axes[2].transAxes); axes[2].axis('off')

        fig.tight_layout()
        out_path = f'{GRADCAM_DIR}/referral_summary_k{k_shot}.png'
        tmp = f'{out_path}.tmp.png'; fig.savefig(tmp, dpi=150, bbox_inches='tight'); plt.close(fig); os.replace(tmp, out_path)
        print(f'Saved -> {out_path}')

        if tau is not None:
            cm_acc, cm_ref = accepted_referred_confusion(sub, 'predictive_variance', tau, True, class_names=REFERRAL_CLASS_NAMES)
            fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5.5))
            im = None
            for ax, cm_, title in zip(ax2, [cm_acc, cm_ref], ['Accepted (auto-decided)', 'Referred to clinician']):
                n_total = int(cm_.values.sum())
                if n_total == 0:
                    ax.text(0.5, 0.5, 'No cases in this bucket', ha='center', va='center', transform=ax.transAxes); ax.axis('off'); continue
                row_sums = cm_.values.sum(axis=1, keepdims=True)
                cm_norm = np.divide(cm_.values.astype(float), row_sums, out=np.zeros_like(cm_.values, dtype=float), where=row_sums != 0)
                im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
                ax.set_xticks(range(len(cm_.columns))); ax.set_xticklabels(cm_.columns, rotation=45, ha='right', fontsize=7)
                ax.set_yticks(range(len(cm_.index))); ax.set_yticklabels(cm_.index, fontsize=7)
                ax.set_title(f'{title} (n={n_total})', fontsize=9)
                for r in range(cm_norm.shape[0]):
                    for c in range(cm_norm.shape[1]):
                        val = cm_norm[r, c]
                        if val == 0: continue
                        ax.text(c, r, f'{val:.2f}', ha='center', va='center', fontsize=6, color='white' if val > 0.5 else '#222222')
            if im is not None:
                fig2.colorbar(im, ax=ax2, fraction=0.035, pad=0.02, label='Row-normalized rate')
            fig2.suptitle(f'Confusion Matrices -- Accepted vs. Referred (k={k_shot})', fontsize=11)
            out_path2 = f'{GRADCAM_DIR}/referral_confusion_k{k_shot}.png'
            tmp2 = f'{out_path2}.tmp.png'; fig2.savefig(tmp2, dpi=150, bbox_inches='tight'); plt.close(fig2); os.replace(tmp2, out_path2)
            print(f'Saved -> {out_path2}')


DECISION_CARD_DIR = f'{GRADCAM_DIR}/decision_cards'
os.makedirs(DECISION_CARD_DIR, exist_ok=True)

def _tier_label_and_action(u, tau_1, tau_2, ascending):
    if ascending:
        tier = 'Level 1 - Confident' if u < tau_1 else ('Level 2 - Uncertain' if u < tau_2 else 'Level 3 - Highly uncertain')
    else:
        tier = 'Level 1 - Confident' if u > tau_1 else ('Level 2 - Uncertain' if u > tau_2 else 'Level 3 - Highly uncertain')
    action = {'Level 1 - Confident': 'Auto-classify', 'Level 2 - Uncertain': 'Specialist review recommended',
              'Level 3 - Highly uncertain': 'Mandatory specialist review'}[tier]
    return tier, action

TIER_COLORS = {'Level 1 - Confident': '#2E7D32', 'Level 2 - Uncertain': '#B8860B', 'Level 3 - Highly uncertain': '#C62828'}

def render_decision_card(query_img_rgb, true_name, pred_name, correct, confidence, uncertainty_pct,
                          predictive_variance, predictive_entropy, margin_val, tier, action, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), gridspec_kw={'width_ratios': [1, 1.35]})
    axes[0].imshow(query_img_rgb); axes[0].set_title('Input frame', fontsize=10); axes[0].set_xticks([]); axes[0].set_yticks([])
    ax = axes[1]; ax.axis('off')
    status = 'CORRECT' if correct else 'INCORRECT'
    status_color = '#2E7D32' if correct else '#C62828'
    tier_color = TIER_COLORS.get(tier, '#222222')
    rows = [
        ('AI Prediction', pred_name, '#222222', False),
        ('Confidence', f'{confidence * 100:.1f}%', '#222222', False),
        ('Uncertainty (%)', f'{uncertainty_pct:.1f}%', '#B8860B', True),
        ('Predictive variance', f'{predictive_variance:.4f}', '#222222', False),
        ('Predictive entropy', f'{predictive_entropy:.4f}', '#222222', False),
        ('Margin', f'{margin_val:.4f}', '#222222', False),
        ('True class', f'{true_name}  [{status}]', status_color, False),
        ('Risk category', tier, tier_color, True),
        ('Decision', action, tier_color, True),
    ]
    y = 0.95
    for label, value, color, bold in rows:
        ax.text(0.0, y, f'{label}:', fontsize=9.5, fontweight='bold', transform=ax.transAxes)
        ax.text(0.58, y, value, fontsize=9.5, color=color, transform=ax.transAxes, fontweight='bold' if bold else 'normal')
        y -= 0.105
    fig.suptitle('Uncertainty-Aware Clinical Decision Support -- Example Output', fontsize=10.5, fontweight='bold')
    fig.tight_layout()
    tmp = f'{save_path}.tmp.png'; fig.savefig(tmp, dpi=150, bbox_inches='tight'); plt.close(fig); os.replace(tmp, save_path)

def build_decision_card_gallery(model, decision_tiers, primary_score='predictive_variance', k_shot=None, seed=None, n_episodes=6):
    k_shot = k_shot or CFG['cam_k_shot']
    seed = seed if seed is not None else CFG['referral_cal_seeds'][0]
    n_way = REFERRAL_N_WAY
    t = decision_tiers.get(str(k_shot), {}).get(primary_score)
    assert t is not None and t['tau_1'] is not None and t['tau_2'] is not None, (
        f'No feasible tau_1/tau_2 for k={k_shot}, {primary_score} -- run 2.7 first.')
    tau_1, tau_2, ascending = t['tau_1'], t['tau_2'], t['ascending_is_more_confident']
    index_rows = []
    for ep_idx in range(n_episodes):
        sup, sup_lbl, qry, qry_lbl, qry_class_names, _ep_class_ids = sample_referral_episode(
            TEST_SAMPLES_BY_LOCAL, n_way, k_shot, N_QUERY, seed, episode_idx=200_000 + ep_idx)
        sup_dev, sup_lbl_dev = sup.to(DEVICE), sup_lbl.to(DEVICE)
        qry_dev = qry.to(DEVICE)
        pred, mean_p, unc = model.predict_with_uncertainty(sup_dev, sup_lbl_dev, qry_dev, n_way, n_passes=CFG['mc_passes'])
        entropy = -(mean_p * mean_p.clamp_min(1e-8).log()).sum(-1)
        confidence = mean_p.max(-1).values
        unc_pct_all = 100.0 * entropy / np.log(n_way)
        model.eval()
        with torch.no_grad():
            det_logits = model(sup_dev, sup_lbl_dev, qry_dev, n_way)
            top2 = det_logits.topk(2, dim=-1).values
            margin_vals = top2[:, 0] - top2[:, 1]
        score_vals = {'predictive_variance': unc, 'predictive_entropy': entropy, 'margin': -margin_vals}[primary_score]
        target_local_class = ep_idx % n_way
        class_positions = [i for i, lbl in enumerate(qry_lbl.tolist()) if lbl == target_local_class]
        pick_rng = _rand.Random(seed * 7919 + ep_idx)
        qi = pick_rng.choice(class_positions)
        u_val = float(score_vals[qi].item())
        tier, action = _tier_label_and_action(u_val, tau_1, tau_2, ascending)
        pred_idx = int(pred[qi].item())
        pred_name = REFERRAL_CLASS_NAMES[pred_idx]
        true_name = qry_class_names[qi]
        correct = int(pred_name == true_name)
        card_path = f'{DECISION_CARD_DIR}/card_k{k_shot}_ep{ep_idx}_q{qi}.png'
        render_decision_card(tensor_to_rgb(qry[qi]), true_name, pred_name, correct,
                              float(confidence[qi].item()), float(unc_pct_all[qi].item()),
                              float(unc[qi].item()), float(entropy[qi].item()), float(margin_vals[qi].item()),
                              tier, action, card_path)
        index_rows.append({'k_shot': k_shot, 'episode_idx': 200_000 + ep_idx, 'query_idx': qi,
                            'true_class': true_name, 'pred_class': pred_name, 'correct': correct,
                            'confidence': float(confidence[qi].item()), 'uncertainty_pct': float(unc_pct_all[qi].item()),
                            'predictive_variance': float(unc[qi].item()), 'predictive_entropy': float(entropy[qi].item()),
                            'margin': float(margin_vals[qi].item()), 'decision_tier': tier, 'decision_action': action,
                            'figure_path': card_path})
    index_df = pd.DataFrame(index_rows)
    n_unique_classes = index_df['true_class'].nunique()
    if n_unique_classes < min(n_episodes, n_way):
        print(f'  WARNING: gallery shows only {n_unique_classes} distinct true class(es) across '
              f'{n_episodes} cards -- expected up to {min(n_episodes, n_way)}.')
    index_csv = f'{DECISION_CARD_DIR}/decision_card_index.csv'
    index_df.to_csv(index_csv, index=False)
    print(f'{len(index_rows)} visual decision cards -> {DECISION_CARD_DIR}/ ({n_unique_classes} distinct true classes)')
    print(index_df[['episode_idx', 'query_idx', 'true_class', 'pred_class', 'correct', 'confidence',
                     'uncertainty_pct', 'decision_tier']].to_string(index=False))
    return index_df


from tqdm.auto import tqdm

print(f'[A] Referral evaluation | REFERRAL_N_WAY={REFERRAL_N_WAY} | k_shots={K_SHOTS} | '
      f'cal_seeds={CFG["referral_cal_seeds"]} x {CFG["referral_cal_episodes"]} ep | '
      f'test_seeds={CFG["referral_test_seeds"]} x {CFG["referral_test_episodes"]} ep')

run_referral_sweep(protonet, VAL_SAMPLES_BY_LOCAL, 'val', REFERRAL_VAL_LOG,
                    CFG['referral_cal_seeds'], CFG['referral_cal_episodes'], K_SHOTS, REFERRAL_N_WAY)
run_referral_sweep(protonet, TEST_SAMPLES_BY_LOCAL, 'test', REFERRAL_TEST_LOG,
                    CFG['referral_test_seeds'], CFG['referral_test_episodes'], K_SHOTS, REFERRAL_N_WAY)

print('\n[B] Calibration (VAL-only) -- pooled (deployed) + per-seed (CI)')
pooled_cal = run_calibration_pooled()
perseed_cal = run_calibration_perseed()

print('\n[C] TEST evaluation -- pooled headline metrics + per-seed AURC & significance')
curves, headline_df = evaluate_test_risk_coverage_pooled()
aurc_perseed_df = evaluate_test_risk_coverage_perseed()

print('\n[D] Confidence calibration (ECE/Brier + isotonic) -- pooled + per-seed')
calibrators = fit_confidence_calibrators()
ece_summary_df = apply_confidence_calibration_and_report_pooled(calibrators)
ece_perseed_df = ece_perseed(calibrators)

print('\n[E] Aggregate figures')
plot_referral_results(curves, headline_df)

print('\n[F] Three-level decision layer -- variance AND margin signals, pooled + per-seed composition')
tiers = calibrate_tier_thresholds()
_, summary_variance_df = apply_decision_tiers_and_report(tiers, primary_score='predictive_variance', file_suffix='')
_, summary_margin_df = apply_decision_tiers_and_report(tiers, primary_score='margin', file_suffix='_margin')
tier_composition_perseed(tiers, primary_score='predictive_variance')

print('\n[G] Visual decision-card gallery')
decision_card_index = build_decision_card_gallery(protonet, tiers, primary_score='predictive_variance', n_episodes=6)

print('\nPart 2 complete.')

# Cell 6 - ## PART 3 — Evaluation Protocol Audit & Manuscript-Ready Summary.

def evaluation_protocol_audit():
    print('=' * 78)
    print('EVALUATION PROTOCOL AUDIT')
    print('=' * 78)

    print('\n[Grad-CAM / Uncertainty, Part 1]')
    if Path(GRADCAM_LOG).exists():
        g = pd.read_csv(GRADCAM_LOG)
        print(f'  seeds logged        : {sorted(g.seed.unique().tolist())}')
        print(f'  total queries logged: {len(g)}')
        print(f'  n_way / k_shot       : {CFG["cam_n_way"]} / {CFG["cam_k_shot"]}')
        print(f'  queries per class    : {CFG["cam_n_query_per_class"]}')
        print(f'  MC passes (unc.)     : {CFG["mc_passes"]}  |  MC passes (CAM): {CFG["mc_passes_cam"]}')
    else:
        print('  NOT YET RUN -- execute Part 1 (1.7) before citing any Grad-CAM/uncertainty number.')

    print('\n[Referral / Decision Support, Part 2]')
    if Path(REFERRAL_VAL_LOG).exists() and Path(REFERRAL_TEST_LOG).exists():
        v = pd.read_csv(REFERRAL_VAL_LOG); t = pd.read_csv(REFERRAL_TEST_LOG)
        v_cfg = v[v.seed.isin(CFG['referral_cal_seeds'])]
        t_cfg = t[t.seed.isin(CFG['referral_test_seeds'])]
        print(f'  cal seeds (config)   : {CFG["referral_cal_seeds"]}  | episodes/seed: {CFG["referral_cal_episodes"]}')
        print(f'  cal seeds (on disk)  : {sorted(v.seed.unique().tolist())}')
        print(f'  test seeds (config)  : {CFG["referral_test_seeds"]}  | episodes/seed: {CFG["referral_test_episodes"]}')
        print(f'  test seeds (on disk) : {sorted(t.seed.unique().tolist())}')
        print(f'  VAL rows used (config-filtered)  : {len(v_cfg)} / {len(v)} on disk')
        print(f'  TEST rows used (config-filtered) : {len(t_cfg)} / {len(t)} on disk')
        if len(v_cfg) < len(v) or len(t_cfg) < len(t):
            print('  NOTE: rows on disk outside the configured seed lists exist (from a prior run) '
                  'and are excluded above by _load_seed_filtered -- not deleted, not double-counted.')
    else:
        print('  NOT YET RUN -- execute Part 2 (2.10) before citing any referral/decision-tier number.')

    print('\n[Leakage control]')
    print('  - Threshold/calibrator fitting (2.4, 2.6, 2.7) reads ONLY REFERRAL_VAL_LOG.')
    print('  - Headline metrics / tiering / confusion matrices (2.5, 2.8) read ONLY REFERRAL_TEST_LOG.')
    print('  - VAL and TEST are disjoint splits fixed in split_indices.json (Part 0.4), and')
    print('    Grad-CAM (Part 1) and referral TEST (Part 2) both draw exclusively from the same')
    print('    already-isolated in-domain TEST pool (INDOMAIN_SAMPLES) -- no split is shared')
    print('    between calibration and reporting at any point in this notebook.')

    print('\n[Statistical tests actually performed this run]')
    print('  - Uncertainty(%) correct vs incorrect: Mann-Whitney U (pooled queries, 1.6)')
    print('    + Wilcoxon signed-rank across seeds (1.6, requires >=6 seeds for a valid p-value --')
    print('    check the printed n_seeds in 1.6\'s output before citing that p-value).')
    print('  - Uncertainty-signal comparison (variance/entropy/margin) on AURC: Wilcoxon')
    print('    signed-rank across seeds (2.5, same >=6-seed caveat).')
    print('  - All other reported spreads are seed-level 95% CIs (t-distribution, mean_ci()),')
    print('    NOT hypothesis tests -- report them as such, not as significance claims.')

    print('\n[Known limitations to state explicitly in the manuscript, not just here]')
    print(f'  - dropout_rate={CFG["dropout_rate"]}: predictive_variance is correctly ranked but')
    print('    numerically compressed (not interpretable in absolute magnitude on its own);')
    print('    uncertainty_pct (normalized entropy) is the percentage-scale figure to cite instead.')
    print(f'  - Seed counts are small (cal={len(CFG["referral_cal_seeds"])}, test={len(CFG["referral_test_seeds"])}, ')
    print(f'    cam={len(CFG["cam_seeds"])}) -- typical of a T4-budget-constrained sweep. Any seed-level')
    print('    significance test above is correspondingly low-powered; report effect sizes/CIs')
    print('    alongside p-values, and state the T4 compute constraint explicitly in Discussion/Limitations')
    print('    (do not silently omit it, and do not overstate power the seed count does not support).')
    print('=' * 78)

evaluation_protocol_audit()


def build_manuscript_summary_tables():
    out_paths = []

    if Path(f'{GRADCAM_DIR}/referral_test_aurc_perseed.csv').exists():
        aurc_ps = pd.read_csv(f'{GRADCAM_DIR}/referral_test_aurc_perseed.csv')
        rows = []
        for (k_shot, score), g in aurc_ps.groupby(['k_shot', 'score']):
            m, h, n = mean_ci(g['aurc'].values)
            rows.append({'k_shot': k_shot, 'score': score, 'aurc_mean': m, 'aurc_ci95_halfwidth': h, 'n_seeds': n})
        out = pd.DataFrame(rows)
        p = f'{GRADCAM_DIR}/referral_test_aurc_manuscript.csv'
        out.to_csv(p, index=False); out_paths.append(p)

    if Path(GRADCAM_LOG).exists():
        g = pd.read_csv(GRADCAM_LOG)
        per_seed_mean = g.groupby('seed')['uncertainty_pct'].mean()
        m, h, n = mean_ci(per_seed_mean.values)
        out = pd.DataFrame([{'uncertainty_pct_mean': m, 'uncertainty_pct_ci95_halfwidth': h, 'n_seeds': n,
                             'n_total_queries': len(g)}])
        p = f'{GRADCAM_DIR}/gradcam_uncertainty_pct_manuscript.csv'
        out.to_csv(p, index=False); out_paths.append(p)

    print('Manuscript-ready summary tables written:')
    for p in out_paths:
        print(f'  -> {p}')
    if not out_paths:
        print('  Nothing to summarize yet -- run Parts 1 and 2 first.')
    return out_paths

build_manuscript_summary_tables()
