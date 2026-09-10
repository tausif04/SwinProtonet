# Cell 0 - GPU check and Google Drive mount.
import os, gc, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
gc.collect(); torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f'GPU  : {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info()
    print(f'VRAM : {free/1024**3:.1f} / {total/1024**3:.1f} GiB free')
else:
    raise RuntimeError('No GPU -- switch to a GPU runtime in Colab.')

from google.colab import drive
drive.mount('/content/drive')

# Cell 1 - Install required Python packages.
import subprocess, sys
for pkg in ['timm==0.9.16', 'albumentations>=1.3.0', 'einops', 'tqdm',
            'scikit-learn', 'kagglehub']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
print('packages ready')

# Cell 2 - Import dependencies, set seeds, and configure evaluation.
import zipfile, random, json, hashlib, csv, warnings
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
from scipy import stats as sp_stats
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda')

CFG = {
    'output_dir'    : '/content/drive/MyDrive/SSL_FYDP/Outputs',
    'ssl_ckpt'       : '/content/drive/MyDrive/SSL_FYDP/Outputs/swin_ssl_pretrained.pth',
    'proto_ckpt'     : '/content/drive/MyDrive/SSL_FYDP/Outputs/protonet_best.pth',
    'split_file'     : '/content/drive/MyDrive/SSL_FYDP/Outputs/split_indices.json',
    'kvasir_v2_dir'  : '/content/drive/MyDrive/SSL_FYDP/kvasir-dataset-v2',
    'backbone'       : 'swin_tiny_patch4_window7_224',
    'embed_dim'      : 768,
    'img_size'       : 224,
    'dropout_rate'   : 0.1,
    'temperature'    : 0.5,
    'mc_passes'      : 20,
}
SIZE = CFG['img_size']

ABLATION_DIR = f"{CFG['output_dir']}/ablation_ladder"

DATASET_A_KEY   = 'kvasir_capsule'
DATASET_A_KAGGLE = 'mdtausifbinmozid/kvasir-capsule'
DATASET_A_KEYWORDS = {
    'polyp'  : ['polyp'],
    'ulcer'  : ['ulcer'],
    'pylorus': ['pylorus'],
    'normal' : ['normalmucosa', 'normal-mucosa', 'normal_mucosa', 'normal mucosa'],
}

DATASET_B_KEY    = 'seeai'
DATASET_B_KAGGLE  = 'capsuleyolo/kyucapsule'
DATASET_B_KEYWORDS = {
    'polyp' : ['polyp'],
    'ulcer' : ['erosion', 'ulcer'],
    'normal': ['normal'],
}

EVAL_DIR    = f"{CFG['output_dir']}/crossdataset_validation"
os.makedirs(EVAL_DIR, exist_ok=True)
METRICS_CSV = f'{EVAL_DIR}/crossdataset_episode_metrics.csv'
PRED_LOG    = f'{EVAL_DIR}/crossdataset_predictions.jsonl'

K_SHOTS        = [1, 5]
N_QUERY        = 15
FULL_SEEDS     = [42, 43, 44, 45, 46]
FULL_EPISODES  = 600
SWEEP_SEEDS    = [42, 43, 44]
SWEEP_EPISODES = 150

print(f'Metrics log     : {METRICS_CSV}')
print(f'Predictions log : {PRED_LOG}')
print(f'Ablation ckpts  : {ABLATION_DIR}  (optional -- checked in Cell 5)')
print(f'Dataset A       : {DATASET_A_KEY}  ({DATASET_A_KAGGLE}) -- SSL encoder HAS seen this collection')
print(f'Dataset B       : {DATASET_B_KEY}  ({DATASET_B_KAGGLE}) -- SSL encoder has NOT seen this collection')

# Cell 3 - Verify the held-out split provenance.
with open(CFG['split_file']) as f:
    split_data = json.load(f)
test_idx  = sorted(split_data['test'])
test_hash = hashlib.sha256(json.dumps(test_idx).encode()).hexdigest()
print(f'test_idx count  : {len(test_idx)}')
print(f'test_idx sha256 : {test_hash}')
print('Compare against Part 3b Cell 3 -- must match exactly.')
print('(Not used to build held-out episodes below.)')

# Cell 4 - Define the SwinProtoNet model.
class SwinProtoNet(nn.Module):
    def __init__(self, init_mode='ssl_full'):
        super().__init__()
        assert init_mode in ('random', 'imagenet', 'ssl_lp', 'ssl_full')
        self.init_mode = init_mode
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

print('SwinProtoNet(init_mode) defined')

# Cell 5 - Load the main model and optional ablation checkpoints.
VARIANT_CKPTS = {
    'ssl_full': CFG['proto_ckpt'],
    'random'  : f'{ABLATION_DIR}/random_best.pth',
    'imagenet': f'{ABLATION_DIR}/imagenet_best.pth',
    'ssl_lp'  : f'{ABLATION_DIR}/ssl_lp_best.pth',
}

assert Path(VARIANT_CKPTS['ssl_full']).exists(), (
    f"Main checkpoint missing: {VARIANT_CKPTS['ssl_full']}\n"
    f"Run Part 2 first -- this notebook cannot proceed without the trained thesis model.")

models_to_eval = {}
for mode, path in VARIANT_CKPTS.items():
    p = Path(path)
    if not p.exists():
        tag = 'REQUIRED' if mode == 'ssl_full' else 'optional'
        print(f'[{mode}] ({tag}) checkpoint NOT FOUND at {path} -- skipping.')
        continue
    m = SwinProtoNet(mode).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    m.eval()
    models_to_eval[mode] = m
    print(f'[{mode}] loaded from {path}')

assert 'ssl_full' in models_to_eval, 'Main model failed to load -- cannot continue.'

ABLATION_VARIANTS = [v for v in ('random', 'imagenet', 'ssl_lp') if v in models_to_eval]
ABLATION_READY = len(ABLATION_VARIANTS) == 3
if ABLATION_READY:
    print(f'\nAll 3 ablation-ladder checkpoints found -- comparison runs for all 4 arms.')
elif ABLATION_VARIANTS:
    print(f'\nOnly partial ablation ladder found ({ABLATION_VARIANTS}) -- comparisons below '
          f'will be flagged PARTIAL.')
else:
    print(f'\nNo ablation-ladder checkpoints found under {ABLATION_DIR}.')
    print('Head-to-head dataset comparison will still run for ssl_full alone.')

# Cell 6 - Discover and build the held-out class sets.
import kagglehub

def norm(s):
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')

def keyword_match(folder_name_norm, keyword_map):
    for label, kws in keyword_map.items():
        for kw in kws:
            if norm(kw) in folder_name_norm:
                return label
    return None

def scan_dataset_by_keywords(root, keyword_map):
    """Walk every image under root; match its nearest matching ancestor
    folder to a canonical label via substring keywords. Returns
    {canonical_label: {actual_folder_name: [Path, ...]}} so the real,
    discovered folder names are always inspectable, never assumed."""
    found = {}
    for p in Path(root).rglob('*'):
        if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        for parent in p.parents:
            label = keyword_match(norm(parent.name), keyword_map)
            if label is not None:
                found.setdefault(label, {}).setdefault(parent.name, []).append(p)
                break
    return found

def summarize_and_filter(found, min_needed, dataset_label):
    print(f'--- {dataset_label} ---')
    kept, dropped = {}, {}
    for label, folders in found.items():
        imgs = [p for plist in folders.values() for p in plist]
        folder_names = sorted(folders.keys())
        print(f'  {label:<10} <- folders {folder_names} : {len(imgs)} images')
        if len(imgs) >= min_needed:
            kept[label] = imgs
        else:
            dropped[label] = len(imgs)
    if dropped:
        print(f'  Dropped (below {min_needed} images): {dropped}')
    if not found:
        print('  NOTHING MATCHED -- check the dataset actually downloaded and keyword_map.')
    return kept

MAX_SHOT   = max(K_SHOTS)
MIN_NEEDED = MAX_SHOT + N_QUERY + 10

CAPSULE_DIR = kagglehub.dataset_download(DATASET_A_KAGGLE)
print('Dataset A root:', CAPSULE_DIR)
found_a = scan_dataset_by_keywords(CAPSULE_DIR, DATASET_A_KEYWORDS)
kept_a  = summarize_and_filter(found_a, MIN_NEEDED, 'Dataset A (Kvasir-Capsule)')

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

kv2_root    = resolve_kv2_root(CFG)
kv2_classes = {norm(d.name) for d in kv2_root.iterdir() if d.is_dir()}
overlap_a   = {norm(k) for k in kept_a} & kv2_classes
assert not overlap_a, f'Dataset A held-out labels overlap Kvasir v2 training classes: {overlap_a}'
print('Dataset A confirmed disjoint from Kvasir v2 training class names\n')

def build_samples_dict(kept):
    class_names  = sorted(kept.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    samples = {class_to_idx[c]: sorted(kept[c], key=str) for c in class_names}
    return class_names, samples

DATASET_A_CLASS_NAMES, DATASET_A_SAMPLES = build_samples_dict(kept_a)
N_WAY_MAX_A = len(DATASET_A_CLASS_NAMES)
assert N_WAY_MAX_A >= 2, 'Not enough Dataset A held-out classes with sufficient images'
print(f'N_WAY_MAX  Dataset A (Kvasir-Capsule) = {N_WAY_MAX_A}  classes: {DATASET_A_CLASS_NAMES}')

N_WAY_COMPARE = N_WAY_MAX_A
print(f'\nN_WAY_COMPARE = {N_WAY_COMPARE}')

DATASETS = {
    DATASET_A_KEY: {'label': 'Kvasir-Capsule', 'class_names': DATASET_A_CLASS_NAMES,
                    'samples': DATASET_A_SAMPLES, 'seen_by_ssl_encoder': True},
}

N_WAYS_SWEEP = list(range(2, N_WAY_MAX_A + 1))
print(f'\nDataset-A-only N-way sweep points (ssl_full, extra characterization): {N_WAYS_SWEEP}')

est_sweep   = len(K_SHOTS) * len(FULL_SEEDS) * FULL_EPISODES \
              + max(0, len(N_WAYS_SWEEP) - 1) * len(K_SHOTS) * len(SWEEP_SEEDS) * SWEEP_EPISODES
n_variants  = len(models_to_eval)
est_fixed_n = n_variants * len(K_SHOTS) * len(FULL_SEEDS) * FULL_EPISODES
print(f'Dataset-A-only sweep episodes (ssl_full)                      : {est_sweep}')
print(f'Dataset-A fixed-N episodes ({n_variants} variant(s) x {len(K_SHOTS)} k-shots) : {est_fixed_n}')
print(f'Total target episodes                                     : {est_sweep + est_fixed_n}')

# Cell 7 - Define dataset-agnostic episode sampling.
val_tfm = A.Compose([
    A.Resize(SIZE, SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def load_image(path):
    img = np.array(Image.open(path).convert('RGB'))
    return val_tfm(image=img)['image']

def pick_classes(samples, n_way, seed, k_shot, episode_idx):
    n_available = len(samples)
    if n_way == n_available:
        return list(range(n_available))
    rng = random.Random(seed * 10_000_000 + n_way * 1_000_000 +
                         k_shot * 10_000 + episode_idx)
    return rng.sample(range(n_available), n_way)

def sample_episode(samples, n_way, k_shot, seed, episode_idx):
    rng = random.Random(seed * 1_000_000 + n_way * 100_000 +
                         k_shot * 10_000 + episode_idx)
    class_ids = pick_classes(samples, n_way, seed, k_shot, episode_idx)
    sup_imgs, sup_lbls, qry_imgs, qry_lbls = [], [], [], []
    for local, cls in enumerate(class_ids):
        pool   = samples[cls]
        chosen = rng.sample(pool, k_shot + N_QUERY)
        for path in chosen[:k_shot]:
            sup_imgs.append(load_image(path)); sup_lbls.append(local)
        for path in chosen[k_shot:]:
            qry_imgs.append(load_image(path)); qry_lbls.append(local)
    return (torch.stack(sup_imgs), torch.tensor(sup_lbls),
            torch.stack(qry_imgs), torch.tensor(qry_lbls))

print('Episode sampler ready (dataset-agnostic -- takes samples/n_way explicitly)')

# Cell 8 - Run the resumable cross-dataset evaluation.
def completed_episodes():
    done = set()
    if Path(METRICS_CSV).exists():
        with open(METRICS_CSV) as f:
            for row in csv.DictReader(f):
                done.add((row['dataset'], row['variant'], int(row['n_way']),
                          int(row['k_shot']), int(row['seed']), int(row['episode_idx'])))
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

def run_episode(model, dataset_key, samples, variant, n_way, k_shot, seed,
                 episode_idx, log_predictions):
    sup, sup_lbl, qry, qry_lbl = sample_episode(samples, n_way, k_shot, seed, episode_idx)
    sup, sup_lbl = sup.to(DEVICE), sup_lbl.to(DEVICE)
    qry, qry_lbl = qry.to(DEVICE), qry_lbl.to(DEVICE)

    with torch.no_grad():
        logits = model(sup, sup_lbl, qry, n_way)
        probs  = F.softmax(logits / CFG['temperature'], dim=-1)

    y_true = qry_lbl.cpu().numpy()
    y_pred = probs.argmax(-1).cpu().numpy()

    acc      = float((y_pred == y_true).mean())
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)

    append_metrics_row({
        'dataset': dataset_key, 'variant': variant, 'n_way': n_way, 'k_shot': k_shot,
        'seed': seed, 'episode_idx': episode_idx,
        'accuracy': acc, 'macro_f1': macro_f1, 'balanced_acc': bal_acc,
    })
    if log_predictions:
        append_prediction_log({
            'dataset': dataset_key, 'variant': variant, 'n_way': n_way, 'k_shot': k_shot,
            'seed': seed, 'episode_idx': episode_idx,
            'y_true': y_true.tolist(), 'y_pred': y_pred.tolist(),
            'y_prob': probs.cpu().numpy().tolist(),
        })

done = completed_episodes()
print(f'Episodes already completed: {len(done)}')

from tqdm.auto import tqdm

ssl_model = models_to_eval['ssl_full']
ssl_model.eval()
for n_way in N_WAYS_SWEEP:
    seeds, n_episodes = (FULL_SEEDS, FULL_EPISODES) if n_way == N_WAY_MAX_A \
                        else (SWEEP_SEEDS, SWEEP_EPISODES)
    for k_shot in K_SHOTS:
        for seed in seeds:
            for episode_idx in tqdm(range(n_episodes),
                                     desc=f'[sweep|A|ssl_full] N={n_way} k={k_shot} seed={seed}'):
                if (DATASET_A_KEY, 'ssl_full', n_way, k_shot, seed, episode_idx) in done:
                    continue
                run_episode(ssl_model, DATASET_A_KEY, DATASET_A_SAMPLES, 'ssl_full',
                            n_way, k_shot, seed, episode_idx,
                            log_predictions=(n_way == N_WAY_MAX_A))

for dataset_key, dcfg in DATASETS.items():
    for variant, model in models_to_eval.items():
        model.eval()
        for k_shot in K_SHOTS:
            for seed in FULL_SEEDS:
                for episode_idx in tqdm(range(FULL_EPISODES),
                                         desc=f'[fixed-N|{dataset_key}|{variant}] '
                                              f'N={N_WAY_COMPARE} k={k_shot} seed={seed}'):
                    if (dataset_key, variant, N_WAY_COMPARE, k_shot, seed, episode_idx) in done:
                        continue
                    run_episode(model, dataset_key, dcfg['samples'], variant,
                                N_WAY_COMPARE, k_shot, seed, episode_idx,
                                log_predictions=True)

print('\nEvaluation sweep complete')

# Cell 9 - Aggregate episode statistics and perform comparisons.
import pandas as pd

df = pd.read_csv(METRICS_CSV)

sweep_rows = []
for n_way in N_WAYS_SWEEP:
    for k_shot in K_SHOTS:
        sub = df[(df['dataset'] == DATASET_A_KEY) & (df['variant'] == 'ssl_full') &
                 (df['n_way'] == n_way) & (df['k_shot'] == k_shot)]
        if len(sub) == 0:
            continue
        acc  = sub['accuracy'].values
        ci95 = 1.96 * acc.std(ddof=1) / np.sqrt(len(acc)) if len(acc) > 1 else float('nan')
        sweep_rows.append({'n_way': n_way, 'k_shot': k_shot, 'n_episodes': len(acc),
                           'mean_accuracy': acc.mean(), 'ci95': ci95,
                           'mean_macro_f1': sub['macro_f1'].mean(),
                           'mean_balanced_acc': sub['balanced_acc'].mean()})
sweep_summary_df = pd.DataFrame(sweep_rows)
sweep_summary_df.to_csv(f'{EVAL_DIR}/crossdataset_A_sweep_summary.csv', index=False)
print('Dataset A only -- N-way x K-shot sweep (ssl_full):')
print(sweep_summary_df.to_string(index=False))

h2h_rows = []
for dataset_key in DATASETS:
    for variant in models_to_eval:
        for k_shot in K_SHOTS:
            sub = df[(df['dataset'] == dataset_key) & (df['variant'] == variant) &
                     (df['n_way'] == N_WAY_COMPARE) & (df['k_shot'] == k_shot)]
            acc = sub['accuracy'].values
            if len(acc) == 0:
                continue
            ci95 = 1.96 * acc.std(ddof=1) / np.sqrt(len(acc)) if len(acc) > 1 else float('nan')
            h2h_rows.append({'dataset': dataset_key, 'variant': variant, 'k_shot': k_shot,
                             'n_episodes': len(acc), 'mean_accuracy': acc.mean(), 'ci95': ci95,
                             'mean_macro_f1': sub['macro_f1'].mean(),
                             'mean_balanced_acc': sub['balanced_acc'].mean()})
h2h_df = pd.DataFrame(h2h_rows)
h2h_df.to_csv(f'{EVAL_DIR}/crossdataset_head2head_summary.csv', index=False)
print(f'\nDataset-A fixed-N summary (N={N_WAY_COMPARE}):')
print(h2h_df.to_string(index=False))

if ABLATION_VARIANTS:
    ab_sig_rows = []
    for dataset_key in DATASETS:
        for k_shot in K_SHOTS:
            ref = df[(df['dataset']==dataset_key) & (df['variant']=='ssl_full') &
                     (df['n_way']==N_WAY_COMPARE) & (df['k_shot']==k_shot)]['accuracy'].values
            for variant in ABLATION_VARIANTS:
                comp = df[(df['dataset']==dataset_key) & (df['variant']==variant) &
                          (df['n_way']==N_WAY_COMPARE) & (df['k_shot']==k_shot)]['accuracy'].values
                n = min(len(ref), len(comp))
                stat, p = sp_stats.mannwhitneyu(ref[:n], comp[:n], alternative='greater')
                ab_sig_rows.append({'dataset': dataset_key, 'k_shot': k_shot,
                                    'comparison': f'ssl_full > {variant}',
                                    'n_episodes_each': n, 'mann_whitney_u': stat, 'p_value': p})
    ab_sig_df = pd.DataFrame(ab_sig_rows)
    ab_sig_df.to_csv(f'{EVAL_DIR}/crossdataset_ablation_significance.csv', index=False)
    print('\nAblation significance (ssl_full vs each variant, per dataset):')
    print(ab_sig_df.to_string(index=False))
else:
    print('\n[TABLE PENDING -- ABLATION CHECKPOINTS NOT FOUND] Ablation significance skipped.')

# Cell 10 - Compute pooled confusion matrices, per-class metrics, and AUC.
pooled = {}
with open(PRED_LOG) as f:
    for line in f:
        r = json.loads(line)
        if r['n_way'] != N_WAY_COMPARE or r['dataset'] not in DATASETS:
            continue
        key = (r['dataset'], r['variant'], r['k_shot'])
        d = pooled.setdefault(key, {'y_true': [], 'y_pred': [], 'y_prob': []})
        d['y_true'].extend(r['y_true']); d['y_pred'].extend(r['y_pred']); d['y_prob'].extend(r['y_prob'])

per_class_records   = []
auc_combined_rows   = []
confusion_matrices  = {}

for (dataset_key, variant, k_shot), d in pooled.items():
    y_true = np.array(d['y_true']); y_pred = np.array(d['y_pred']); y_prob = np.array(d['y_prob'])
    class_names = DATASETS[dataset_key]['class_names']

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_WAY_COMPARE)))
    if variant == 'ssl_full':
        confusion_matrices[(dataset_key, k_shot)] = cm

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', labels=list(range(N_WAY_COMPARE)))
    except ValueError:
        auc = float('nan')
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    combined = (auc + bal_acc) / 2 if not np.isnan(auc) else float('nan')
    auc_combined_rows.append({'dataset': dataset_key, 'variant': variant, 'k_shot': k_shot,
                              'auc': auc, 'balanced_acc': bal_acc,
                              'combined_metric_UNVERIFIED': combined})

    for c in range(N_WAY_COMPARE):
        yt = (y_true == c).astype(int); yp = (y_pred == c).astype(int)
        tn = int(((yt == 0) & (yp == 0)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        per_class_records.append({
            'dataset': dataset_key, 'variant': variant, 'k_shot': k_shot,
            'class': class_names[c],
            'precision': precision_score(yt, yp, zero_division=0),
            'recall_sensitivity': recall_score(yt, yp, zero_division=0),
            'specificity': specificity, 'f1': f1_score(yt, yp, zero_division=0),
        })

per_class_df = pd.DataFrame(per_class_records)
per_class_df.to_csv(f'{EVAL_DIR}/crossdataset_per_class_metrics.csv', index=False)
auc_combined_df = pd.DataFrame(auc_combined_rows)
auc_combined_df.to_csv(f'{EVAL_DIR}/crossdataset_auc_combined.csv', index=False)

print(f'Per-class metrics, N={N_WAY_COMPARE}, Dataset A and variant:')
print(per_class_df.to_string(index=False))
print('\nAUC / balanced-acc / combined metric (UNVERIFIED), Dataset A, variant, k-shot:')
print(auc_combined_df.to_string(index=False))

# Cell 11 - Generate publication-style evaluation figures.
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family'      : 'serif', 'font.serif'  : ['Times New Roman', 'DejaVu Serif'],
    'font.size'        : 9,   'axes.linewidth'   : 0.8, 'axes.edgecolor'  : '#3A3A3A',
    'axes.labelcolor'  : '#222222', 'text.color' : '#222222',
    'xtick.color'      : '#3A3A3A', 'ytick.color': '#3A3A3A',
    'axes.grid'        : True, 'grid.alpha'      : 0.25, 'grid.linewidth': 0.5,
    'legend.frameon'   : False, 'savefig.dpi'    : 300,  'figure.dpi'    : 150,
    'axes.spines.top'  : False, 'axes.spines.right': False,
})
PALETTE    = ['#4C72B0', '#DD8452', '#55A868', '#C44E52',
              '#8172B2', '#937860', '#64B5CD', '#CCB974']
FIG_SINGLE = (3.45, 2.6)
FIG_WIDE   = (7.16, 3.0)

fig, ax = plt.subplots(figsize=FIG_SINGLE)
for i, k_shot in enumerate(K_SHOTS):
    sub = sweep_summary_df[sweep_summary_df['k_shot'] == k_shot].sort_values('n_way')
    ax.errorbar(sub['n_way'], sub['mean_accuracy'] * 100, yerr=sub['ci95'] * 100,
                marker='o', ms=4, lw=1.4, color=PALETTE[i], capsize=2.5, label=f'k={k_shot}')
ax.set_xlabel('N-way'); ax.set_ylabel('Accuracy (%)')
ax.set_title('Dataset A (Kvasir-Capsule): Accuracy vs N-way (ssl_full)', fontsize=9.5)
ax.legend(); fig.tight_layout()
fig.savefig(f'{EVAL_DIR}/fig_A_accuracy_vs_nway.png', bbox_inches='tight'); plt.show()

fig, ax = plt.subplots(figsize=FIG_SINGLE)
sub = sweep_summary_df[sweep_summary_df['n_way'] == N_WAY_MAX_A].sort_values('k_shot')
ax.errorbar(sub['k_shot'], sub['mean_accuracy'] * 100, yerr=sub['ci95'] * 100,
            marker='o', ms=5, lw=1.6, color=PALETTE[0], capsize=3, capthick=1.2, elinewidth=1.2)
ax.set_xticks(sub['k_shot']); ax.set_xlabel('K-shot'); ax.set_ylabel('Accuracy (%)')
ax.set_title(f'Dataset A: Accuracy vs K-shot (N={N_WAY_MAX_A}, ssl_full)', fontsize=9.5)
fig.tight_layout()
fig.savefig(f'{EVAL_DIR}/fig_A_accuracy_vs_kshot.png', bbox_inches='tight'); plt.show()

fig, axes = plt.subplots(len(DATASETS), len(K_SHOTS), figsize=(FIG_WIDE[0], FIG_WIDE[1]*len(DATASETS)))
for r, (dataset_key, dcfg) in enumerate(DATASETS.items()):
    for c, k_shot in enumerate(K_SHOTS):
        ax = axes[r][c] if len(DATASETS) > 1 else axes[c]
        cm = confusion_matrices.get((dataset_key, k_shot))
        if cm is None:
            ax.axis('off'); continue
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(N_WAY_COMPARE)); ax.set_yticks(range(N_WAY_COMPARE))
        ax.set_xticklabels(dcfg['class_names'], rotation=45, ha='right', fontsize=6.5)
        ax.set_yticklabels(dcfg['class_names'], fontsize=6.5)
        ax.set_title(f"{dcfg['label']}, k={k_shot}", fontsize=8.5)
        for rr in range(N_WAY_COMPARE):
            for cc in range(N_WAY_COMPARE):
                color = 'white' if cm_norm[rr, cc] > 0.5 else '#222222'
                ax.text(cc, rr, f'{cm_norm[rr, cc]:.2f}', ha='center', va='center',
                        fontsize=6, color=color)
fig.suptitle(f'Dataset A Held-Out Class Confusion Matrices (ssl_full, N={N_WAY_COMPARE})', fontsize=9.5, y=1.02)
fig.tight_layout()
fig.savefig(f'{EVAL_DIR}/fig_confusion_matrices_both_datasets.png', bbox_inches='tight'); plt.show()

fig, ax = plt.subplots(figsize=FIG_WIDE)
metrics_to_plot = ['precision', 'recall_sensitivity', 'specificity', 'f1']
k_shot_plot = K_SHOTS[-1]
sub = per_class_df[(per_class_df['dataset'] == DATASET_A_KEY) &
                    (per_class_df['variant'] == 'ssl_full') & (per_class_df['k_shot'] == k_shot_plot)]
xpos = np.arange(len(DATASETS[DATASET_A_KEY]['class_names'])); width = 0.19
for i, m in enumerate(metrics_to_plot):
    ax.bar(xpos + (i - 1.5) * width, sub[m].values, width, label=m, color=PALETTE[i])
ax.set_xticks(xpos); ax.set_xticklabels(DATASETS[DATASET_A_KEY]['class_names'], rotation=30, ha='right')
ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
ax.set_title(f'Dataset A Per-Class Metrics, k={k_shot_plot}, N={N_WAY_COMPARE} (ssl_full)', fontsize=9.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=7.5)
fig.tight_layout()
fig.savefig(f'{EVAL_DIR}/fig_A_per_class_metrics.png', bbox_inches='tight'); plt.show()

if ABLATION_VARIANTS:
    LABELS = {'random': 'Random init', 'imagenet': 'ImageNet init',
              'ssl_lp': 'SSL + linear probe', 'ssl_full': 'SSL + ProtoNet (ours)'}
    variants_present = ['ssl_full'] + ABLATION_VARIANTS
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(FIG_WIDE[0], FIG_WIDE[1]), sharey=True)
    for i, (dataset_key, dcfg) in enumerate(DATASETS.items()):
        ax = axes[i] if len(DATASETS) > 1 else axes
        xg = np.arange(len(K_SHOTS)); w = 0.19
        offsets = np.linspace(-1.5, 1.5, len(variants_present)) if len(variants_present) > 1 else [0]
        for j, variant in enumerate(variants_present):
            sub = h2h_df[(h2h_df['dataset'] == dataset_key) & (h2h_df['variant'] == variant)].sort_values('k_shot')
            ax.bar(xg + offsets[j]*w, sub['mean_accuracy']*100, w, yerr=sub['ci95']*100,
                   label=LABELS.get(variant, variant), color=PALETTE[j % len(PALETTE)], capsize=2.5)
        ax.set_xticks(xg); ax.set_xticklabels([f'k={k}' for k in K_SHOTS])
        ax.set_title(dcfg['label'], fontsize=9)
        if i == 0:
            ax.set_ylabel('Accuracy (%)')
    title_suffix = '' if ABLATION_READY else ' (partial -- not all 4 arms present)'
    fig.suptitle(f'Dataset A Ablation Ladder{title_suffix}', fontsize=9.5, y=1.05)
    handles, labels_ = (axes[0] if len(DATASETS) > 1 else axes).get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', bbox_to_anchor=(0.5, -0.15),
               ncol=len(variants_present), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f'{EVAL_DIR}/fig_ablation_by_dataset.png', bbox_inches='tight'); plt.show()
    print('\nFigures saved to', EVAL_DIR)
else:
    print('\n[FIGURE PENDING -- ABLATION CHECKPOINTS NOT FOUND] Fig 6 skipped.')
    print('All other figures saved to', EVAL_DIR)
