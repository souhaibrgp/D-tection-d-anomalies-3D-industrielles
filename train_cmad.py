"""
CMAD-Net Training Script — Reverse Distillation.
Cross-Modal Attention Detection Network for 3D Anomaly Detection.

Entraînement en 2 phases :
  Phase 1 : Entraîner FPN + CMCA + Student Decoders avec loss de distillation
  Phase 2 : Calibrer la normalisation sur le validation set

Usage:
    python train_cmad.py
"""

import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve

from models.cmad_net import CMADNet
from dataset.dataset_cmad import MVTecADCMADDataset
from utils.losses import cmad_distillation_total_loss
from evaluation.metrics_utils import calculate_au_pro

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/working')
if IS_KAGGLE:
    import kagglehub
    DATA_ROOT = kagglehub.dataset_download("gengchensun/mvtec-3d-ad") + '/'
    LOG_DIR = '/kaggle/working/logs_cmad/'
    NUM_WORKERS = 2
else:
    DATA_ROOT = './data/'
    LOG_DIR = './logs/'
    NUM_WORKERS = 4

CLASSNAMES = [
    'bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
    'foam', 'peach', 'potato', 'rope', 'tire'
]

IMG_SIZE = 224  # ViT-B/16 natif
K_DICT = {'cookie': 3, 'dowel': 7, 'foam': 7, 'tire': 7}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


# ─────────────────────────────────────────────
# Phase 1 : Entraîner avec Reverse Distillation
# ─────────────────────────────────────────────
def train_phase1(model, train_dataloader, optimizer, scheduler, device, log, epochs):
    """
    Entraîne le modèle avec la loss de reverse distillation.

    Le student apprend à reconstruire les features du teacher (FPN+CMCA)
    sur les échantillons normaux.

    Modules entraînés : FPN_rgb, FPN_depth, CMCA, Student_rgb, Student_depth, Fusion
    Modules gelés : ViT Encoder (backbone)
    """
    print("=" * 60)
    print("PHASE 1 : Entraînement Reverse Distillation")
    print("=" * 60)
    print("PHASE 1 : Entraînement Reverse Distillation", file=log)

    model.train()

    for epoch in range(epochs):
        loss_list = []
        loss_details = {'rgb_cosine': [], 'rgb_l2': [], 'depth_cosine': [], 'depth_l2': []}

        for batch_idx, data in enumerate(train_dataloader):
            rgb_image, depth_image, _, _, _, _ = data
            rgb_image = rgb_image.to(device)
            depth_image = depth_image.to(device)

            # Forward pass (mode training → retourne teacher + student features)
            teacher_rgb, student_rgb, teacher_depth, student_depth = model(
                rgb_image, depth_image
            )

            # Loss de distillation inversée
            loss, loss_dict = cmad_distillation_total_loss(
                teacher_rgb, student_rgb,
                teacher_depth, student_depth,
            )

            loss_list.append(loss.item())
            for key in loss_details:
                if key in loss_dict:
                    loss_details[key].append(loss_dict[key])

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping pour la stabilité
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)

            optimizer.step()

        scheduler.step()
        mean_loss = np.mean(loss_list)
        mean_cos_rgb = np.mean(loss_details['rgb_cosine']) if loss_details['rgb_cosine'] else 0
        mean_cos_depth = np.mean(loss_details['depth_cosine']) if loss_details['depth_cosine'] else 0

        msg = (
            f'[Phase 1] Epoch {epoch+1}/{epochs}, '
            f'Loss: {mean_loss:.6f}, '
            f'Cos_RGB: {mean_cos_rgb:.4f}, '
            f'Cos_Depth: {mean_cos_depth:.4f}'
        )
        print(msg)
        print(msg, file=log)


# ─────────────────────────────────────────────
# Phase 2 : Calibration de la normalisation
# ─────────────────────────────────────────────
def calibrate_normalization(model, valid_dataloader, device, category_id):
    """
    Calcule les paramètres de normalisation (µ, σ) sur le validation set.

    Ces paramètres sont utilisés en test pour normaliser les cartes
    d'anomalies, ce qui améliore la comparabilité entre catégories.
    """
    model.eval()
    all_scores = []

    with torch.no_grad():
        for data in valid_dataloader:
            rgb_image, depth_image, _, _, _, _ = data
            rgb_image = rgb_image.to(device)
            depth_image = depth_image.to(device)

            anomaly_map, _, _ = model(rgb_image, depth_image, category_id)
            all_scores.append(anomaly_map.cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    return float(np.mean(all_scores)), float(np.std(all_scores))


# ─────────────────────────────────────────────
# Test / Évaluation
# ─────────────────────────────────────────────
def test(model, test_dataloader, device, log, classname, category_id,
         ckp_path=None, norm_params=None):
    """Évalue le modèle sur le set de test."""
    if ckp_path is not None:
        torch.save({
            'fpn_rgb': model.fpn_rgb.state_dict(),
            'fpn_depth': model.fpn_depth.state_dict(),
            'cmca': model.cmca.state_dict(),
            'student_rgb': model.student_rgb.state_dict(),
            'student_depth': model.student_depth.state_dict(),
            'fusion': model.fusion.state_dict(),
        }, ckp_path)

    model.eval()
    gt_list_px, pr_list_px = [], []
    gt_list_sp, pr_list_sp = [], []
    gts, predictions = [], []

    with torch.no_grad():
        for data in test_dataloader:
            rgb_image, depth_image, gt, ad_label, ad_type, _ = data
            rgb_image = rgb_image.to(device)
            depth_image = depth_image.to(device)

            anomaly_map, map_rgb, map_depth = model(
                rgb_image, depth_image, category_id
            )

            anomaly_map_np = anomaly_map[0, 0].cpu().numpy()

            # Normalisation avec les paramètres de calibration
            if norm_params is not None:
                norm_mean, norm_std = norm_params
                anomaly_map_np = (anomaly_map_np - norm_mean) / (norm_std + 1e-6)

            # Lissage gaussien
            anomaly_map_np = gaussian_filter(anomaly_map_np, sigma=4)

            gt_np = gt.cpu().numpy()
            gt_np[gt_np > 0.5] = 1
            gt_np[gt_np <= 0.5] = 0

            gt_list_px.extend(gt_np.astype(int).ravel())
            pr_list_px.extend(anomaly_map_np.ravel())
            gt_list_sp.append(np.max(gt_np.astype(int)))
            pr_list_sp.append(np.max(anomaly_map_np))
            gts.append(gt_np.squeeze())
            predictions.append(anomaly_map_np)

    # Métriques
    print(f'\n--- Results for {classname} ---')
    print(f'--- Results for {classname} ---', file=log)
    test_metric(gt_list_px, pr_list_px, gt_list_sp, pr_list_sp,
                predictions, gts, log)


def test_metric(gt_list_px, pr_list_px, gt_list_sp, pr_list_sp,
                predictions, gts, log=None):
    """Calcule et affiche toutes les métriques."""
    auroc_px = roc_auc_score(gt_list_px, pr_list_px)
    auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
    ap_px = average_precision_score(gt_list_px, pr_list_px)
    ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
    f1_px = f1_score_max(gt_list_px, pr_list_px)
    f1_sp = f1_score_max(gt_list_sp, pr_list_sp)

    au_pros, _ = calculate_au_pro(gts, predictions)
    pro, pro_10, pro_5, pro_1 = au_pros[0], au_pros[1], au_pros[2], au_pros[3]

    header = " I-AUROC | P-AUROC | AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% |   I-AP   |   P-AP   |   I-F1   |   P-F1"
    values = (f'  {auroc_sp:.3f}  |  {auroc_px:.3f}  |   {pro:.3f}   |   '
              f'{pro_10:.3f}   |   {pro_5:.3f}  |   {pro_1:.3f}  |   '
              f'{ap_sp:.3f}  |   {ap_px:.3f}  |   {f1_sp:.3f}  |   {f1_px:.3f}')

    print(header)
    print(values)
    if log:
        print(header, file=log)
        print(values, file=log)


# ─────────────────────────────────────────────
# Pipeline complet par catégorie
# ─────────────────────────────────────────────
def train_category(device, classname, data_root, log, epochs,
                   learning_rate, batch_size, ckp_dir=None):
    """Pipeline d'entraînement complet pour une catégorie."""
    category_id = CLASSNAMES.index(classname)
    k = K_DICT.get(classname, 5)
    print(f'\nSobel kernel k={k}')
    print(f'Sobel kernel k={k}', file=log)

    # Transforms
    train_mean = [0.485, 0.456, 0.406]
    train_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])
    depth_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.Normalize(train_mean, train_std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])
    gt_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Datasets
    train_dir = data_root + classname + '/train'
    valid_dir = data_root + classname + '/validation'
    test_dir = data_root + classname + '/test'

    train_data = MVTecADCMADDataset(
        data_dir=train_dir, classname=classname,
        transform=transform, depth_transform=depth_transform, k=k
    )
    valid_data = MVTecADCMADDataset(
        data_dir=valid_dir, classname=classname,
        transform=transform, depth_transform=depth_transform, k=k
    )
    test_data = MVTecADCMADDataset(
        data_dir=test_dir, classname=classname,
        transform=test_transform, depth_transform=depth_transform,
        test=True, gt_transform=gt_transform, k=k
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Modèle
    model = CMADNet(n_categories=len(CLASSNAMES), img_size=IMG_SIZE)
    model.to(device)

    # Optimiseur (uniquement les paramètres entraînables)
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Phase 1 : Entraîner avec Reverse Distillation
    train_phase1(model, train_loader, optimizer, scheduler, device, log, epochs)

    # Phase 2 : Calibration de la normalisation
    print("=" * 60)
    print("PHASE 2 : Calibration de la normalisation")
    print("=" * 60)
    print("PHASE 2 : Calibration de la normalisation", file=log)
    norm_params = calibrate_normalization(model, valid_loader, device, category_id)
    print(f'Normalisation params: mean={norm_params[0]:.4f}, std={norm_params[1]:.4f}')
    print(f'Normalisation params: mean={norm_params[0]:.4f}, std={norm_params[1]:.4f}', file=log)

    # Test
    ckp_path = None
    if ckp_dir:
        os.makedirs(ckp_dir, exist_ok=True)
        ckp_path = os.path.join(ckp_dir, f'{classname}.pth')

    test(model, test_loader, device, log, classname, category_id, ckp_path, norm_params)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    learning_rate = 1e-4
    batch_size = 16
    epochs = 50 if IS_KAGGLE else 100
    data_root = DATA_ROOT
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(LOG_DIR, exist_ok=True)
    log = open(os.path.join(LOG_DIR, "cmad_net_mvtec3d_seed42.txt"), 'a')
    ckp_dir = os.path.join(LOG_DIR, 'checkpoints')

    print(f"Device: {device}")
    print(f"Data root: {data_root}")
    print(f"Device: {device}", file=log)

    for i, classname in enumerate(CLASSNAMES):
        setup_seed(42)
        print('\n' + '=' * 70)
        print(f'  CMAD-Net — Training [{i+1}/{len(CLASSNAMES)}] : {classname}')
        print('=' * 70)
        print(f'Training [{i+1}/{len(CLASSNAMES)}] : {classname}', file=log)

        train_category(
            device=device,
            classname=classname,
            data_root=data_root,
            log=log,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            ckp_dir=ckp_dir,
        )

    log.close()
    print("\n✅ Entraînement CMAD-Net terminé pour toutes les catégories.")
