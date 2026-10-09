"""
Loss functions pour CMAD-Net — Paradigme Reverse Distillation.

Remplace les anciennes losses (consistency, diversity) qui étaient
fondamentalement incorrectes par des losses de distillation inversée :
le student doit reconstruire les features du teacher sur les samples normaux.

Losses disponibles :
- reverse_distillation_loss: Cosine + L2 entre teacher et student features
- cmad_distillation_total_loss: Loss combinée pour l'entraînement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def reverse_distillation_loss(
    teacher_feats: list,
    student_feats: list,
    lambda_l2: float = 0.1,
) -> tuple:
    """
    Loss de distillation inversée multi-échelles.

    Pour chaque niveau FPN, calcule la discordance entre les features du
    teacher (cible, gelé) et les reconstructions du student (entraîné).

    Combine deux composantes complémentaires :
    - Cosine distance : capture les différences de direction dans l'espace
      des features (insensible à la norme)
    - MSE (L2) : capture les différences d'amplitude

    Sur les samples normaux (entraînement), cette loss tend vers 0.
    Sur les samples anomaux (inférence), la discordance est élevée.

    Args:
        teacher_feats: Features teacher [(B, C, H, W)] × L niveaux.
                       Détachées du graphe (pas de gradient vers le teacher).
        student_feats: Features student [(B, C, H, W)] × L niveaux.
                       Connectées au graphe (gradients vers le student).
        lambda_l2: Poids de la composante L2 relative au cosine (défaut: 0.1).

    Returns:
        loss: Scalar tensor.
        loss_dict: Dict avec les losses individuelles par composante.
    """
    assert len(teacher_feats) == len(student_feats), \
        f"Nombre de niveaux teacher ({len(teacher_feats)}) != student ({len(student_feats)})"

    cos_losses = []
    l2_losses = []

    for i, (t, s) in enumerate(zip(teacher_feats, student_feats)):
        # Détacher le teacher pour ne pas calculer de gradient
        t = t.detach()

        # Cosine distance : 1 - cos_sim, moyennée sur les pixels et le batch
        # cos_sim(t, s) ∈ [-1, 1], on veut minimiser 1 - cos_sim → pousser vers 1
        cos_dist = torch.mean(1.0 - F.cosine_similarity(t, s, dim=1))
        cos_losses.append(cos_dist)

        # MSE (L2) normalisée par le nombre d'éléments
        mse = F.mse_loss(s, t)
        l2_losses.append(mse)

    # Moyenne sur les niveaux
    total_cos = sum(cos_losses) / len(cos_losses)
    total_l2 = sum(l2_losses) / len(l2_losses)
    total = total_cos + lambda_l2 * total_l2

    loss_dict = {
        'cosine': total_cos.item(),
        'l2': total_l2.item(),
        'total': total.item(),
    }

    return total, loss_dict


def cmad_distillation_total_loss(
    teacher_rgb_feats: list,
    student_rgb_feats: list,
    teacher_depth_feats: list,
    student_depth_feats: list,
    lambda_l2: float = 0.1,
    w_rgb: float = 1.0,
    w_depth: float = 1.0,
) -> tuple:
    """
    Loss totale de distillation pour CMAD-Net.

    Combine les losses de distillation pour les deux modalités (RGB et Depth).
    Les poids w_rgb et w_depth permettent d'équilibrer les deux branches.

    Args:
        teacher_rgb_feats: Features teacher RGB [(B, C, H, W)] × L.
        student_rgb_feats: Features student RGB [(B, C, H, W)] × L.
        teacher_depth_feats: Features teacher Depth [(B, C, H, W)] × L.
        student_depth_feats: Features student Depth [(B, C, H, W)] × L.
        lambda_l2: Poids de la composante L2 (défaut: 0.1).
        w_rgb: Poids de la branche RGB (défaut: 1.0).
        w_depth: Poids de la branche Depth (défaut: 1.0).

    Returns:
        total_loss: Scalar tensor.
        loss_dict: Dict avec les losses détaillées.
    """
    loss_rgb, dict_rgb = reverse_distillation_loss(
        teacher_rgb_feats, student_rgb_feats, lambda_l2
    )
    loss_depth, dict_depth = reverse_distillation_loss(
        teacher_depth_feats, student_depth_feats, lambda_l2
    )

    total = w_rgb * loss_rgb + w_depth * loss_depth

    loss_dict = {
        'rgb_cosine': dict_rgb['cosine'],
        'rgb_l2': dict_rgb['l2'],
        'rgb_total': dict_rgb['total'],
        'depth_cosine': dict_depth['cosine'],
        'depth_l2': dict_depth['l2'],
        'depth_total': dict_depth['total'],
        'total': total.item(),
    }

    return total, loss_dict
