"""
dpmm.py – Processo de Dirichlet / mistura infinita para subpopulações do halo
Cap. 15. Stick-breaking + clustering simplificado.
Autor: Luiz Tiago Wilcke – 2026
"""
import numpy as np
from scipy.stats import multivariate_normal

def stick_breaking(alpha, K_max=10, semente=42):
    """Construção stick-breaking do Processo de Dirichlet."""
    rng = np.random.default_rng(semente)
    betas = rng.beta(1, alpha, size=K_max)
    remaining = 1.0
    weights = np.zeros(K_max)
    for k in range(K_max):
        weights[k] = betas[k] * remaining
        remaining *= (1 - betas[k])
    return weights / weights.sum()

def gerar_subpopulacoes_halo(n=3000, K=4, semente=2026):
    """Gera estrelas do halo com K subpopulações químicas/cinemáticas."""
    rng = np.random.default_rng(semente)
    # Médias em ([Fe/H], v_phi) aproximadas
    means = np.array([[-1.5, 50], [-0.8, 120], [-2.0, 20], [-1.2, 180]])
    covs = [np.array([[0.15,0],[0,800]]), np.array([[0.1,0],[0,600]]),
            np.array([[0.2,0],[0,1000]]), np.array([[0.12,0],[0,700]])]
    weights = stick_breaking(1.5, K, semente)
    labels = rng.choice(K, size=n, p=weights[:K]/weights[:K].sum())
    data = np.zeros((n, 2))
    for k in range(K):
        m = labels == k
        data[m] = rng.multivariate_normal(means[k], covs[k], size=m.sum())
    return data, labels, weights[:K]

def cluster_simples(data, K=4, n_iter=30, semente=7):
    """K-means simplificado como aproximação de atribuição DPMM."""
    rng = np.random.default_rng(semente)
    cents = data[rng.choice(len(data), K, replace=False)]
    for _ in range(n_iter):
        dist = np.linalg.norm(data[:,None,:] - cents[None,:,:], axis=2)
        lab = dist.argmin(1)
        for k in range(K):
            if (lab==k).sum() > 0:
                cents[k] = data[lab==k].mean(0)
    return lab, cents
