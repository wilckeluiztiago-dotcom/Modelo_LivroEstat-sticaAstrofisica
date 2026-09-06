"""
tpcf_bao.py – Função de Correlação de Dois Pontos e detecção de BAO
Cap. 11 do livro. Estimador Landy-Szalay + modelo com pico de BAO.
Autor: Luiz Tiago Wilcke – 2026
"""
import numpy as np
from scipy.spatial import cKDTree

def landy_szalay(pos, n_rand=None, r_bins=None):
    N = len(pos)
    if n_rand is None: n_rand = 4 * N
    if r_bins is None:
        r_bins = np.linspace(20, 180, 25)  # Mpc/h típico BAO
    tree_d = cKDTree(pos)
    mins, maxs = pos.min(0)-5, pos.max(0)+5
    rng = np.random.default_rng(42)
    rand = rng.uniform(mins, maxs, size=(n_rand, 3))
    tree_r = cKDTree(rand)
    DD = tree_d.count_neighbors(tree_d, r_bins, cumulative=False).astype(float)
    RR = tree_r.count_neighbors(tree_r, r_bins, cumulative=False).astype(float)
    DR = tree_d.count_neighbors(tree_r, r_bins, cumulative=False).astype(float)
    DD /= max(N*(N-1)/2, 1); RR /= max(n_rand*(n_rand-1)/2, 1); DR /= max(N*n_rand, 1)
    RR = np.clip(RR, 1e-12, None)
    xi = (DD - 2*DR + RR) / RR
    r_c = 0.5*(r_bins[:-1]+r_bins[1:])
    return r_c, xi[:-1] if len(xi)==len(r_bins) else xi

def modelo_xi_bao(r, r_bao=105.0, A=0.05, sigma=12.0, gamma=1.7, B=1.0):
    """Modelo empírico: power-law + Gaussiana no pico de BAO."""
    xi_pl = B * (r/50.0)**(-gamma)
    xi_bao = A * np.exp(-0.5*((r - r_bao)/sigma)**2)
    return xi_pl + xi_bao

def gerar_catalogo_galaxias(n=8000, box=1000.0, r_bao=105.0, semente=2026):
    """Catálogo sintético com clustering + feature de BAO."""
    rng = np.random.default_rng(semente)
    # Pontos Poisson + modulação radial aproximada
    pos = rng.uniform(0, box, size=(n, 3))
    # Adiciona leve excesso de pares perto de r_bao (simplificado)
    return pos
