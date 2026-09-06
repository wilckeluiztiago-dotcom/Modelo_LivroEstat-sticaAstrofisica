"""
estatistica_espacial.py
-----------------------
Função de Correlação de Dois Pontos (TPCF) e estatísticas espaciais
(Capítulo 11 do livro – Estatística Espacial e Análise de Grandes Estruturas).

Implementa o estimador de Landy-Szalay (padrão ouro em cosmologia/astrofísica).

Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.spatial import cKDTree


def estimador_landy_szalay(posicoes, n_random=None, r_bins=None, boxsize=None):
    """
    Estimador de Landy-Szalay da função de correlação de dois pontos ξ(r).
    
    ξ_LS(r) = (DD - 2 DR + RR) / RR
    
    onde DD, DR, RR são contagens de pares normalizadas.
    
    posicoes : array (N, 3) de coordenadas cartesianas
    """
    N = len(posicoes)
    if n_random is None:
        n_random = 3 * N

    if r_bins is None:
        r_bins = np.logspace(np.log10(0.15), np.log10(8.0), 18)

    # Árvore de dados
    tree_d = cKDTree(posicoes)

    # Catálogo aleatório (uniforme no mesmo volume aproximado)
    mins = posicoes.min(axis=0) - 0.5
    maxs = posicoes.max(axis=0) + 0.5
    rng = np.random.default_rng(123)
    random = rng.uniform(mins, maxs, size=(n_random, 3))
    tree_r = cKDTree(random)

    # Contagens de pares
    DD = tree_d.count_neighbors(tree_d, r_bins, cumulative=False)
    RR = tree_r.count_neighbors(tree_r, r_bins, cumulative=False)
    DR = tree_d.count_neighbors(tree_r, r_bins, cumulative=False)

    # Normalização
    nD = N
    nR = n_random
    DD = DD.astype(float) / (nD*(nD-1)/2)
    RR = RR.astype(float) / (nR*(nR-1)/2)
    DR = DR.astype(float) / (nD * nR)

    # Evitar divisão por zero
    RR = np.clip(RR, 1e-12, None)
    xi = (DD - 2*DR + RR) / RR

    # count_neighbors retorna um valor por limite em r_bins;
    # usamos os centros dos intervalos entre os limites
    r_centros = 0.5 * (r_bins[:-1] + r_bins[1:])
    xi = xi[:-1]   # alinhar com os centros
    return r_centros, xi, DD, DR, RR


def densidades_locais(posicoes, k=20):
    """
    Estimativa de densidade local via k-vizinhos mais próximos
    (útil para identificar superdensidades / voids).
    """
    tree = cKDTree(posicoes)
    dist, _ = tree.query(posicoes, k=k+1)
    # volume da esfera do k-ésimo vizinho
    vol = (4/3)*np.pi * dist[:, -1]**3
    dens = k / np.clip(vol, 1e-8, None)
    return dens
