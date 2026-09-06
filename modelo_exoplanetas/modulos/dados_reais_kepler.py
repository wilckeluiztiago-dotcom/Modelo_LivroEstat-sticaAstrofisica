"""
dados_reais_kepler.py
---------------------
Geração de curvas de luz sintéticas calibradas em estatísticas reais
do Kepler (KOI – Kepler Objects of Interest) e TESS.

Parâmetros médios baseados em catálogos públicos (NASA Exoplanet Archive).
Autor: Luiz Tiago Wilcke – 2026
"""

import numpy as np
from modulos.modelo_transito import gerar_curva_luz_realista, modelo_transito_suave


# Estatísticas reais aproximadas de planetas Kepler confirmados / candidatos
# (período médio ~ 10-20 d, Rp/R* típico ~ 0.02-0.08 para super-Terras/Netunos)
PARAMETROS_TIPICOS_KEPLER = [
    {"nome": "Kepler-like Super-Earth", "periodo": 8.46, "rp_rs": 0.045, "a_rs": 18.5, "b": 0.35, "t0": 5.2},
    {"nome": "Kepler-like Hot Neptune", "periodo": 4.23, "rp_rs": 0.070, "a_rs": 11.2, "b": 0.22, "t0": 2.1},
    {"nome": "Kepler-like Mini-Neptune", "periodo": 15.7, "rp_rs": 0.055, "a_rs": 25.0, "b": 0.48, "t0": 7.8},
    {"nome": "Kepler-like Gas Giant",   "periodo": 3.52, "rp_rs": 0.110, "a_rs": 8.8,  "b": 0.15, "t0": 1.4},
    {"nome": "TESS-like Ultra-short",   "periodo": 1.21, "rp_rs": 0.080, "a_rs": 5.1,  "b": 0.40, "t0": 0.6},
]


def gerar_amostra_temporal(duracao_dias=90.0, cadencia_min=30.0, semente=None):
    """
    Gera tempos de observação no estilo Kepler long-cadence (30 min)
    ou TESS (2 min / 30 min).
    """
    rng = np.random.default_rng(semente)
    dt = cadencia_min / (24 * 60)  # dias
    t = np.arange(0, duracao_dias, dt)
    # Pequenas irregularidades (gaps)
    mask = rng.random(len(t)) > 0.03  # ~3% de pontos faltantes
    return t[mask]


def simular_sistema(indice=0, duracao=80.0, cadencia=30.0, ruido=0.00025, semente=42):
    """
    Simula uma curva de luz completa de um sistema no estilo Kepler/TESS.
    """
    params = PARAMETROS_TIPICOS_KEPLER[indice % len(PARAMETROS_TIPICOS_KEPLER)].copy()
    t = gerar_amostra_temporal(duracao, cadencia, semente)
    fluxo = gerar_curva_luz_realista(t, params, ruido_sigma=ruido, semente=semente)
    return t, fluxo, params


def gerar_catalogo_sistemas(n_sistemas=12, semente=2026):
    """
    Gera um pequeno catálogo de sistemas sintéticos calibrados em dados reais.
    """
    rng = np.random.default_rng(semente)
    catalogo = []
    for i in range(n_sistemas):
        base = PARAMETROS_TIPICOS_KEPLER[i % len(PARAMETROS_TIPICOS_KEPLER)].copy()
        # Perturbações realistas
        base["periodo"] *= rng.uniform(0.85, 1.20)
        base["rp_rs"]   *= rng.uniform(0.80, 1.25)
        base["a_rs"]    *= rng.uniform(0.90, 1.15)
        base["b"]        = np.clip(base["b"] + rng.normal(0, 0.1), 0, 1.1)
        base["t0"]       = rng.uniform(0, base["periodo"])
        base["id"]       = f"SYN-KOI-{i+1:03d}"
        catalogo.append(base)
    return catalogo
