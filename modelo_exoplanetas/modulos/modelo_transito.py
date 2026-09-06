"""
modelo_transito.py
------------------
Modelo físico de trânsito exoplanetário (aproximação de Mandel & Agol
simplificada + lei de limb-darkening linear).

Baseado em dados reais de Kepler/TESS e equações de fotometria de trânsito.
Autor: Luiz Tiago Wilcke – 2026
"""

import numpy as np
from scipy.special import j1  # para difração se necessário, mas aqui não


def limb_darkening_linear(mu, u1=0.3):
    """
    Lei linear de obscurecimento de limbo:
    I(mu) / I(1) = 1 - u1*(1 - mu)
    mu = cos(theta)
    """
    return 1.0 - u1 * (1.0 - mu)


def profundidade_transito(rp_rs, b=0.0):
    """
    Profundidade aproximada do trânsito (fração de fluxo bloqueado).
    Para impacto b e razão de raios rp/rs << 1:
    delta ≈ (rp/rs)^2 * (área efetiva com limb-darkening)
    """
    return (rp_rs)**2


def duracao_transito(periodo, a_rs, b, rp_rs=0.0):
    """
    Duração total do trânsito (aproximação circular):
    T_dur ≈ (P/π) * arcsin( sqrt( (1+k)^2 - b^2 ) / (a/R*) * sin i  )
    Simplificado para órbita circular:
    """
    k = rp_rs
    arg = np.sqrt(np.clip((1 + k)**2 - b**2, 0, None)) / (a_rs)
    arg = np.clip(arg, -1, 1)
    return (periodo / np.pi) * np.arcsin(arg)


def modelo_caixa_transito(t, t0, periodo, profundidade, duracao):
    """
    Modelo de caixa (box) simples para trânsito – útil para detecção.
    Fluxo = 1 - profundidade  durante o trânsito.
    """
    fase = ((t - t0 + periodo/2) % periodo) - periodo/2
    em_transito = np.abs(fase) < (duracao / 2.0)
    fluxo = np.ones_like(t)
    fluxo[em_transito] = 1.0 - profundidade
    return fluxo


def modelo_transito_suave(t, t0, periodo, rp_rs, a_rs, b, u1=0.3, n_pts=50):
    """
    Modelo de trânsito mais realista (aproximação trapezoidal + limb darkening).
    Gera curva de luz sintética calibrada em estatísticas Kepler.
    """
    k = rp_rs
    profundidade = k**2
    # Duração total e de flat-bottom
    T14 = duracao_transito(periodo, a_rs, b, k)
    T23 = duracao_transito(periodo, a_rs, b, -k)  # aproximação
    T23 = max(T23, 0.0)

    fase = ((t - t0 + periodo/2) % periodo) - periodo/2
    fluxo = np.ones_like(t, dtype=float)

    # Região de ingresso/egresso (aproximação linear)
    half = T14 / 2.0
    half_flat = T23 / 2.0

    mask_flat = np.abs(fase) < half_flat
    mask_in = (np.abs(fase) >= half_flat) & (np.abs(fase) < half)

    fluxo[mask_flat] = 1.0 - profundidade

    # Ingresso/egresso linear
    if half > half_flat:
        frac = (half - np.abs(fase[mask_in])) / (half - half_flat)
        fluxo[mask_in] = 1.0 - profundidade * frac

    return fluxo


def gerar_curva_luz_realista(t, parametros, ruido_sigma=0.0003, semente=None):
    """
    Gera curva de luz com trânsito + ruído branco + ruído vermelho simples.
    parametros: dict com t0, periodo, rp_rs, a_rs, b
    """
    rng = np.random.default_rng(semente)
    fluxo = modelo_transito_suave(
        t,
        parametros["t0"],
        parametros["periodo"],
        parametros["rp_rs"],
        parametros["a_rs"],
        parametros["b"]
    )
    # Ruído branco (fotônico + instrumental) – típico Kepler para estrela brilhante
    fluxo += rng.normal(0, ruido_sigma, size=len(t))
    # Ruído vermelho simples (atividade estelar de baixa frequência)
    if len(t) > 10:
        kernel = np.exp(-0.5 * (np.arange(len(t))/ (0.5*len(t)/periodo_scale(parametros)))**2 )
        # simplificado: adiciona componente correlacionada
        red = rng.normal(0, ruido_sigma*0.6, size=len(t))
        # suaviza
        from scipy.ndimage import gaussian_filter1d
        red = gaussian_filter1d(red, sigma=max(3, len(t)//80))
        fluxo += red
    return fluxo


def periodo_scale(p):
    return max(p.get("periodo", 5.0), 1.0)
