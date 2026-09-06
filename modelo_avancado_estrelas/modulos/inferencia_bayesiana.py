"""
inferencia_bayesiana.py
-----------------------
Inferência Bayesiana dos parâmetros estruturais do disco
usando MCMC Metropolis-Hastings (Cap. 4 do livro).

Modelo hierárquico simplificado:
- Likelihood: produto de densidades avaliadas nas posições amostradas
- Priors informativos baseados em resultados de Gaia/APOGEE
- Posterior via MCMC

Também calcula a matriz de Informação de Fisher (Cap. 3 e 10)
para incertezas aproximadas.

Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.special import logsumexp


def log_prior(theta):
    """
    Prior informativo (quase Jeffreys + bounds físicos).
    theta = [Rd_interno, Rd_externo, zd_fino]
    """
    Rd_in, Rd_out, zd = theta
    if not (1.0 < Rd_in < 4.5 and 2.0 < Rd_out < 6.0 and 0.15 < zd < 0.60):
        return -np.inf
    # Prior log-uniforme suave
    return -0.5 * ((Rd_in-2.2)/0.4)**2 - 0.5*((Rd_out-3.5)/0.6)**2 - 0.5*((zd-0.28)/0.05)**2


def log_likelihood(theta, R, z, R_break=8.5):
    """
    Log-likelihood sob o modelo de densidade quebrada.
    Usa apenas a forma funcional (normalização cancelada na razão).
    """
    Rd_in, Rd_out, zd = theta
    R = np.asarray(R)
    z = np.asarray(z)

    # Perfil radial
    mask = R <= R_break
    Sigma = np.empty_like(R)
    Sigma[mask]  = np.exp(-R[mask] / Rd_in)
    Sigma[~mask] = np.exp(-R_break / Rd_in) * np.exp(-(R[~mask]-R_break)/Rd_out)

    sech2 = 1.0 / np.cosh(z / (2.0 * zd))**2
    dens = Sigma * sech2
    dens = np.clip(dens, 1e-30, None)
    return np.sum(np.log(dens))


def log_posterior(theta, R, z):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, R, z)


def metropolis_hastings(R, z, n_passos=8000, n_burn=2000, semente=42):
    """
    Cadeia de Markov Monte Carlo simples (Metropolis-Hastings).
    Implementação didática do Cap. 4.
    """
    rng = np.random.default_rng(semente)
    # Ponto inicial
    theta = np.array([2.2, 3.5, 0.28])
    cadeia = np.zeros((n_passos, 3))
    logpost = np.zeros(n_passos)

    atual_logp = log_posterior(theta, R, z)
    aceitos = 0

    # Proposta: Gaussian random walk
    sigma_prop = np.array([0.08, 0.12, 0.015])

    for i in range(n_passos):
        prop = theta + rng.normal(0, sigma_prop)
        prop_logp = log_posterior(prop, R, z)

        if np.log(rng.random()) < (prop_logp - atual_logp):
            theta = prop
            atual_logp = prop_logp
            aceitos += 1

        cadeia[i] = theta
        logpost[i] = atual_logp

    taxa = aceitos / n_passos
    amostra = cadeia[n_burn:]
    return amostra, taxa, logpost


def fisher_aproximada(theta, R, z, eps=1e-4):
    """
    Matriz de Informação de Fisher numérica (Cap. 3 / 10).
    I_ij = -E[∂² log L / ∂θi ∂θj]
    Aqui usamos a amostra observada como estimativa.
    """
    n_par = len(theta)
    I = np.zeros((n_par, n_par))
    base = log_likelihood(theta, R, z)

    for i in range(n_par):
        for j in range(i, n_par):
            # Diferenças finitas de segunda ordem
            th_pp = theta.copy(); th_pp[i] += eps; th_pp[j] += eps
            th_pm = theta.copy(); th_pm[i] += eps; th_pm[j] -= eps
            th_mp = theta.copy(); th_mp[i] -= eps; th_mp[j] += eps
            th_mm = theta.copy(); th_mm[i] -= eps; th_mm[j] -= eps

            d2 = (log_likelihood(th_pp, R, z) - log_likelihood(th_pm, R, z)
                  - log_likelihood(th_mp, R, z) + log_likelihood(th_mm, R, z)) / (4*eps*eps)
            I[i,j] = I[j,i] = -d2

    return I
