"""
inferencia_bayesiana_exo.py
---------------------------
Inferência Bayesiana dos parâmetros de trânsito via MCMC Metropolis-Hastings
(Cap. 4 do livro). Também inclui kernel Quasi-Periódico de Processo Gaussiano
para atividade estelar (Apêndice F do livro).

Autor: Luiz Tiago Wilcke – 2026
"""

import numpy as np


def log_prior_transito(theta):
    """
    Prior para theta = [periodo, t0, rp_rs, a_rs, b]
    Bounds físicos + Gaussianas suaves centradas em valores típicos Kepler.
    """
    P, t0, k, a_rs, b = theta
    if not (0.5 < P < 50 and 0.001 < k < 0.2 and 3 < a_rs < 40 and 0 <= b < 1.2):
        return -np.inf
    # Prior informativo
    lp = 0.0
    lp += -0.5 * ((P - 8.0)/6.0)**2
    lp += -0.5 * ((k - 0.05)/0.03)**2
    lp += -0.5 * ((a_rs - 12.0)/5.0)**2
    return lp


def log_likelihood_caixa(theta, t, y, sigma=0.0004):
    """
    Likelihood Gaussiana sob modelo de caixa de trânsito.
    """
    from modulos.modelo_transito import modelo_caixa_transito, duracao_transito
    P, t0, k, a_rs, b = theta
    profundidade = k**2
    dur = duracao_transito(P, a_rs, b, k)
    modelo = modelo_caixa_transito(t, t0, P, profundidade, dur)
    residuo = y - modelo
    return -0.5 * np.sum((residuo/sigma)**2) - len(y)*np.log(sigma*np.sqrt(2*np.pi))


def log_posterior(theta, t, y, sigma=0.0004):
    lp = log_prior_transito(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_caixa(theta, t, y, sigma)


def metropolis_hastings_exo(t, y, n_passos=5000, n_burn=1000, semente=42):
    """
    MCMC Metropolis-Hastings para parâmetros de trânsito.
    """
    rng = np.random.default_rng(semente)
    # Inicialização perto de valores típicos
    theta = np.array([10.0, t.mean(), 0.04, 15.0, 0.3])
    cadeia = np.zeros((n_passos, 5))
    logp = np.zeros(n_passos)

    atual = log_posterior(theta, t, y)
    aceitos = 0
    sigma_prop = np.array([0.15, 0.01, 0.003, 0.8, 0.05])

    for i in range(n_passos):
        prop = theta + rng.normal(0, sigma_prop)
        # Garante b >= 0
        prop[4] = np.abs(prop[4])
        prop_logp = log_posterior(prop, t, y)
        if np.log(rng.random()) < (prop_logp - atual):
            theta = prop
            atual = prop_logp
            aceitos += 1
        cadeia[i] = theta
        logp[i] = atual

    taxa = aceitos / n_passos
    return cadeia[n_burn:], taxa


# ------------------------------------------------------------
# Processo Gaussiano – Kernel Quasi-Periódico (Apêndice F)
# ------------------------------------------------------------
def kernel_quasi_periodico(t1, t2, A, Gamma, P_rot, sigma_white):
    """
    Kernel Quasi-Periódico usado para modelar atividade estelar:

    k(t,t') = A² exp( -Γ sin²(π|t-t'|/P_rot) - (|t-t'|/λ)² ) + σ² δ_{tt'}

    Aqui simplificamos λ → ∞ (só o termo periódico + ruído branco).
    """
    dt = np.abs(t1[:, None] - t2[None, :])
    k = A**2 * np.exp( -Gamma * np.sin(np.pi * dt / P_rot)**2 )
    if t1 is t2 or np.array_equal(t1, t2):
        k += np.eye(len(t1)) * sigma_white**2
    return k


def log_likelihood_gp_simples(t, y, A, Gamma, P_rot, sigma_w):
    """
    Log-likelihood marginal de um GP com kernel QP (versão simplificada).
    Usa Cholesky para estabilidade numérica.
    """
    K = kernel_quasi_periodico(t, t, A, Gamma, P_rot, sigma_w)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - y.mean()))
        logdet = 2 * np.sum(np.log(np.diag(L)))
        n = len(y)
        return -0.5 * (np.dot(y-y.mean(), alpha) + logdet + n*np.log(2*np.pi))
    except np.linalg.LinAlgError:
        return -np.inf
