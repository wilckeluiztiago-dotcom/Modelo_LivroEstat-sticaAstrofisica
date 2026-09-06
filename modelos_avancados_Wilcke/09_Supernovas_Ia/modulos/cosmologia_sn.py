"""
cosmologia_sn.py – Distância de luminosidade e likelihood de SN Ia
Cap. 13 (Hierarchical Bayes) + cosmologia.
Autor: Luiz Tiago Wilcke – 2026
"""
import numpy as np

def E_z(z, Om, w=-1.0):
    """H(z)/H0 para flat wCDM."""
    Ol = 1.0 - Om
    return np.sqrt(Om*(1+z)**3 + Ol*(1+z)**(3*(1+w)))

def distancia_comovel(z, Om, H0=70.0, w=-1.0, n=200):
    """Integração simples de chi(z) = c/H0 * int dz'/E(z')."""
    c = 299792.458  # km/s
    z = np.atleast_1d(z)
    chi = np.zeros_like(z, dtype=float)
    for i, zi in enumerate(z):
        zz = np.linspace(0, zi, n)
        chi[i] = np.trapezoid(1.0/E_z(zz, Om, w), zz)
    return (c/H0) * chi

def modulo_distancia(z, Om, H0=70.0, w=-1.0):
    """mu = 5 log10(d_L / 10 pc) com d_L = (1+z) chi."""
    d_L = (1+z) * distancia_comovel(z, Om, H0, w)  # Mpc
    return 5*np.log10(np.clip(d_L, 1e-5, None)*1e6 / 10.0)  # 10 pc

def log_likelihood_sn(theta, z, mu_obs, sigma_mu):
    """theta = (H0, Om, w)"""
    H0, Om, w = theta
    if not (50 < H0 < 90 and 0.05 < Om < 0.6 and -2 < w < -0.3):
        return -np.inf
    mu_mod = modulo_distancia(z, Om, H0, w)
    return -0.5 * np.sum(((mu_obs - mu_mod)/sigma_mu)**2)

def gerar_sn_sinteticas(n=80, H0=70.0, Om=0.3, w=-1.0, semente=42):
    rng = np.random.default_rng(semente)
    z = rng.uniform(0.01, 1.2, n)
    mu = modulo_distancia(z, Om, H0, w)
    sigma = 0.12 + 0.05*z
    mu_obs = mu + rng.normal(0, sigma)
    return z, mu_obs, sigma
