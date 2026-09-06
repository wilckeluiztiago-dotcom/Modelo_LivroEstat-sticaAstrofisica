"""
coincidencia.py – Coincidência temporal/espacial multi-mensageira
Cap. 12. Exemplo inspirado em GW170817 + GRB 170817A.
Autor: Luiz Tiago Wilcke – 2026
"""
import numpy as np

def probabilidade_coincidencia_temporal(dt, sigma_t, janela=10.0):
    """P(coincidência) sob hipótese de associação vs fundo uniforme."""
    # Likelihood de associação: Gaussiana centrada em dt=0
    L_assoc = np.exp(-0.5*(dt/sigma_t)**2) / (sigma_t*np.sqrt(2*np.pi))
    L_fundo = 1.0 / janela
    return L_assoc / (L_assoc + L_fundo)

def mapa_probabilidade_skymap(ra, dec, ra0, dec0, sigma_deg, nside_approx=50):
    """Mapa Gaussiano simplificado de localização na esfera celeste."""
    # Distância angular aproximada (pequenos ângulos)
    dra = (ra - ra0) * np.cos(np.radians(dec0))
    ddec = dec - dec0
    dist2 = dra**2 + ddec**2
    return np.exp(-0.5 * dist2 / sigma_deg**2)

def fusao_likelihoods(L_gw, L_em, prior_assoc=0.5):
    """Fusão Bayesiana de likelihoods multi-detectores."""
    post = (L_gw * L_em * prior_assoc) / (
        L_gw * L_em * prior_assoc + (1-prior_assoc)*0.01)
    return post
