"""
densidade_espacial.py
---------------------
Modelo de densidade espacial avançado para estrelas de metalicidade solar
na Via Láctea, baseado em dados reais de Gaia DR3 + APOGEE DR17.

Incorpora:
- Disco fino com perfil radial quebrado (broken exponential) – realismo de Bovy et al. / Lian et al.
- Disco espesso residual
- Gradiente radial de metalicidade
- Forma sech² vertical (equilíbrio isotérmico – Cap. 2 do livro)

Autor: Luiz Tiago Wilcke
"""

import numpy as np

# ============================================================
# PARÂMETROS CALIBRADOS EM DADOS REAIS (Gaia + APOGEE)
# ============================================================
R_SOL = 8.2          # kpc  (Gravity Collaboration 2021)
Z_SOL = 0.0208       # kpc

# Disco fino – população low-α de metalicidade solar
# Valores típicos de Lian et al. 2025 / Bovy et al. para [Fe/H]≈0
RD_INTERNO = 2.2     # kpc  (escala interna)
RD_EXTERNO = 3.5     # kpc  (escala externa – flaring/break)
R_BREAK    = 8.5     # kpc  (raio de quebra para metalicidade solar)
ZD_FINO    = 0.28    # kpc  (escala vertical)
RHO0_FINO  = 1.0     # normalização

# Disco espesso residual
RD_ESPECO  = 2.8
ZD_ESPECO  = 0.85
FRAC_ESPECO = 0.07

# Gradiente radial de metalicidade (APOGEE: ~ -0.06 dex/kpc no disco fino)
GRADIENTE_FEH = -0.055   # dex / kpc
FEH_SOLAR_LOCAL = 0.00


def densidade_fino_quebrado(R, z):
    """
    Densidade do disco fino com perfil radial quebrado (broken exponential).
    
    ρ_fino(R,z) = ρ0 · Σ(R) · sech²(z / (2 zd))
    
    onde Σ(R) = exp(-R/Rd_in)                    se R ≤ R_break
               exp(-R_break/Rd_in) · exp(-(R-R_break)/Rd_out)  se R > R_break
    """
    R = np.asarray(R)
    z = np.asarray(z)
    
    # Perfil radial quebrado
    mask_in = R <= R_BREAK
    Sigma = np.empty_like(R, dtype=float)
    Sigma[mask_in]  = np.exp(-R[mask_in] / RD_INTERNO)
    Sigma[~mask_in] = (np.exp(-R_BREAK / RD_INTERNO) *
                       np.exp(-(R[~mask_in] - R_BREAK) / RD_EXTERNO))
    
    # Perfil vertical sech²
    sech2 = 1.0 / np.cosh(z / (2.0 * ZD_FINO))**2
    
    return RHO0_FINO * Sigma * sech2


def densidade_especo(R, z):
    """Densidade residual do disco espesso."""
    termo_r = np.exp(-R / RD_ESPECO)
    sech2 = 1.0 / np.cosh(z / (2.0 * ZD_ESPECO))**2
    return FRAC_ESPECO * RHO0_FINO * termo_r * sech2


def densidade_total(R, z):
    """Densidade total de estrelas com [Fe/H] próximo de solar."""
    return densidade_fino_quebrado(R, z) + densidade_especo(R, z)


def metalicidade_esperada(R):
    """
    Valor esperado de [Fe/H] em função do raio galactocêntrico
    (gradiente linear calibrado em APOGEE).
    """
    return FEH_SOLAR_LOCAL + GRADIENTE_FEH * (R - R_SOL)


def densidade_maxima(raio_max=20.0, altura_max=3.0, n_grid=150):
    """Estima ρ_max no volume de interesse (para rejeição de Monte Carlo)."""
    r = np.linspace(0.05, raio_max, n_grid)
    z = np.linspace(-altura_max, altura_max, n_grid)
    R, Z = np.meshgrid(r, z)
    return float(np.max(densidade_total(R, Z)) * 1.08)
