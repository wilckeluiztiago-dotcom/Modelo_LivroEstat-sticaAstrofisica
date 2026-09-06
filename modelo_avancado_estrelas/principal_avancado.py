#!/usr/bin/env python3
"""
principal_avancado.py
---------------------
Modelo estatístico avançado de estrelas com metalicidade solar na Via Láctea.

Incorpora teoria dos Capítulos 1-4, 7, 10 e 11 do livro
"Estatística Aplicada à Astrofísica" – Luiz Tiago Wilcke (2026).

- Densidade com broken exponential calibrada em Gaia/APOGEE
- Gradiente de metalicidade real
- MCMC Metropolis-Hastings para parâmetros estruturais
- Matriz de Informação de Fisher
- Função de Correlação de Dois Pontos (Landy-Szalay)
- Catálogo sintético + gráficos + resultados numéricos

Autor: Luiz Tiago Wilcke
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from modulos.amostragem_avancada import amostrar_catalogo
from modulos.inferencia_bayesiana import metropolis_hastings, fisher_aproximada
from modulos.estatistica_espacial import estimador_landy_szalay
from modulos.visualizacao_avancada import (
    plot_face_on, plot_edge_on, plot_feh_vs_R,
    plot_tpcf, plot_corner_mcmc, plot_perfil_vertical_radial
)

# ============================================================
print("="*70)
print("  MODELO AVANÇADO – ESTRELAS COM METALICIDADE SOLAR NA VIA LÁCTEA")
print("  Teoria: Kolmogorov, Bayes, MCMC, Fisher, TPCF (Landy-Szalay)")
print("  Autor: Luiz Tiago Wilcke – 2026")
print("="*70)

# ------------------------------------------------------------
# 1. Geração do catálogo
# ------------------------------------------------------------
print("\n[1/6] Amostragem de Monte Carlo (rejeição + gradiente de [Fe/H])...")
cat = amostrar_catalogo(n_estrelas=75000, semente=2026)
print(f"      Estrelas geradas        : {cat['n']}")
print(f"      Taxa de aceitação       : {cat['taxa_aceitacao']:.3%}")
print(f"      <R>                     : {cat['R'].mean():.2f} kpc")
print(f"      <|z|>                   : {np.abs(cat['z']).mean():.3f} kpc")
print(f"      <[Fe/H]>                : {cat['feh'].mean():.4f} dex")
print(f"      σ([Fe/H])               : {cat['feh'].std():.4f} dex")
print(f"      Fração disco fino       : {(cat['componente']==0).mean():.1%}")

# ------------------------------------------------------------
# 2. Salvar catálogo
# ------------------------------------------------------------
print("\n[2/6] Salvando catálogo CSV...")
df = pd.DataFrame({
    "x_gal_kpc": cat["x"], "y_gal_kpc": cat["y"], "z_gal_kpc": cat["z"],
    "R_kpc": cat["R"], "feh": cat["feh"], "componente": cat["componente"],
    "x_hel_kpc": cat["x_hel"], "y_hel_kpc": cat["y_hel"], "z_hel_kpc": cat["z_hel"],
    "d_hel_kpc": cat["d_hel"], "l_deg": cat["l"], "b_deg": cat["b"]
})
os.makedirs("resultados", exist_ok=True)
df.to_csv("resultados/catalogo_avancado_estrelas_solar.csv", index=False, float_format="%.5f")
print(f"      Arquivo: resultados/catalogo_avancado_estrelas_solar.csv ({os.path.getsize('resultados/catalogo_avancado_estrelas_solar.csv')/1e6:.2f} MB)")

# ------------------------------------------------------------
# 3. Inferência Bayesiana (MCMC)
# ------------------------------------------------------------
print("\n[3/6] MCMC Metropolis-Hastings dos parâmetros estruturais...")
# Subamostra para acelerar
idx = np.random.default_rng(1).choice(cat["n"], size=12000, replace=False)
amostra, taxa_aceite, _ = metropolis_hastings(cat["R"][idx], cat["z"][idx],
                                              n_passos=6000, n_burn=1500)
print(f"      Taxa de aceitação MCMC : {taxa_aceite:.1%}")
print(f"      Posterior média Rd_in   : {amostra[:,0].mean():.3f} ± {amostra[:,0].std():.3f} kpc")
print(f"      Posterior média Rd_out  : {amostra[:,1].mean():.3f} ± {amostra[:,1].std():.3f} kpc")
print(f"      Posterior média zd      : {amostra[:,2].mean():.3f} ± {amostra[:,2].std():.3f} kpc")

# ------------------------------------------------------------
# 4. Matriz de Fisher
# ------------------------------------------------------------
print("\n[4/6] Matriz de Informação de Fisher (ponto MAP aproximado)...")
theta_map = np.median(amostra, axis=0)
I = fisher_aproximada(theta_map, cat["R"][idx], cat["z"][idx])
try:
    cov = np.linalg.inv(I)
    sigma_fisher = np.sqrt(np.diag(cov))
    print(f"      σ_Fisher(Rd_in)  ≈ {sigma_fisher[0]:.4f} kpc")
    print(f"      σ_Fisher(Rd_out) ≈ {sigma_fisher[1]:.4f} kpc")
    print(f"      σ_Fisher(zd)     ≈ {sigma_fisher[2]:.4f} kpc")
except np.linalg.LinAlgError:
    print("      Matriz de Fisher singular (amostra limitada) – usando dispersão MCMC.")
    sigma_fisher = amostra.std(axis=0)

# ------------------------------------------------------------
# 5. Função de Correlação de Dois Pontos
# ------------------------------------------------------------
print("\n[5/6] Estimador de Landy-Szalay da TPCF...")
# Usar subamostra espacial
pos = np.column_stack([cat["x"][idx], cat["y"][idx], cat["z"][idx]])
r_cent, xi, _, _, _ = estimador_landy_szalay(pos, n_random=25000)
print(f"      ξ(r) calculado em {len(r_cent)} bins de separação")

# ------------------------------------------------------------
# 6. Gráficos
# ------------------------------------------------------------
print("\n[6/6] Gerando gráficos científicos...")
os.makedirs("figuras", exist_ok=True)
plot_face_on(cat, "figuras/01_face_on.png")
plot_edge_on(cat, "figuras/02_edge_on.png")
plot_feh_vs_R(cat, "figuras/03_gradiente_feh.png")
plot_perfil_vertical_radial(cat, "figuras/04_perfis.png")
plot_tpcf(r_cent, xi, "figuras/05_tpcf_landy_szalay.png")
plot_corner_mcmc(amostra, "figuras/06_corner_mcmc.png")

# ------------------------------------------------------------
# Resumo final
# ------------------------------------------------------------
print("\n" + "="*70)
print("RESULTADOS NUMÉRICOS PRINCIPAIS")
print("="*70)
print(f"Número de estrelas no catálogo      : {cat['n']}")
print(f"Parâmetros estruturais (posterior MCMC):")
print(f"  Rd_interno  = {amostra[:,0].mean():.3f} ± {amostra[:,0].std():.3f} kpc")
print(f"  Rd_externo  = {amostra[:,1].mean():.3f} ± {amostra[:,1].std():.3f} kpc")
print(f"  zd_fino     = {amostra[:,2].mean():.3f} ± {amostra[:,2].std():.3f} kpc")
print(f"Incertezas Fisher (aprox.)          : {sigma_fisher}")
print(f"Gradiente de [Fe/H] adotado         : -0.055 dex/kpc")
print(f"Dispersão intrínseca de [Fe/H]      : 0.075 dex")
print("="*70)
print("Arquivos gerados em resultados/ e figuras/")
print("Modelo avançado concluído com sucesso.")
print("="*70)
