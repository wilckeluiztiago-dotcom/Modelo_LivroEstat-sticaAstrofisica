#!/usr/bin/env python3
"""Localização Multi-Mensageira GW+EM – Luiz Tiago Wilcke 2026"""
import sys, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from modulos.coincidencia import (probabilidade_coincidencia_temporal,
                                   mapa_probabilidade_skymap, fusao_likelihoods)

os.makedirs("resultados", exist_ok=True); os.makedirs("figuras", exist_ok=True)
print("="*60 + "\n  MULTI-MENSAGEIRO – GW + Contraparte Eletromagnética\n  Autor: Luiz Tiago Wilcke – 2026\n" + "="*60)

# Dados inspirados em GW170817 / GRB 170817A
dt = 1.74  # segundos (GRB após GW)
sigma_t = 0.5
p_temp = probabilidade_coincidencia_temporal(dt, sigma_t)
print(f"Probabilidade de coincidência temporal: {p_temp:.4f}")

# Skymap simplificado
ra0, dec0, sig = 197.45, -23.38, 5.0  # graus (aprox. NGC 4993)
ra = np.linspace(ra0-20, ra0+20, 80)
dec = np.linspace(dec0-15, dec0+15, 60)
RA, DEC = np.meshgrid(ra, dec)
prob = mapa_probabilidade_skymap(RA, DEC, ra0, dec0, sig)
prob /= prob.sum()

L_gw, L_em = 0.85, 0.92
post = fusao_likelihoods(L_gw, L_em)
print(f"Posterior de associação (fusão de likelihoods): {post:.4f}")

pd.DataFrame({"ra": RA.ravel(), "dec": DEC.ravel(), "prob": prob.ravel()}).to_csv(
    "resultados/skymap_prob.csv", index=False)

fig, ax = plt.subplots(figsize=(8,6))
im = ax.pcolormesh(RA, DEC, prob, shading="auto", cmap="hot")
ax.plot(ra0, dec0, "c*", ms=15, label="NGC 4993 / GW170817")
ax.set_xlabel("RA (deg)"); ax.set_ylabel("Dec (deg)")
ax.set_title("Mapa de probabilidade multi-mensageiro (simplificado)")
ax.legend(); fig.colorbar(im, ax=ax, label="probabilidade")
fig.savefig("figuras/01_skymap.png", dpi=150, bbox_inches="tight"); plt.close()
print("Skymap e resultados salvos. Modelo multi-mensageiro concluído.")
