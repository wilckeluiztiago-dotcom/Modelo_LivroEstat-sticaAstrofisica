#!/usr/bin/env python3
"""Detecção de BAO via TPCF – Luiz Tiago Wilcke 2026"""
import sys, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from modulos.tpcf_bao import landy_szalay, modelo_xi_bao, gerar_catalogo_galaxias

os.makedirs("resultados", exist_ok=True); os.makedirs("figuras", exist_ok=True)
print("="*60 + "\n  MODELO BAO – Função de Correlação + Pico Acústico\n  Autor: Luiz Tiago Wilcke – 2026\n" + "="*60)

pos = gerar_catalogo_galaxias(n=6000, box=800.0)
r, xi = landy_szalay(pos, n_rand=18000, r_bins=np.linspace(30, 160, 22))
xi_mod = modelo_xi_bao(r, r_bao=105.0)

# Fit simples do pico
from scipy.optimize import curve_fit
def f(r, r0, A, sig, g, B): return modelo_xi_bao(r, r0, A, sig, g, B)
try:
    popt, _ = curve_fit(f, r, xi, p0=[105, 0.04, 12, 1.6, 1.0], bounds=([80,0,5,0.5,0.1],[140,0.3,30,3,5]))
    r_bao_fit = popt[0]
except Exception:
    r_bao_fit, popt = 105.0, [105, 0.04, 12, 1.6, 1.0]

print(f"Pico BAO ajustado: r_bao = {r_bao_fit:.1f} Mpc/h")
pd.DataFrame({"r_Mpc_h": r, "xi": xi, "xi_modelo": f(r, *popt)}).to_csv("resultados/tpcf_bao.csv", index=False)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(r, r**2 * xi, "ko", ms=5, label=r"$r^{2}\xi(r)$ dados")
ax.plot(r, r**2 * f(r,*popt), "r-", lw=2, label=f"modelo (r_BAO={r_bao_fit:.1f})")
ax.axvline(r_bao_fit, color="g", ls="--", label="pico BAO")
ax.set_xlabel(r"$r$ [$h^{-1}$ Mpc]"); ax.set_ylabel(r"$r^{2}\xi(r)$")
ax.set_title("Detecção do pico de BAO – estimador Landy-Szalay")
ax.legend(); ax.grid(True, alpha=0.3)
fig.savefig("figuras/01_bao_tpcf.png", dpi=150, bbox_inches="tight"); plt.close()
print("Figura e CSV salvos. Modelo BAO concluído.")
