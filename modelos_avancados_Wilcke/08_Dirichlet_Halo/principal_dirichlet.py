#!/usr/bin/env python3
"""Subpopulações do Halo via Processo de Dirichlet – Luiz Tiago Wilcke 2026"""
import sys, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from modulos.dpmm import gerar_subpopulacoes_halo, cluster_simples, stick_breaking

os.makedirs("resultados", exist_ok=True); os.makedirs("figuras", exist_ok=True)
print("="*60 + "\n  PROCESSO DE DIRICHLET – Subpopulações do Halo\n  Autor: Luiz Tiago Wilcke – 2026\n" + "="*60)

data, labels_true, w = gerar_subpopulacoes_halo(n=2500, K=4)
lab, cents = cluster_simples(data, K=4)
print(f"Pesos stick-breaking: {np.round(w,3)}")
print(f"Centróides recuperados:\n{np.round(cents,2)}")

pd.DataFrame({"FeH": data[:,0], "v_phi": data[:,1], "label": lab}).to_csv(
    "resultados/halo_subpops.csv", index=False)

fig, ax = plt.subplots(figsize=(8,6))
for k in range(4):
    m = lab == k
    ax.scatter(data[m,0], data[m,1], s=8, alpha=0.5, label=f"comp. {k+1}")
ax.scatter(cents[:,0], cents[:,1], c="k", s=120, marker="x", lw=2, label="centróides")
ax.set_xlabel("[Fe/H]"); ax.set_ylabel(r"$v_\phi$ [km/s]")
ax.set_title("Subpopulações do halo – mistura via Processo de Dirichlet")
ax.legend(); ax.grid(True, alpha=0.3)
fig.savefig("figuras/01_halo_dpmm.png", dpi=150, bbox_inches="tight"); plt.close()
print("Modelo Dirichlet / halo concluído.")
