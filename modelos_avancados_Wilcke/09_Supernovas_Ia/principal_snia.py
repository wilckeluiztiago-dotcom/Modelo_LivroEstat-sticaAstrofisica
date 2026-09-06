#!/usr/bin/env python3
"""Cosmologia com Supernovas Ia – Luiz Tiago Wilcke 2026"""
import sys, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from modulos.cosmologia_sn import (gerar_sn_sinteticas, log_likelihood_sn,
                                    modulo_distancia)

os.makedirs("resultados", exist_ok=True); os.makedirs("figuras", exist_ok=True)
print("="*60 + "\n  COSMOLOGIA COM SUPERNOVAS Ia\n  Autor: Luiz Tiago Wilcke – 2026\n" + "="*60)

z, mu, sig = gerar_sn_sinteticas(n=100, H0=70, Om=0.3, w=-1.0)
print(f"Catálogo sintético: {len(z)} SN Ia")

# MCMC simples
rng = np.random.default_rng(7)
theta = np.array([70.0, 0.3, -1.0])
cadeia = []
atual = log_likelihood_sn(theta, z, mu, sig)
for i in range(4000):
    prop = theta + rng.normal(0, [1.5, 0.04, 0.08])
    lp = log_likelihood_sn(prop, z, mu, sig)
    if np.log(rng.random()) < lp - atual:
        theta, atual = prop, lp
    if i > 800:
        cadeia.append(theta.copy())
cadeia = np.array(cadeia)
print(f"Posterior H0  = {cadeia[:,0].mean():.1f} ± {cadeia[:,0].std():.1f}")
print(f"Posterior Om  = {cadeia[:,1].mean():.3f} ± {cadeia[:,1].std():.3f}")
print(f"Posterior w   = {cadeia[:,2].mean():.3f} ± {cadeia[:,2].std():.3f}")

pd.DataFrame({"z": z, "mu": mu, "sigma": sig}).to_csv("resultados/snia_catalogo.csv", index=False)
np.savetxt("resultados/mcmc_H0_Om_w.txt", cadeia, header="H0 Om w")

# Hubble diagram
z_plot = np.linspace(0.01, 1.3, 100)
mu_fid = modulo_distancia(z_plot, 0.3, 70, -1)
fig, ax = plt.subplots(figsize=(8,5))
ax.errorbar(z, mu, yerr=sig, fmt="k.", ms=4, alpha=0.6, label="SN Ia")
ax.plot(z_plot, mu_fid, "r-", lw=2, label="fiducial flat LCDM")
ax.set_xlabel("redshift z"); ax.set_ylabel(r"$\mu$ (mag)")
ax.set_title("Diagrama de Hubble – Supernovas Ia")
ax.legend(); ax.grid(True, alpha=0.3)
fig.savefig("figuras/01_hubble_diagram.png", dpi=150, bbox_inches="tight"); plt.close()

fig, axes = plt.subplots(1,3, figsize=(11,3.5))
for i, (lab, name) in enumerate(zip(["$H_0$", r"$\Omega_m$", r"$w$"], ["H0","Om","w"])):
    axes[i].hist(cadeia[:,i], bins=40, color="steelblue", density=True)
    axes[i].set_xlabel(lab)
fig.suptitle("Posteriors MCMC"); fig.tight_layout()
fig.savefig("figuras/02_posteriors.png", dpi=150, bbox_inches="tight"); plt.close()
print("Modelo Supernovas Ia concluído.")
