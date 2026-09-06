"""
visualizacao_avancada.py
------------------------
Gráficos científicos avançados do modelo.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "savefig.bbox": "tight"
})


def plot_face_on(cat, path):
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    hb = ax.hist2d(cat["x"], cat["y"], bins=140, range=[[-17,17],[-17,17]],
                   norm=LogNorm(), cmap="inferno")
    ax.set_xlabel(r"$X$ galactocêntrico (kpc)")
    ax.set_ylabel(r"$Y$ galactocêntrico (kpc)")
    ax.set_title("Distribuição face-on – estrelas [Fe/H]≈0 (modelo avançado)")
    ax.set_aspect("equal")
    ax.plot(0,0,"w+",ms=14,mew=2,label="Centro")
    ax.plot(-8.2,0,"c*",ms=11,label="Sol")
    ax.legend(loc="upper right",fontsize=9)
    fig.colorbar(hb[3], ax=ax, label="contagem / bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path); plt.close(fig)


def plot_edge_on(cat, path):
    fig, ax = plt.subplots(figsize=(9,4.8))
    hb = ax.hist2d(cat["R"], cat["z"], bins=110, range=[[0,17],[-2.5,2.5]],
                   norm=LogNorm(), cmap="magma")
    ax.set_xlabel(r"$R$ galactocêntrico (kpc)")
    ax.set_ylabel(r"$z$ (kpc)")
    ax.set_title("Vista edge-on + perfil vertical")
    ax.axhline(0,color="w",ls="--",lw=0.7)
    fig.colorbar(hb[3], ax=ax, label="contagem / bin")
    fig.savefig(path); plt.close(fig)


def plot_feh_vs_R(cat, path):
    fig, ax = plt.subplots(figsize=(8,5))
    # densidade 2D
    hb = ax.hist2d(cat["R"], cat["feh"], bins=80, range=[[0.5,16],[-0.45,0.35]],
                   norm=LogNorm(), cmap="viridis")
    # gradiente médio
    r_bins = np.linspace(1,15,12)
    medians = []
    for i in range(len(r_bins)-1):
        m = (cat["R"]>=r_bins[i]) & (cat["R"]<r_bins[i+1])
        medians.append(np.median(cat["feh"][m]) if m.sum()>20 else np.nan)
    rc = 0.5*(r_bins[:-1]+r_bins[1:])
    ax.plot(rc, medians, "r-o", lw=2, ms=5, label="mediana por bin")
    ax.axhline(0, color="w", ls=":", lw=1)
    ax.set_xlabel(r"$R$ (kpc)")
    ax.set_ylabel(r"[Fe/H]")
    ax.set_title("Gradiente radial de metalicidade (calibrado em APOGEE)")
    ax.legend()
    fig.colorbar(hb[3], ax=ax, label="contagem")
    fig.savefig(path); plt.close(fig)


def plot_tpcf(r, xi, path):
    fig, ax = plt.subplots(figsize=(7.5,5.5))
    ax.loglog(r, np.abs(xi)+1e-4, "o-", color="darkblue", ms=5, label=r"$|\xi(r)|$ (Landy-Szalay)")
    ax.set_xlabel(r"$r$ (kpc)")
    ax.set_ylabel(r"$|\xi(r)|$")
    ax.set_title("Função de Correlação de Dois Pontos (TPCF)\nEstimador de Landy-Szalay")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(path); plt.close(fig)


def plot_corner_mcmc(amostra, path):
    """Corner plot simplificado (3 parâmetros)."""
    labels = [r"$R_{d,\rm in}$ (kpc)", r"$R_{d,\rm out}$ (kpc)", r"$z_d$ (kpc)"]
    fig, axes = plt.subplots(3,3, figsize=(9,9))
    for i in range(3):
        for j in range(3):
            ax = axes[i,j]
            if i == j:
                ax.hist(amostra[:,i], bins=40, color="steelblue", density=True)
                ax.set_xlabel(labels[i])
            elif i > j:
                ax.hist2d(amostra[:,j], amostra[:,i], bins=35, cmap="Blues")
                ax.set_xlabel(labels[j])
                ax.set_ylabel(labels[i])
            else:
                ax.axis("off")
    fig.suptitle("Posterior MCMC – parâmetros estruturais do disco", y=1.01)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def plot_perfil_vertical_radial(cat, path):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,4.5))

    # Vertical no anel solar
    m = (cat["R"]>7.0) & (cat["R"]<9.5)
    z = cat["z"][m]
    bins = np.linspace(-2.2,2.2,45)
    h, e = np.histogram(z, bins=bins, density=True)
    c = 0.5*(e[:-1]+e[1:])
    ax1.semilogy(c, h, "s-", color="purple", ms=4, label="dados modelo")
    zteo = np.linspace(-2.2,2.2,200)
    dteo = 1/np.cosh(zteo/(2*0.28))**2
    dteo /= dteo.max(); dteo *= h.max()
    ax1.plot(zteo, dteo, "k--", lw=2, label=r"sech² teórico")
    ax1.set_xlabel(r"$z$ (kpc)"); ax1.set_ylabel("densidade")
    ax1.set_title("Perfil vertical (7 < R < 9.5 kpc)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Radial
    m2 = np.abs(cat["z"]) < 0.4
    R = cat["R"][m2]
    binsR = np.linspace(0.8,16,30)
    hR, eR = np.histogram(R, bins=binsR)
    cR = 0.5*(eR[:-1]+eR[1:])
    area = np.pi*(eR[1:]**2 - eR[:-1]**2)
    dens = hR / area
    ax2.semilogy(cR, dens, "o-", color="darkorange", ms=4)
    ax2.set_xlabel(r"$R$ (kpc)"); ax2.set_ylabel(r"$\Sigma$ relativa")
    ax2.set_title("Perfil radial (|z|<0.4 kpc)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path); plt.close(fig)
