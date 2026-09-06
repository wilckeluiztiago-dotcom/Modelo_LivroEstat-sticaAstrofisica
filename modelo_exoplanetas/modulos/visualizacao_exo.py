"""
visualizacao_exo.py
-------------------
Gráficos científicos para detecção de exoplanetas.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "savefig.bbox": "tight"
})


def plot_curva_luz(t, fluxo, params=None, path="figuras/01_curva_luz.png"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, fluxo, "k.", ms=2, alpha=0.6, label="dados")
    if params is not None:
        from modulos.modelo_transito import modelo_transito_suave
        modelo = modelo_transito_suave(t, params["t0"], params["periodo"],
                                      params["rp_rs"], params["a_rs"], params["b"])
        ax.plot(t, modelo, "r-", lw=1.5, label="modelo de trânsito")
    ax.set_xlabel("Tempo (dias)")
    ax.set_ylabel("Fluxo relativo")
    ax.set_title("Curva de luz sintética calibrada em Kepler/TESS")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(fluxo.min()-0.001, fluxo.max()+0.001)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_periodograma(freqs, pot, periodo_true=None, path="figuras/02_periodograma.png"):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(1/freqs, pot, "b-", lw=0.9)
    if periodo_true is not None:
        ax.axvline(periodo_true, color="r", ls="--", lw=1.5, label=f"P verdadeiro = {periodo_true:.2f} d")
    ax.set_xlabel("Período (dias)")
    ax.set_ylabel("Potência Lomb-Scargle (normalizada)")
    ax.set_title("Periodograma de Lomb-Scargle")
    ax.set_xlim(0.5, 40)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


def plot_faseada(t, fluxo, periodo, t0, path="figuras/03_faseada.png"):
    fase = ((t - t0 + periodo/2) % periodo) / periodo - 0.5
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(fase, fluxo, "k.", ms=3, alpha=0.5)
    ax.set_xlabel("Fase orbital")
    ax.set_ylabel("Fluxo relativo")
    ax.set_title(f"Curva de luz faseada (P = {periodo:.3f} d)")
    ax.set_xlim(-0.5, 0.5)
    fig.savefig(path)
    plt.close(fig)


def plot_corner_exo(amostra, path="figuras/04_corner_mcmc.png"):
    labels = [r"$P$ (d)", r"$t_0$", r"$R_p/R_\star$", r"$a/R_\star$", r"$b$"]
    n = amostra.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(amostra[:, i], bins=35, color="steelblue", density=True)
                ax.set_xlabel(labels[i])
            elif i > j:
                ax.hist2d(amostra[:, j], amostra[:, i], bins=30, cmap="Blues")
                ax.set_xlabel(labels[j])
                ax.set_ylabel(labels[i])
            else:
                ax.axis("off")
    fig.suptitle("Posterior MCMC – parâmetros de trânsito", y=1.01)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_deteccoes(resultados, path="figuras/05_resumo_deteccoes.png"):
    """Gráfico resumo de FAPs e períodos detectados."""
    periodos = [r["periodo"] for r in resultados]
    faps = [r["fap"] for r in resultados]
    nomes = [r.get("nome", f"#{i}") for i, r in enumerate(resultados)]

    fig, ax = plt.subplots(figsize=(9, 5))
    cores = ["green" if f < 0.01 else "orange" if f < 0.05 else "red" for f in faps]
    ax.scatter(periodos, -np.log10(np.clip(faps, 1e-10, 1)), c=cores, s=80, edgecolors="k")
    ax.axhline(-np.log10(0.01), color="g", ls="--", label="FAP = 1%")
    ax.axhline(-np.log10(0.05), color="orange", ls="--", label="FAP = 5%")
    ax.set_xlabel("Período detectado (dias)")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{FAP})$")
    ax.set_title("Resumo de detecções – False Alarm Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


def plot_gp_exemplo(t, y, path="figuras/06_gp_atividade.png"):
    """Exemplo visual de kernel quasi-periódico (atividade estelar)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, y, "k.", ms=2, alpha=0.5, label="curva de luz")
    # Suavização visual
    from scipy.ndimage import gaussian_filter1d
    suav = gaussian_filter1d(y, sigma=15)
    ax.plot(t, suav, "r-", lw=1.5, label="tendência (atividade)")
    ax.set_xlabel("Tempo (dias)")
    ax.set_ylabel("Fluxo")
    ax.set_title("Atividade estelar modelada por Processo Gaussiano (kernel QP)")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
