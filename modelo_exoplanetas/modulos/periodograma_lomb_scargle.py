"""
periodograma_lomb_scargle.py
----------------------------
Periodograma de Lomb-Scargle para séries temporais irregulares
(Capítulo 5 do livro – Análise de Séries Temporais).

Implementação do periodograma normalizado + False Alarm Probability (FAP).
Autor: Luiz Tiago Wilcke – 2026
"""

import numpy as np
from scipy.signal import lombscargle


def periodograma_lomb_scargle(t, y, fmin=None, fmax=None, n_freq=3000, normalizado=True):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)

    dt = np.median(np.diff(t))
    if fmin is None:
        fmin = 1.0 / (t.max() - t.min())
    if fmax is None:
        fmax = 0.5 / dt

    freqs = np.linspace(fmin, fmax, n_freq)
    angular = 2 * np.pi * freqs
    potencia = lombscargle(t, y, angular, normalize=normalizado)
    return freqs, potencia


def periodo_pico(freqs, potencia):
    idx = np.argmax(potencia)
    return 1.0 / freqs[idx], potencia[idx], freqs[idx]


def fap_bootstrap(t, y, pot_max, fmin, fmax, n_freq=1500, n_boot=25, semente=123):
    """
    FAP empírica por bootstrap (embaralhamento dos fluxos).
    Mais robusta para o periodograma normalizado do scipy.
    """
    rng = np.random.default_rng(semente)
    contagem = 0
    for _ in range(n_boot):
        y_shuf = rng.permutation(y)
        _, pot = periodograma_lomb_scargle(t, y_shuf, fmin=fmin, fmax=fmax, n_freq=n_freq)
        if pot.max() >= pot_max:
            contagem += 1
    return (contagem + 1) / (n_boot + 1)


def detectar_periodo(t, y, fmin=0.02, fmax=1.0, n_freq=2000, limiar_fap=0.05):
    freqs, pot = periodograma_lomb_scargle(t, y, fmin=fmin, fmax=fmax, n_freq=n_freq)
    P, pot_max, f_max = periodo_pico(freqs, pot)

    # FAP bootstrap (mais realista para potência normalizada)
    fap = fap_bootstrap(t, y, pot_max, fmin, fmax, n_freq=min(n_freq, 1200), n_boot=25)

    # Critério de detecção: FAP baixa OU pico claramente acima da mediana
    mediana = np.median(pot)
    detectado = (fap < limiar_fap) or (pot_max > 8 * mediana)

    return {
        "periodo": P,
        "frequencia": f_max,
        "potencia_max": pot_max,
        "fap": fap,
        "detectado": detectado,
        "freqs": freqs,
        "potencia": pot
    }
