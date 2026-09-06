#!/usr/bin/env python3
"""
principal_exoplanetas.py
------------------------
Modelo completo de detecção de exoplanetas por trânsito.

Métodos do livro Estatística Aplicada à Astrofísica (Luiz Tiago Wilcke):
- Lomb-Scargle + False Alarm Probability (Cap. 5)
- MCMC Metropolis-Hastings (Cap. 4)
- Processo Gaussiano Quasi-Periódico (Apêndice F)
- Monte Carlo e simulação de dados realistas (Cap. 7)

Dados calibrados em estatísticas reais do Kepler e TESS.

Autor: Luiz Tiago Wilcke – 2026
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from modulos.dados_reais_kepler import simular_sistema, gerar_catalogo_sistemas
from modulos.periodograma_lomb_scargle import detectar_periodo
from modulos.inferencia_bayesiana_exo import metropolis_hastings_exo
from modulos.visualizacao_exo import (
    plot_curva_luz, plot_periodograma, plot_faseada,
    plot_corner_exo, plot_deteccoes, plot_gp_exemplo
)

print("=" * 72)
print("  MODELO DE DETECÇÃO DE EXOPLANETAS POR TRÂNSITO")
print("  Lomb-Scargle · FAP · MCMC · Processo Gaussiano · Dados Kepler/TESS")
print("  Autor: Luiz Tiago Wilcke – 2026")
print("=" * 72)

os.makedirs("resultados", exist_ok=True)
os.makedirs("figuras", exist_ok=True)

# ------------------------------------------------------------
# 1. Simulação de sistemas calibrados em dados reais
# ------------------------------------------------------------
print("\n[1/6] Gerando catálogo de sistemas sintéticos (estilo Kepler/TESS)...")
catalogo = gerar_catalogo_sistemas(n_sistemas=5, semente=2026)
print(f"      {len(catalogo)} sistemas gerados")

# ------------------------------------------------------------
# 2. Detecção via Lomb-Scargle + FAP
# ------------------------------------------------------------
print("\n[2/6] Rodando periodograma de Lomb-Scargle e cálculo de FAP...")
resultados_deteccao = []
for i, params in enumerate(catalogo):
    t, fluxo, _ = simular_sistema(i, duracao=75.0, ruido=0.00012, semente=100+i)
    det = detectar_periodo(t, fluxo, fmin=0.02, fmax=1.5, n_freq=2000, limiar_fap=0.01)
    det["nome"] = params["id"]
    det["periodo_verdadeiro"] = params["periodo"]
    det["rp_rs_verdadeiro"] = params["rp_rs"]
    resultados_deteccao.append(det)
    status = "DETECTADO" if det["detectado"] else "não detectado"
    print(f"      {params['id']}: P_true={params['periodo']:.2f} d | "
          f"P_det={det['periodo']:.2f} d | FAP={det['fap']:.2e} → {status}")

# ------------------------------------------------------------
# 3. Análise detalhada de um sistema de referência
# ------------------------------------------------------------
print("\n[3/6] Análise detalhada do sistema de referência (MCMC)...")
t_ref, fluxo_ref, params_ref = simular_sistema(0, duracao=90.0, ruido=0.00010, semente=42)
det_ref = detectar_periodo(t_ref, fluxo_ref, fmin=0.02, fmax=1.2, n_freq=2500)

print(f"      Período verdadeiro : {params_ref['periodo']:.4f} d")
print(f"      Período detectado  : {det_ref['periodo']:.4f} d")
print(f"      FAP                : {det_ref['fap']:.3e}")
print(f"      Potência máxima    : {det_ref['potencia_max']:.3f}")

# MCMC
print("      Executando MCMC Metropolis-Hastings...")
amostra, taxa = metropolis_hastings_exo(t_ref, fluxo_ref, n_passos=2500, n_burn=500, semente=7)
print(f"      Taxa de aceitação MCMC : {taxa:.1%}")
print(f"      Posterior P     = {amostra[:,0].mean():.3f} ± {amostra[:,0].std():.3f} d")
print(f"      Posterior Rp/R* = {amostra[:,2].mean():.4f} ± {amostra[:,2].std():.4f}")
print(f"      Posterior a/R*  = {amostra[:,3].mean():.2f} ± {amostra[:,3].std():.2f}")
print(f"      Posterior b     = {amostra[:,4].mean():.3f} ± {amostra[:,4].std():.3f}")

# ------------------------------------------------------------
# 4. Salvar catálogo de resultados
# ------------------------------------------------------------
print("\n[4/6] Salvando resultados...")
df = pd.DataFrame({
    "id": [r["nome"] for r in resultados_deteccao],
    "periodo_verdadeiro_d": [r["periodo_verdadeiro"] for r in resultados_deteccao],
    "periodo_detectado_d": [r["periodo"] for r in resultados_deteccao],
    "potencia_max": [r["potencia_max"] for r in resultados_deteccao],
    "fap": [r["fap"] for r in resultados_deteccao],
    "detectado": [r["detectado"] for r in resultados_deteccao],
    "rp_rs_verdadeiro": [r["rp_rs_verdadeiro"] for r in resultados_deteccao]
})
df.to_csv("resultados/catalogo_deteccoes_exoplanetas.csv", index=False, float_format="%.6f")
print("      resultados/catalogo_deteccoes_exoplanetas.csv")

# Salvar cadeia MCMC
np.savetxt("resultados/cadeia_mcmc_parametros.txt", amostra,
           header="P t0 rp_rs a_rs b", comments="# ")
print("      resultados/cadeia_mcmc_parametros.txt")

# ------------------------------------------------------------
# 5. Gráficos
# ------------------------------------------------------------
print("\n[5/6] Gerando gráficos científicos...")
plot_curva_luz(t_ref, fluxo_ref, params_ref, "figuras/01_curva_luz.png")
plot_periodograma(det_ref["freqs"], det_ref["potencia"],
                  periodo_true=params_ref["periodo"], path="figuras/02_periodograma.png")
plot_faseada(t_ref, fluxo_ref, det_ref["periodo"], params_ref["t0"], "figuras/03_faseada.png")
plot_corner_exo(amostra, "figuras/04_corner_mcmc.png")
plot_deteccoes(resultados_deteccao, "figuras/05_resumo_deteccoes.png")
plot_gp_exemplo(t_ref, fluxo_ref, "figuras/06_gp_atividade.png")

# ------------------------------------------------------------
# 6. Resumo final
# ------------------------------------------------------------
n_det = sum(1 for r in resultados_deteccao if r["detectado"])
print("\n" + "=" * 72)
print("RESULTADOS NUMÉRICOS PRINCIPAIS")
print("=" * 72)
print(f"Sistemas simulados              : {len(catalogo)}")
print(f"Detecções com FAP < 1%          : {n_det}/{len(catalogo)}")
print(f"Taxa de recuperação             : {100*n_det/len(catalogo):.0f}%")
print(f"Sistema de referência – FAP     : {det_ref['fap']:.3e}")
print(f"Posterior MCMC (P)              : {amostra[:,0].mean():.3f} ± {amostra[:,0].std():.3f} d")
print(f"Posterior MCMC (Rp/R*)          : {amostra[:,2].mean():.4f} ± {amostra[:,2].std():.4f}")
print("=" * 72)
print("Arquivos em resultados/ e figuras/")
print("Modelo de detecção de exoplanetas concluído com sucesso.")
print("=" * 72)
