# Modelo Estatístico Avançado de Estrelas com Metalicidade Solar na Via Láctea

**Autor:** Luiz Tiago Wilcke  
**Ano:** 2026  
**Referência teórica:** *Estatística Aplicada à Astrofísica – Métodos Quantitativos para a Descoberta Cosmológica*

---

## 1. Visão Geral

Este repositório implementa um **modelo estatístico avançado** da distribuição espacial e química de estrelas com metalicidade próxima à solar ([Fe/H] ≈ 0) na Via Láctea.  

O modelo é fundamentado em:

- Teoria da medida e axiomas de Kolmogorov (Cap. 1)
- Distribuições de probabilidade e densidades (Cap. 2)
- Inferência clássica e matriz de Informação de Fisher (Cap. 3 e 10)
- Inferência Bayesiana e MCMC Metropolis-Hastings (Cap. 4)
- Métodos de Monte Carlo e amostragem por rejeição (Cap. 7)
- Estatística espacial e função de correlação de dois pontos – estimador de Landy-Szalay (Cap. 11)

Parâmetros estruturais foram calibrados em resultados publicados de **Gaia DR3** e **APOGEE DR17**.

---

## 2. Equações Fundamentais do Modelo

### 2.1 Densidade do disco fino com perfil radial quebrado (broken exponential)

A densidade volumétrica do disco fino (população low-α de metalicidade solar) é:

\[
\rho_{\rm fino}(R,z) = \rho_0\,\Sigma(R)\,\operatorname{sech}^{2}\!\left(\frac{z}{2z_{d}}\right)
\]

onde o perfil de superfície radial é quebrado em \(R_{\rm break}\):

\[
\Sigma(R) =
\begin{cases}
\exp\!\left(-\dfrac{R}{R_{d,{\rm in}}}\right) & R\le R_{\rm break}\\[8pt]
\exp\!\left(-\dfrac{R_{\rm break}}{R_{d,{\rm in}}}\right)\exp\!\left(-\dfrac{R-R_{\rm break}}{R_{d,{\rm out}}}\right) & R > R_{\rm break}
\end{cases}
\]

Valores adotados (calibrados em literatura Gaia/APOGEE para [Fe/H]≈0):

- \(R_{d,{\rm in}} = 2{,}2\,\mathrm{kpc}\)
- \(R_{d,{\rm out}} = 3{,}5\,\mathrm{kpc}\)
- \(R_{\rm break} = 8{,}5\,\mathrm{kpc}\)
- \(z_{d} = 0{,}28\,\mathrm{kpc}\)

A forma \(\operatorname{sech}^{2}\) é a solução de equilíbrio de um disco isotérmico (Spitzer).

### 2.2 Contribuição residual do disco espesso

\[
\rho_{\rm espesso}(R,z) = f_{\rm esp}\,\rho_0\exp\!\left(-\frac{R}{R_{d,{\rm esp}}}\right)\operatorname{sech}^{2}\!\left(\frac{z}{2z_{d,{\rm esp}}}\right)
\]

com \(f_{\rm esp}\approx 0{,}07\), \(R_{d,{\rm esp}}=2{,}8\,\mathrm{kpc}\), \(z_{d,{\rm esp}}=0{,}85\,\mathrm{kpc}\).

### 2.3 Densidade total

\[
\rho_{\rm total}(R,z) = \rho_{\rm fino}(R,z) + \rho_{\rm espesso}(R,z)
\]

### 2.4 Gradiente radial de metalicidade

Calibrado em APOGEE (≈ −0,06 dex kpc⁻¹ no disco fino):

\[
\langle[\mathrm{Fe/H}]\rangle(R) = 0{,}00 - 0{,}055\,(R - R_{\odot})\qquad[\mathrm{dex}]
\]

com dispersão intrínseca

\[
[\mathrm{Fe/H}] \;\sim\; \mathcal{N}\bigl(\langle[\mathrm{Fe/H}]\rangle(R),\;\sigma=0{,}075\bigr)
\]

### 2.5 Amostragem de Monte Carlo (método de rejeição)

Proposta uniforme no volume cilíndrico \(\to\) aceitação com probabilidade proporcional a \(\rho_{\rm total}\):

\[
u\sim\mathcal{U}(0,\rho_{\max}),\qquad\text{aceita se }u < \rho_{\rm total}(R,z)
\]

### 2.6 Inferência Bayesiana (MCMC)

**Prior** (informativo + suporte físico):

\[
\pi(\theta) \propto \exp\!\Bigl(-\tfrac12\sum_i\bigl(\tfrac{\theta_i-\mu_i}{\sigma_i}\bigr)^2\Bigr)
\quad\text{dentro dos bounds físicos}
\]

**Likelihood** (produto das densidades):

\[
\mathcal{L}(\theta\mid\{R_i,z_i\}) \propto \prod_i\rho(R_i,z_i;\theta)
\]

**Posterior**:

\[
p(\theta\mid\mathrm{dados}) \propto \mathcal{L}(\theta\mid\mathrm{dados})\,\pi(\theta)
\]

Amostrada por **Metropolis-Hastings** (Cap. 4).

### 2.7 Matriz de Informação de Fisher

\[
\mathcal{I}_{ij}(\theta) = -\mathbb{E}\!\left[\frac{\partial^{2}\log\mathcal{L}}{\partial\theta_i\partial\theta_j}\right]
\]

A inversa fornece a cota de Cramér-Rao para as incertezas dos parâmetros.

### 2.8 Função de Correlação de Dois Pontos – estimador de Landy-Szalay

\[
\xi_{\rm LS}(r) = \frac{DD(r)-2\,DR(r)+RR(r)}{RR(r)}
\]

onde \(DD\), \(DR\) e \(RR\) são as contagens de pares normalizadas (dados-dados, dados-random, random-random).

---

## 3. Estrutura dos Módulos

```
modelo_avancado_estrelas/
├── principal_avancado.py
├── README.md
├── modulos/
│   ├── densidade_espacial.py
│   ├── amostragem_avancada.py
│   ├── inferencia_bayesiana.py
│   ├── estatistica_espacial.py
│   └── visualizacao_avancada.py
├── resultados/
│   └── catalogo_avancado_estrelas_solar.csv
└── figuras/
    ├── 01_face_on.png
    ├── 02_edge_on.png
    ├── 03_gradiente_feh.png
    ├── 04_perfis.png
    ├── 05_tpcf_landy_szalay.png
    └── 06_corner_mcmc.png
```

---

## 4. Execução

```bash
cd modelo_avancado_estrelas
python principal_avancado.py
```

Dependências: `numpy`, `scipy`, `pandas`, `matplotlib`.

---

## 5. Resultados Esperados

- Catálogo de ~75 000 estrelas com coordenadas galactocêntricas e heliocêntricas, [Fe/H] e classificação fino/espesso.
- Posteriors MCMC de \(R_{d,{\rm in}}\), \(R_{d,{\rm out}}\) e \(z_d\).
- Incertezas de Fisher.
- Função de correlação \(\xi(r)\) via Landy-Szalay.
- Seis gráficos científicos de alta qualidade.

---

**Luiz Tiago Wilcke**  
Curitiba, Brasil – 2026  
*Estatística Aplicada à Astrofísica*
