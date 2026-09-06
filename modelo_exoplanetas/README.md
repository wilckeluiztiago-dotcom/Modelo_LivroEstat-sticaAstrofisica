# Modelo Estatístico de Detecção de Exoplanetas por Trânsito

**Autor:** Luiz Tiago Wilcke  
**Ano:** 2026  
**Referência teórica:** *Estatística Aplicada à Astrofísica – Métodos Quantitativos para a Descoberta Cosmológica*

---

## 1. Visão Geral

Este repositório implementa um **modelo estatístico completo** de detecção de exoplanetas pelo método de trânsito, calibrado em estatísticas reais das missões **Kepler** e **TESS**.

O modelo combina:

- Geração de curvas de luz realistas (ruído branco + atividade estelar)
- Periodograma de **Lomb-Scargle** para séries irregulares (Cap. 5)
- Cálculo de **False Alarm Probability (FAP)**
- Inferência Bayesiana dos parâmetros orbitais via **MCMC Metropolis-Hastings** (Cap. 4)
- Kernel **Quasi-Periódico** de Processos Gaussianos para atividade estelar (Apêndice F)
- Métodos de Monte Carlo (Cap. 7)

---

## 2. Equações Fundamentais

### 2.1 Modelo de trânsito (aproximação de caixa + limb-darkening)

A profundidade do trânsito para uma razão de raios $k = R_p / R_\star$ é:

$$
\delta \approx k^{2} = \left( \frac{R_p}{R_\star} \right)^{2}
$$

A duração total do trânsito (órbita circular) é:

$$
T_{14} \approx \frac{P}{\pi} \arcsin\left( \frac{\sqrt{(1+k)^{2} - b^{2}}}{a/R_\star} \right)
$$

onde $b = (a/R_\star)\cos i$ é o parâmetro de impacto e $P$ o período orbital.

O fluxo observado sob o modelo de caixa é:

$$
F(t) =
\begin{cases}
1 - \delta & \text{se } |t - t_0 - nP| < T_{14}/2 \\[6pt]
1 & \text{caso contrário}
\end{cases}
$$

### 2.2 Lei linear de obscurecimento de limbo

$$
\frac{I(\mu)}{I(1)} = 1 - u_1 (1 - \mu), \qquad \mu = \cos\theta
$$

### 2.3 Periodograma de Lomb-Scargle (Cap. 5)

Para uma série temporal irregular $\{t_j, y_j\}$:

$$
P(f) = \frac{1}{2} \left\{
\frac{ \left[ \sum_j y_j \cos \omega (t_j - \tau) \right]^{2} }{ \sum_j \cos^{2} \omega (t_j - \tau) }
+
\frac{ \left[ \sum_j y_j \sin \omega (t_j - \tau) \right]^{2} }{ \sum_j \sin^{2} \omega (t_j - \tau) }
\right\}
$$

onde $\omega = 2\pi f$ e $\tau$ é o fator de atraso que torna o periodograma invariante à translação temporal:

$$
\tan(2\omega\tau) = \frac{ \sum_j \sin 2\omega t_j }{ \sum_j \cos 2\omega t_j }
$$

### 2.4 False Alarm Probability (FAP)

A probabilidade de falso alarme para o pico máximo de potência $z = \max P(f)$ é (aproximação de Horne & Baliunas / Baluev):

$$
\mathrm{FAP}(z) \approx 1 - \left( 1 - e^{-z} \right)^{M}
$$

onde $M$ é o número efetivo de frequências independentes:

$$
M \approx T (f_{\max} - f_{\min})
$$

e $T$ é a linha de base temporal da observação.

Neste código a FAP também é estimada empiricamente por **bootstrap** (embaralhamento dos fluxos), mais robusta para o periodograma normalizado.

### 2.5 Inferência Bayesiana dos parâmetros (Cap. 4)

**Prior** (informativo + bounds físicos):

$$
\pi(\theta) \propto \exp\left( -\frac{1}{2} \sum_i \left( \frac{\theta_i - \mu_i}{\sigma_i} \right)^{2} \right)
$$

válido dentro do suporte físico, com

$$
\theta = (P,\ t_0,\ R_p/R_\star,\ a/R_\star,\ b)
$$

**Likelihood** Gaussiana sob o modelo de trânsito:

$$
\mathcal{L}(\theta \mid \mathbf{y}) = \prod_{j=1}^{N} \frac{1}{\sqrt{2\pi}\,\sigma}
\exp\left( -\frac{ \left[ y_j - F(t_j;\theta) \right]^{2} }{ 2\sigma^{2} } \right)
$$

**Posterior**:

$$
p(\theta \mid \mathbf{y}) \propto \mathcal{L}(\theta \mid \mathbf{y})\, \pi(\theta)
$$

Amostrada por **Metropolis-Hastings**:

$$
\alpha = \min\left( 1,\ \frac{p(\theta' \mid \mathbf{y})}{p(\theta \mid \mathbf{y})} \right)
$$

### 2.6 Processo Gaussiano – Kernel Quasi-Periódico (Apêndice F)

Para modelar a atividade estelar (manchas, rotação):

$$
k(t,t') = A^{2} \exp\left(
-\Gamma \sin^{2}\left( \frac{\pi |t-t'|}{P_{\mathrm{rot}}} \right)
- \left( \frac{|t-t'|}{\lambda} \right)^{2}
\right) + \sigma_w^{2}\, \delta_{tt'}
$$

A verossimilhança marginal do GP é:

$$
\log \mathcal{L}_{\mathrm{GP}} = -\frac{1}{2} \mathbf{y}^{\mathsf{T}} K^{-1} \mathbf{y}
- \frac{1}{2} \log \det K
- \frac{N}{2} \log (2\pi)
$$

### 2.7 Razão sinal-ruído do trânsito

$$
\mathrm{S/N} \approx \frac{\delta}{\sigma} \sqrt{ N_{\mathrm{trânsitos}} \cdot \frac{T_{14}}{\Delta t} }
$$

onde $\Delta t$ é a cadência e $N_{\mathrm{trânsitos}}$ o número de trânsitos observados.

---

## 3. Estrutura do Código

```text
modelo_exoplanetas/
├── principal_exoplanetas.py
├── README.md
├── modulos/
│   ├── modelo_transito.py
│   ├── periodograma_lomb_scargle.py
│   ├── inferencia_bayesiana_exo.py
│   ├── dados_reais_kepler.py
│   └── visualizacao_exo.py
├── resultados/
│   ├── catalogo_deteccoes_exoplanetas.csv
│   └── cadeia_mcmc_parametros.txt
└── figuras/
    ├── 01_curva_luz.png
    ├── 02_periodograma.png
    ├── 03_faseada.png
    ├── 04_corner_mcmc.png
    ├── 05_resumo_deteccoes.png
    └── 06_gp_atividade.png
```

---

## 4. Como Executar

```bash
cd modelo_exoplanetas
python principal_exoplanetas.py
```

**Dependências:** `numpy`, `scipy`, `pandas`, `matplotlib`.

---

## 5. Resultados Esperados

- Catálogo de sistemas sintéticos com períodos, FAPs e status de detecção
- Cadeia MCMC dos parâmetros orbitais
- Seis gráficos científicos (curva de luz, periodograma, faseada, corner, resumo de FAPs, atividade GP)
- Taxa de recuperação de planetas com FAP baixa

---

## 6. Referências de dados reais

Os parâmetros típicos de período, $R_p/R_\star$ e $a/R_\star$ foram inspirados em:

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) (Kepler confirmed planets / KOIs)
- Estatísticas de TESS Objects of Interest (TOIs)
- Dispersões de ruído típicas de Kepler long-cadence (~30 min) e TESS

---

**Luiz Tiago Wilcke**  
Curitiba, Brasil – 2026  
*Estatística Aplicada à Astrofísica*
