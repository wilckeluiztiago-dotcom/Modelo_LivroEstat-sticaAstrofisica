# Estatística Aplicada à Astrofísica

**Modelos quantitativos, inferência Bayesiana e algoritmos avançados**

**Autor:** Luiz Tiago Wilcke  
**Local:** Curitiba, Brasil  
**Ano:** 2026  

Baseado no livro:

> *Estatística Aplicada à Astrofísica – Métodos Quantitativos para a Descoberta Cosmológica*  
> Teoria consistente, aplicações práticas e algoritmos avançados  
> Luiz Tiago Wilcke, 2026

---

## Visão geral

Este repositório reúne **modelos estatísticos completos** aplicados a problemas reais de astrofísica e cosmologia. Cada modelo é modular, documentado com equações, usa dados/calibrações reais (Gaia, APOGEE, Kepler, TESS, LIGO, SDSS) e gera gráficos e catálogos.

Os métodos seguem os capítulos do livro:

| Capítulo | Tema |
|----------|------|
| 1–2 | Probabilidade, distribuições (Poisson, Gauss, power-law, …) |
| 3–4 | Inferência clássica e Bayesiana, MCMC, Fisher |
| 5 | Séries temporais, Lomb-Scargle, FAP |
| 7 | Monte Carlo |
| 9 | Machine Learning |
| 11 | Estatística espacial, TPCF, Landy-Szalay |
| 12 | Astrofísica multi-mensageira |
| 13 | Modelagem Bayesiana hierárquica |
| 15 | Processos de Dirichlet, ML probabilístico |

---

## Estrutura do repositório

```text
Modelo_LivroEstatisticaAstrofisica/
├── modelo_avancado_estrelas/     # Estrelas com metalicidade solar na Via Láctea
├── modelo_exoplanetas/           # Detecção de exoplanetas por trânsito
└── modelos_avancados_Wilcke/
    ├── 03_BAO/                   # Oscilações Acústicas de Bárions
    ├── 06_MultiMensageiro/       # GW + contraparte eletromagnética
    ├── 08_Dirichlet_Halo/        # Subpopulações do halo (Processo de Dirichlet)
    └── 09_Supernovas_Ia/         # Cosmologia com Supernovas Ia
```

---

## Modelos inclusos

### 1. Estrelas com metalicidade solar na Via Láctea
`modelo_avancado_estrelas/`

- Densidade com perfil radial quebrado (broken exponential)
- Gradiente de $[\mathrm{Fe/H}]$ calibrado em APOGEE
- Amostragem de Monte Carlo + MCMC dos parâmetros estruturais
- Matriz de Informação de Fisher
- Função de correlação de dois pontos (Landy-Szalay)

$$
\rho_{\mathrm{fino}}(R,z) = \rho_0\,\Sigma(R)\,\operatorname{sech}^{2}\!\left(\frac{z}{2z_d}\right)
$$

### 2. Detecção de exoplanetas por trânsito
`modelo_exoplanetas/`

- Curvas de luz estilo Kepler/TESS
- Periodograma de Lomb-Scargle + False Alarm Probability
- MCMC dos parâmetros orbitais ($P$, $R_p/R_\star$, $a/R_\star$, $b$)
- Kernel Quasi-Periódico (Processo Gaussiano) para atividade estelar

$$
\mathrm{FAP}(z) \approx 1 - \left(1 - e^{-z}\right)^{M}
$$

### 3. Oscilações Acústicas de Bárions (BAO)
`modelos_avancados_Wilcke/03_BAO/`

- Estimador de Landy-Szalay da TPCF
- Modelo com pico de BAO em $\sim 105\,h^{-1}\,\mathrm{Mpc}$

$$
\xi_{\mathrm{LS}}(r) = \frac{DD - 2\,DR + RR}{RR}
$$

### 4. Astrofísica multi-mensageira (GW + EM)
`modelos_avancados_Wilcke/06_MultiMensageiro/`

- Coincidência temporal e espacial
- Fusão de likelihoods
- Inspirado em GW170817 / GRB 170817A

$$
p(\mathrm{assoc} \mid d_{\mathrm{GW}}, d_{\mathrm{EM}}) \propto \mathcal{L}_{\mathrm{GW}}\,\mathcal{L}_{\mathrm{EM}}\,\pi(\mathrm{assoc})
$$

### 5. Subpopulações do halo – Processo de Dirichlet
`modelos_avancados_Wilcke/08_Dirichlet_Halo/`

- Stick-breaking
- Mistura infinita de populações estelares
- Dados tipo Gaia ($[\mathrm{Fe/H}]$, $v_\phi$)

$$
\pi_k = \beta_k \prod_{j=1}^{k-1}(1-\beta_j), \qquad \beta_j \sim \mathrm{Beta}(1,\alpha)
$$

### 6. Cosmologia com Supernovas Ia
`modelos_avancados_Wilcke/09_Supernovas_Ia/`

- Módulo de distância em cosmologia flat $w$CDM
- MCMC de $H_0$, $\Omega_m$ e $w$
- Diagrama de Hubble

$$
\mu(z) = 5\log_{10}\left(\frac{(1+z)\,\chi(z)}{10\,\mathrm{pc}}\right)
$$

---

## Como executar

Cada modelo é independente. Entre na pasta e rode o script principal:

```bash
# Exemplo: exoplanetas
cd modelo_exoplanetas
python principal_exoplanetas.py

# Exemplo: Supernovas Ia
cd modelos_avancados_Wilcke/09_Supernovas_Ia
python principal_snia.py
```

**Dependências comuns:** `numpy`, `scipy`, `pandas`, `matplotlib`

---

## Sobre o autor

**Luiz Tiago Wilcke**  
Astrofísica teórica e observacional · Métodos quantitativos · Cosmologia estatística  
Curitiba, Brasil – 2026

---

## Licença e uso

Obra de referência acadêmica e material didático de nível de graduação sênior / pós-graduação.  
Use livremente para estudo, ensino e pesquisa, citando o autor e o livro.
