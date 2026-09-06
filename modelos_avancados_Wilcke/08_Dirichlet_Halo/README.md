# Subpopulações do Halo Galáctico – Processo de Dirichlet

**Autor:** Luiz Tiago Wilcke – 2026  
**Capítulo de referência:** 15 (Aprendizado de Máquina Probabilístico)

## Equações principais

Construção stick-breaking:

$$
\pi_k = \beta_k \prod_{j=1}^{k-1}(1 - \beta_j), \qquad \beta_j \sim \mathrm{Beta}(1, \alpha)
$$

Processo de Dirichlet:

$$
G \sim \mathrm{DP}(\alpha, G_0)
$$

Modelo de mistura infinito:

$$
x_i \mid z_i = k \sim \mathcal{N}(\mu_k, \Sigma_k), \qquad z_i \sim \mathrm{Categorical}(\pi)
$$

Aplicação: descoberta de subestruturas no halo com dados tipo Gaia ($[\mathrm{Fe/H}]$, $v_\phi$).

## Execução

```bash
python principal_dirichlet.py
```
