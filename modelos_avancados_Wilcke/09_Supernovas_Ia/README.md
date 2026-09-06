# Cosmologia com Supernovas Ia (Escada de Distâncias)

**Autor:** Luiz Tiago Wilcke – 2026  
**Capítulos de referência:** 4 e 13 (Bayes + Hierarchical)

## Equações principais

Parâmetro de Hubble normalizado (flat $w$CDM):

$$
E(z) = \sqrt{\Omega_m (1+z)^{3} + (1-\Omega_m)(1+z)^{3(1+w)}}
$$

Distância comóvel:

$$
\chi(z) = \frac{c}{H_0} \int_0^{z} \frac{dz'}{E(z')}
$$

Módulo de distância:

$$
\mu(z) = 5\log_{10}\left(\frac{(1+z)\,\chi(z)}{10\,\mathrm{pc}}\right)
$$

Likelihood:

$$
\mathcal{L}(H_0, \Omega_m, w) \propto \exp\left( -\frac{1}{2} \sum_i \frac{(\mu_i^{\mathrm{obs}} - \mu(z_i))^{2}}{\sigma_i^{2}} \right)
$$

## Execução

```bash
python principal_snia.py
```
