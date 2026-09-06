# Detecção de Oscilações Acústicas de Bárions (BAO)

**Autor:** Luiz Tiago Wilcke – 2026  
**Capítulo de referência:** 11 (Estatística Espacial)

## Equações principais

Estimador de Landy-Szalay:

$$
\xi_{\mathrm{LS}}(r) = \frac{DD(r) - 2\,DR(r) + RR(r)}{RR(r)}
$$

Modelo empírico com pico de BAO:

$$
\xi(r) = B\left(\frac{r}{r_0}\right)^{-\gamma} + A\exp\left(-\frac{(r - r_{\mathrm{BAO}})^{2}}{2\sigma^{2}}\right)
$$

O pico caracteriza a escala de som no Universo primitivo ($\sim 105\,h^{-1}\,\mathrm{Mpc}$).

## Execução

```bash
python principal_bao.py
```
