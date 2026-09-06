# Inferência Estatística Multi-Mensageira (GW + EM)

**Autor:** Luiz Tiago Wilcke – 2026  
**Capítulo de referência:** 12

## Equações principais

Coincidência temporal:

$$
P(\mathrm{assoc} \mid \Delta t) \propto \frac{\mathcal{L}_{\mathrm{assoc}}(\Delta t)}{\mathcal{L}_{\mathrm{assoc}} + \mathcal{L}_{\mathrm{fundo}}}
$$

Mapa de probabilidade na esfera (aproximação Gaussiana):

$$
p(\alpha,\delta) \propto \exp\left(-\frac{1}{2}\frac{\theta^{2}}{\sigma^{2}}\right)
$$

Fusão de likelihoods:

$$
p(\mathrm{assoc} \mid d_{\mathrm{GW}}, d_{\mathrm{EM}}) \propto \mathcal{L}_{\mathrm{GW}}\,\mathcal{L}_{\mathrm{EM}}\,\pi(\mathrm{assoc})
$$

Inspirado no evento **GW170817 / GRB 170817A**.

## Execução

```bash
python principal_multimensageiro.py
```
