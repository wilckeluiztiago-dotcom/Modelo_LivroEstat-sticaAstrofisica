"""
amostragem_avancada.py
----------------------
Amostragem de Monte Carlo avançada + geração de catálogo realista.

Métodos do livro:
- Amostragem por rejeição (Von Neumann) – Cap. 7
- Transformada inversa para φ
- Distribuição de metalicidade com gradiente + dispersão intrínseca
- Conversão para coordenadas heliocêntricas e galácticas

Autor: Luiz Tiago Wilcke
"""

import numpy as np
from modulos.densidade_espacial import (densidade_total, densidade_maxima,
                                         metalicidade_esperada, R_SOL, Z_SOL)


def amostrar_catalogo(n_estrelas=80000,
                      raio_max=18.0,
                      altura_max=2.8,
                      sigma_feh=0.075,
                      semente=2026):
    """
    Gera catálogo sintético calibrado em estatísticas reais de Gaia/APOGEE.
    
    Retorna dicionário com todas as quantidades relevantes.
    """
    rng = np.random.default_rng(semente)
    dens_max = densidade_maxima(raio_max, altura_max)

    x, y, z = [], [], []
    R_list, feh_list, comp = [], [], []

    aceitos = 0
    tentativas = 0
    max_tent = n_estrelas * 100

    while aceitos < n_estrelas and tentativas < max_tent:
        # Proposta em coordenadas cilíndricas
        R_prop = rng.uniform(0.0, raio_max)
        phi = rng.uniform(0.0, 2*np.pi)
        z_prop = rng.uniform(-altura_max, altura_max)

        dens = densidade_total(R_prop, z_prop)
        if rng.uniform(0, dens_max) < dens:
            xx = R_prop * np.cos(phi)
            yy = R_prop * np.sin(phi)

            # Metalicidade: valor esperado local + dispersão
            mu_feh = metalicidade_esperada(R_prop)
            feh = rng.normal(mu_feh, sigma_feh)

            # Classificação aproximada fino/espesso
            p_esp = 0.07 * np.exp(-np.abs(z_prop) / 0.9)
            c = 1 if rng.random() < p_esp else 0

            x.append(xx); y.append(yy); z.append(z_prop)
            R_list.append(R_prop); feh_list.append(feh); comp.append(c)
            aceitos += 1
        tentativas += 1

    cat = {
        "x": np.array(x),
        "y": np.array(y),
        "z": np.array(z),
        "R": np.array(R_list),
        "feh": np.array(feh_list),
        "componente": np.array(comp),
        "n": aceitos,
        "taxa_aceitacao": aceitos / max(tentativas, 1)
    }

    # Coordenadas heliocêntricas (Sol em x = -R_SOL)
    cat["x_hel"] = cat["x"] + R_SOL
    cat["y_hel"] = cat["y"]
    cat["z_hel"] = cat["z"] - Z_SOL
    cat["d_hel"] = np.sqrt(cat["x_hel"]**2 + cat["y_hel"]**2 + cat["z_hel"]**2)

    # Coordenadas galácticas aproximadas (l, b)
    cat["l"] = np.degrees(np.arctan2(cat["y_hel"], cat["x_hel"])) % 360
    cat["b"] = np.degrees(np.arcsin(cat["z_hel"] / cat["d_hel"]))

    return cat
