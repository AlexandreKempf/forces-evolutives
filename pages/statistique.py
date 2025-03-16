import streamlit as st
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import chi2
from scipy.stats import chisquare
import src.plots


def generate_page():
    st.markdown(
        """
    ## Qu'est-ce qu'un échantillon ?

    Si on souhaite mesurer une caractéristique sur une grande population, on doit mesurer tous les individus de la population,
    mais il est souvent impossible de faire la mesure sur la population entière. Dans la pratique, on choisit un échantillon aléatoire de la
    population, c'est a dire plusieurs individus pris au hasard que l'on mesure pour avoir une approximation.
    Dans une population de 3 million d'individus avec deux allèles `A` et `a` pour un gène et ses trois génotypes associés (`AA`), (`Aa`), et (`aa`),
    il est plus facile de compter les génotypes de 100 individus pour avoir les fréquences génotypiques plutôt que de compter les 3 millions d'individus.

    On distingue la **mesure sur l'échantillon** obtenue après la avoir compté les 100 individus, de la **mesure théorique**
    que l'on aurait obtenue si on avait compté toute la population.

    Un échantillon est une représentation imparfaite de la population. Il se peut que par hasard, il contienne plus d'individus (`AA`),
    ou au contraire, plus d'individus (`aa`). Par conséquent, la mesure obtenue sur l'échantillon n'est quasiment jamais exactement égale à
    la mesure théorique.
    D'ailleurs un autre échantillon, contenant 100 individus différents, aurait des fréquences génotypiques sensiblement différentes de notre premier échantillon.

    De manière générale, plus un échantillon est grand, plus il y a de chance que la valeur estimée soit
    proche de la valeur théorique. A l'inverse, plus l'échantillon est petit, plus les valeurs estimées seront, en moyenne, éloignées de la valeur théorique.

    ## Échantillonnage aléatoire d'une population connue

    Pour estimer la fréquence génotypique de la population, on prend plusieurs échantillons aléatoirement dans la population.
    On observe grâce à un tirage aléatoire d'individus qui forment nos échantillons, que chaque échantillon possède des proportions de génotypes différentes.
    """
    )

    x = st.slider(
        "Fréquence théorique des génotypes (AA), (Aa), (aa)",
        0.0,
        1.0,
        (0.25, 0.75),
        0.01,
    )
    population_ratio = np.array([x[0], x[1] - x[0], 1 - x[1]])
    sample_size = st.number_input("Nombre d'individus dans l'échantillon", 0, 1000000, 100, 10)
    nb_echantillons = st.number_input("Nombre d'échantillons", 0, 30, 10, 1)

    multiple_echantillon = np.zeros((nb_echantillons, 3))
    multiple_echantillon[:, 0] = np.random.binomial(sample_size, population_ratio[0], nb_echantillons)
    multiple_echantillon[:, 1] = np.random.binomial(sample_size, population_ratio[1], nb_echantillons)
    multiple_echantillon[:, 2] = sample_size - (multiple_echantillon[:, 0] + multiple_echantillon[:, 1])
    fig = src.plots.display_echantillons(multiple_echantillon, truth=population_ratio * sample_size)
    st.pyplot(fig)

    st.markdown(
        """

    ## Déduire la fréquence des génotypes de la population totale

    Une fois que l'on a mesuré les fréquences des 3 génotypes, on peut demander quelle est la fréquence génotypique dans la population totale.
    La meilleure estimation que l'on puisse avoir est celle de l'échantillon. Par exemple, si on compte 30% de (`AA`) dans notre échantillon,
    on peut dire "la fréquence allélique des (`AA`) dans la population totale est d'environ 30%". On a une chance de me tromper bien sûr,
    mais elle est moins importante que si on avait dit "la fréquence allélique des (`AA`) dans la population totale est d'environ 60%".

    On peut mesurer les chances de se tromper grâce aux tests statistiques.

    Pour cela il faut prendre le problème à l'envers. On va faire une hypothèse sur la population (par exemple, l'hypothèse
    qu'il y a 33% de (`AA`), 33% de (`Aa`), et 34% de (`aa`))
    et on va mesurer la chance d'obtenir aléatoirement l'échantillon que l'on vient de mesurer si cette hypothèse est vraie.

    Ainsi dans cette exemple, si notre échantillon de 100 individus contient 60% de (`AA`), 20% de (`Aa`), et 20% de (`aa`), il y a moins de 1% de chance que
    notre échantillon soit originaire d'une population qui suit notre hypothèse.
    On peut donc dire:
     - L'échantillon provient d'une population avec 33% de (`AA`), 33% de (`Aa`), et 34% de (`aa`) mais on a moins de 0.1% de chance d'avoir raison
     - L'échantillon ne provient pas d'une population avec 33% de (`AA`), 33% de (`Aa`), et 34% de (`aa`) et on a plus de 99.9% de chance d'avoir raison

    Du coup on va conclure que la population a de grande chance de ne pas avoir 33% de (`AA`), 33% de (`Aa`), et 34% de (`aa`).

    En revanche, si notre échantillon de 100 individus contient 32% de (`AA`), 35% de (`Aa`), et 33% de (`aa`), il y a de grande chance que
    notre échantillon soit originaire d'une population qui suit notre hypothèse. La différence entre les fréquences alléliques de notre hypothèse,
    vient certainement du hasard de l'échantillon.
    On en conclura qu'il y a de grande chance que notre échantillon ait été pris dans une population avec des fréquences alléliques de  33% (`AA`), 33% (`Aa`), et 34% (`aa`).

    Il existe une équation qui nous donne la probabilité que notre échantillon ait été pris dans une population en fonction des fréquences génotypiques hypothétiques de la population,
    et de celle mesurées dans notre échantillon. On appelle cette équation l'équation du Chi2.

    Pour chaque génotype $i$, on compare le nombre d'individus que l'on a observé ($Obs_i$) contre le nombre d'individus que l'on aurait dû obtenir si la population suivait l'hypothèse $Theo_i$.

    $$
    Chi2 = \\sum{\\frac{(Obs_i - Theo_i)^2}{Theo_i} }
    $$

    Par exemple dans le cas de notre premier échantillon avec 60% de (`AA`), 20% de (`Aa`), et 20% de (`aa`), on a compté 60 (`AA`), 20 (`Aa`), et 20 (`aa`) dans notre échantillon.
    Si notre population suit l'hypothèse décrite dans notre exemple, on s'attend idéalement à avoir 33 (`AA`), 33 (`Aa`), et 34 (`aa`). Du coup notre Chi2 vaut:

    $$
    Chi2 = \\frac{(60-33)^2}{33} + \\frac{(20-33)^2}{33} + \\frac{(20-34)^2}{34} = 32.977
    $$

    On peut voir quelle est la probabilité d'obtenir cet échantillon sur la courbe du Chi2.

    """
    )

    fig = src.plots.chi2_curve()
    st.pyplot(fig)

    st.markdown(
        """
    On peut voir que l'échantillon a une probabilité très faible de venir de la population de notre hypothèse.
    En d'autre terme, si on observe un tel échantillon, on est presque certain que la population
    n'est pas distribuée avec 33% de (`AA`), 33% de (`Aa`), et 34% de (`aa`).

    ## Exemple pratique avec l'équilibre de Hardy Weinberg

    Grace au test du Chi2, on peut maintenant déterminer si un échantillon dont on vient de mesurer les fréquences génotypiques
    provient d'une population qui est à l'équilibre de Hardy Weinberg. Pour cela on fait l'hypothèse que notre population est à l'équilibre
    de Hardy Weinberg avec $p^2 %$ de (`AA`), $2pq %$ de (`Aa`), et $q^2 %$ de (`aa`), ou $p$ et $q$ sont les fréquences allèliques respectives de l'allèle `A` et `a`.

    Entrez les valeurs que vous observez dans votre échantillon pour chaque génotype:
    """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        AA = st.number_input("nombre de (AA) dans l'échantillon", 0, 1000000, 0, 1)
    with col2:
        Aa = st.number_input("nombre de (Aa) dans l'échantillon", 0, 1000000, 0, 1)
    with col3:
        aa = st.number_input("nombre de (aa) dans l'échantillon", 0, 1000000, 0, 1)

    if AA == Aa == aa == 0:
        # nothing is inputted
        pass
    else:
        N = AA + Aa + aa
        fA = (AA + 0.5 * Aa) / N
        fa = 1 - fA

        fAA = fA**2
        fAa = 2 * fA * fa
        faa = fa**2

        chi2_value, pvalue = chisquare([AA, Aa, aa], f_exp=[fAA * N, fAa * N, faa * N])

        st.markdown(
            f"""
        On a donc un échantillon de taille {N}.

        La fréquence allélique de l'échantillon est de {np.around(fA, 2)} pour `A` et {np.around(fa, 2)} pour `a`.

        Par conséquent, si la population est à l'équilibre de Hardy Weinberg, on s'attend à avoir:
         - {np.around(100*fAA, 2)} % de (`AA`)
         - {np.around(100*fAa, 2)} % de (`Aa`)
         - {np.around(100*faa, 2)} % de (`aa`)

        On observe:
        - {np.around(100*AA/N, 2)} % de (`AA`)
        - {np.around(100*Aa/N, 2)} % de (`Aa`)
        - {np.around(100*aa/N, 2)} % de (`aa`)

        Le test du Chi2 produit une valeur de {np.around(chi2_value, 4)} et il y a, par conséquent, {np.around(pvalue*100, 7)} % de chance que notre échantillon provienne d'une population à l'équilibre de Hardy Weinberg.
        """
        )

        if pvalue < 0.05:
            st.markdown(
                """
            **Comme cette valeur est inférieure à 5%, on en conclut, avec moins de 5% de chance de se tromper que la population ne suit pas l'équilibre de Hardy Weinberg.**
            """
            )
        else:
            st.markdown(
                """
            **Comme cette valeur est supérieure à 5%, on en conclut qu'il a une forte chance que la population suive l'équilibre de Hardy Weinberg**
            """
            )
