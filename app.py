import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import chi2
from scipy.stats import chisquare
import model


@st.cache
def run_model(
    nb_runs,
    nb_generations,
    pop_size,
    p_A,
    mutation_rate_A_to_a,
    mutation_rate_a_to_A,
    fitness_AA,
    fitness_Aa,
    fitness_aa,
):
    multiple_runs = np.zeros((nb_runs, 3, nb_generations))
    for i in range(nb_runs):
        multiple_runs[i] = model.run_simultation(
            nb_generations,
            pop_size,
            p_A,
            mutation_rate_A_to_a,
            mutation_rate_a_to_A,
            fitness_AA,
            fitness_Aa,
            fitness_aa,
        )
    return multiple_runs


def main():

    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container{{max-width: 80%;}}

    </style>
    """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.selectbox(
        "Choisis une page",
        ["Hardy-Wienberg", "Test Statistique", "Modélisation", "Exercice"],
    )

    if page == "Hardy-Wienberg":
        """
        # L'équilibre d'Hardy-Wienberg

        Hardy Wienberg s'interessent à l'évolution de la fréquence des allèles
        et aux génotypes d'une population au cours du temps.
        Ils nous disent dans une population d'effectif infini, sans forces évolutives
        (mutation, sélection naturelle ...) et avec une reproduction panmictique des individus
        qui se reproduisent aléatoirement,
        la probabilité de chaque allèle, ou de chaque génotype, ne change pas au cours des générations.

        ## Démonstration

        On cherche à démontrer que :
         - la proportion de chaque allèle reste constante au cours des générations.
         - la proportion de chaque génotype reste constante au cours des générations.

        Pour cela on considère deux allèles `A` et `a` ce qui nous donne 3 génotypes (`AA`), (`Aa`), (`aa`).

        On peut déduire le nombre d'individus total d'une population en additionnant le nombre d'individus pour chaque génotype.
        Par exemple, si une population a 30 (`AA`), 40 (`Aa`), et 70 (`aa`), on sait que la population totale est de $30 + 40 + 70 = 140$ individus

        ### Calcul de la proportion d'un génotype dans la population

        Pour calculer la proportion d'un génotype, il faut comparer le nombre d'individus de ce génotype ($N_{AA}$
        par exemple pour le génotype (`AA`)) par rapport au nombre total d'individus dans la population ($N_{total}$).
        $$
        proportion\_AA = \\frac{N_{AA}}{N_{total}}
        $$
        Par exemple, si une population a 30 (`AA`), 40 (`Aa`), et 70 (`aa`), la proportion de génotype (`AA`) est de $\\frac{30}{140} = 0.214$


        ### Calcul de la proportion d'un allèle dans la population

        Pour calculer la proportion de l'allèle A dans une population (que l'on appelle $p$),
        il suffit de compter le nombre d'allèle A par rapport à la quantité totale d'allèles.

        $$
        p = \\frac{nb\_allele\_A}{nb\_allele\_total}
        $$

        Les individus homozygotes (`AA`) possèdent deux fois l'allèle `A` et les individus hétérozygotes (`Aa`) ne possèdent qu'un allèle A.
        Du coup, le nombre d'allèle `A` est de : $2 . N_{AA} + 1 . N_{Aa}$.
        Le nombre d'allèle total est de $2 . N_{total}$ puisque chaque individus de la population contient 2 allèles.

        Du coup la fréquence allèlique de `A` est de :

        $$
        p = \\frac{2 . N_{AA} + 1 . N_{Aa}}{2 . N_{total}}
        $$

        Par exemple, si une population a 30 (`AA`), 40 (`Aa`), et 70 (`aa`), la proportion de l'allèle `A`
        est de $\\frac{2 . 30 + 1 . 40}{2 . 140} = 0.357
        $

        Et ca veut dire qu'un individu de cette population à $35.7\%$ chance d'avoir l'allèle A.

        On peut procéder de la même manière pour calculer la proportion de l'allèle `a`,
        que l'on appelle `q` ou simplement faire $q = (1-p)$.

        ### Proportion des génotypes à la génération suivante

        Dans une population panmictique, chaque nouvel individu est formé par la rencontre alêatoire de deux gamètes.
        Chaque gamète possède un seul allèle avec la probabilité $pA$ que cet allèle soit un `A`.
        Comme deux gamètes se rencontrent au hasard, on peut faire l'arbre de probabilité suivant:
        """

        st.image("proba_tree.png", width=800)

        """
        On a donc $p^2$ chance d'obtenir un individu (`AA`), $2pq$ chance d'avoir un individu (`Aa`),
        et $q^2$ chance d'avoir un individu (`aa`).
        Si on appelle $M$ le nombre d'individus de cette nouvelle génération,
        on aura $M . p^2$ individus (`AA`), $M . 2 . p . q$ individus (`Aa`),
        et $M . q^2$ individus (`aa`).

        ### Proportion des allèles à la génération suivante

        On peut calculer, comme on l'a fait précédement la proportion d'allèle `A` dans cette nouvelle génération de M individus.

        $$
        p = \\frac{2 . N_{AA} + 1 . N_{Aa}}{2 . N_{total}} =
        \\frac{2 . M . p^2 + M . 2 . p . q}{2 . M}
        $$

        Qui une fois simplifié en "barrant" les $2M$ donne

        $$
        p^2 + p . q = p^2 + p. (1-p) = p^2 + p - p^2 = p
        $$

        La proportion d'allèle `A` dans la nouvelle génération est donc de $p$, comme celle de la génération précédente.
        Comme $q=1-p$, la proportion de l'allèle `a` est égale à q comme celle de la génération précédente.


        ### Conclusion de la démonstration

        On a montré que, en l'absence de toute force évolutive,
        les proportions d'allèles `A` et `a` (respectivement $p$ et $q$)
        restaient les même d'une génération à l'autre.
        Nous avons aussi montré que la génération suivante a une proportion $p^2$ d'individus (`AA`),
        $2.p.q$ d'individus (`Aa`), et $q^2$ d'individus (`aa`).
        Du coup de manière logique, toutes les générations auront les mêmes proportions de génotypes
        puisque la valeur de $p$ et $q$ ne change pas.

        """

    elif page == "Test Statistique":

        """
        ## Qu'est ce qu'un échantillon ?

        Si on souhaite mesurer une caractéristique sur une grande population, on doit mesurer tous les individus de la population,
        mais il est souvent impossible de faire la mesure sur la population entière. Dans la pratique, on choisit un échantillon aléatoire de la
        population, c'est a dire plusieurs individus pris au hasard que l'on mesure pour avoir une approximation.
        Dans une population de 3 million d'individus avec deux allèles `A` et `a` pour un gène et ses trois génotypes associés (`AA`), (`Aa`), et (`aa`),
        il est plus facile de compter les génotypes de 100 individus pour avoir les fréquences génotypiques plutot que de compter les 3 millions d'individus.

        On distingue la **mesure sur l'échantillon** obtenue après la avoir compté les 100 individus, de la **mesure théorique**
        que l'on aurait obtenue si on avait compté toute la population.

        Un échantillon est une représentation imparfaite de la population. Il se peut que par hasard, il contient plus d'individus (`AA`),
        ou au contraire, plus d'individus (`aa`). Par conséquent, la mesure obtenues sur l'échantillon n'est quasiment jamais exactement égale à
        la mesure théorique.
        D'ailleurs un autre échantillon, contenant 100 individus différents, aurait des fréquence génotypique sensiblement différente de notre premier échantillon.

        De manière générale, plus un échantillon est grand, plus il y a de chance que la valeur estimée soit
        proche de la valeur théorique. A l'inverse, plus l'échantillon est petit, plus les valeurs estimées seront, en moyenne, éloignées de la valeur théorique.

        ## Echantillonage aléatoire d'une population connue

        Pour estimer la fréquence génotypique de la population, on prends plusieurs échantillons aléatoirement dans la population.
        On observe grace a un tirage aléatoire d'individus qui forment nos échantillons, que chaque échantillon possède des proportions de génotypes differentes.
        """

        x = st.slider(
            "Fréquence théorique des génotypes (AA), (Aa), (aa)",
            0.0,
            1.0,
            (0.25, 0.75),
            0.01,
        )
        population_ratio = np.array([x[0], x[1] - x[0], 1 - x[1]])
        sample_size = st.number_input(
            "Nombre d'individus dans l'échantillon", 0, 10000, 100, 10
        )
        nb_echantillons = st.number_input("Nombre d'échantillons", 0, 30, 10, 1)

        multiple_echantillon = np.zeros((nb_echantillons, 3))
        multiple_echantillon[:, 0] = np.random.binomial(
            sample_size, population_ratio[0], nb_echantillons
        )
        multiple_echantillon[:, 1] = np.random.binomial(
            sample_size, population_ratio[1], nb_echantillons
        )
        multiple_echantillon[:, 2] = sample_size - (
            multiple_echantillon[:, 0] + multiple_echantillon[:, 1]
        )
        fig = model.display_echantillons(
            multiple_echantillon, truth=population_ratio * sample_size
        )
        st.pyplot(fig)

        # Illustration dynamique sur les echantillons:

        """

        ## Déduire la fréquence des génotypes de la population totale

        Une fois que l'on a mesuré les fréquences des 3 génotypes, on peut demander quelle est la fréquence génotypique dans la population totale.
        La meilleure estimation que l'on puisse avoir est celle de l'échantillon. Par exemple, si on compte 30% de (`AA`) dans notre échantillon,
        on peux dire "la fréquence allélique des (`AA`) dans la population totale est d'environ 30%". On a une chance de me tromper bien sur,
        mais elle est moins importante que si on avait dit "la fréquence allélique des (`AA`) dans la population totale est d'environ 60%".

        On peut mesurer les chances de se tromper grace aux tests statistiques.

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
        notre échantillon soit originaire d'une population qui suit notre hypothèse. La différence entre les fréquence alléliques de notre hypothèse,
        vient certainement du hasard de l'échantillon.
        On en concluera qu'il y a de grande chance que notre échantillon ait été pris dans une population avec des fréquences alléliques de  33% (`AA`), 33% (`Aa`), et 34% (`aa`).

        Il existe une équation qui nous donne la probabilité que notre échantillon ait été pris dans une population en fonction des fréquences génotypiques hypothétiques de la population,
        et de celle mesurées dans notre échantillon. On appelle cette équation l'équation du Chi2.

        Pour chaque génotype $i$, on compare le nombre d'individus que l'on a observé ($Obs_i$) contre le nombre d'individus que l'on aurait du obtenir si la population suivait l'hypothèse $Theo_i$.

        $$
        Chi2 = \\sum{\\frac{(Obs_i - Theo_i)^2}{Theo_i} }
        $$

        Par exemple dans le cas de notre premier échantillon avec 60% de (`AA`), 20% de (`Aa`), et 20% de (`aa`), on a compté 60 (`AA`), 20 (`Aa`), et 20 (`aa`) dans notre échantillon.
        Si notre population suit l'hypothèse décrite dans notre example, on s'attends idéalement à avoir 33 (`AA`), 33 (`Aa`), et 34 (`aa`). Du coup notre Chi2 vaut:

        $$
        Chi2 = \\frac{(60-33)^2}{33} + \\frac{(20-33)^2}{33} + \\frac{(20-34)^2}{34} = 32.977
        $$

        On peut voir quelle est la probabilité d'obtenir cet échantillon sur la courbe du Chi2.

        """
        from scipy.stats import chi2

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(4, 2)
        x = np.linspace(0, 40, 1000)
        ax.plot(x, chi2.pdf(x, 2), "r-", lw=2, alpha=0.6, label="chi2")
        ax.vlines(32.977, 0, 0.5, linestyle="--")
        ax.set_xlabel("Valeur du Chi2")
        ax.set_ylabel("Probabilité")
        st.pyplot(fig,)

        """
        On peut voir que l'échantillon a une probabilité très faible de venir de la population de notre hypothèse.
        En d'autre terme, si on observe un tel échantillon, on est presque certain que la population
        n'est pas distribuée avec 33% de (`AA`), 33% de (`Aa`), et 34% de (`aa`).

        ## Exemple pratique avec l'équilibre de Hardy Wienberg

        Grace au test du Chi2, on peut maintenant determiner si un echantillon dont on vient de mesurer les fréquence génotypiques
        provient d'une population qui est à l'équilibre de Hardy Wienberg. Pour cela on fait l'hypothèse que notre population est à l'équilibre
        de Hardy Wienberg avec $p^2 %$ de (`AA`), $2pq %$ de (`Aa`), et $q^2 %$ de (`aa`), ou $p$ et $q$ sont les fréquence allèliques respectives de l'allèle `A` et `a`.

        Entrez les valeurs que vous observez dans votre échantillon pour chaque génotype:
        """

        col1, col2, col3 = st.beta_columns(3)
        with col1:
            AA = st.number_input(
                "nombre de (AA) dans l'échantillon", 0, 1000000, 0, 1
            )
        with col2:
            Aa = st.number_input(
                "nombre de (Aa) dans l'échantillon", 0, 1000000, 0, 1
            )
        with col3:
            aa = st.number_input(
                "nombre de (aa) dans l'échantillon", 0, 1000000, 0, 1
            )

        if AA == Aa == aa == 0:
            # nothing is inputted
            pass
        else:
            N = AA + Aa + aa
            fA = (AA + 0.5 * Aa) / N
            fa = 1 - fA

            fAA = fA ** 2
            fAa = 2 * fA * fa
            faa = fa ** 2

            chi2, pvalue = chisquare(
                [AA, Aa, aa], f_exp=[fAA * N, fAa * N, faa * N]
            )

            f"""
            On a donc un échantillon de taille {N}.

            La fréquence allélique de l'échantillon est de {np.around(fa, 2)} pour `A` et {np.around(fa, 2)} pour `a`.

            Par conséquent, si la population si la population est à l'équilibre de Hardy Wienberg, on s'attend à avoir:
             - {np.around(100*fAA, 2)} % de (`AA`)
             - {np.around(100*fAa, 2)} % de (`Aa`)
             - {np.around(100*faa, 2)} % de (`aa`)

            On observe:
            - {np.around(100*AA/N, 2)} % de (`AA`)
            - {np.around(100*Aa/N, 2)} % de (`Aa`)
            - {np.around(100*aa/N, 2)} % de (`aa`)

            Le test du Chi2 produit une valeur de {np.around(chi2, 4)} et il y a par conséquent {np.around(pvalue*100, 7)} % de chance que notre échantillon provienne d'une population à l'équilibre de Hardy Wienberg.
            """

            if pvalue < 0.05:
                """
                **Comme cette valeur est inférieure à 5%, on en conclue, avec moins de 5% de chance de se tromper que la population ne suit pas l'équilibre de Hardy Wienberg.**
                """
            else:
                """
                **Comme cette valeur est supérieure à 5%, on en conclue qu'il a une forte chance que la population suive l'équilibre de Hardy Wienberg**
                """

    elif page == "Modélisation":

        columns = st.multiselect(
            label="Quelles forces evolutives ?",
            options=["Dérive génétique", "Mutation", "Sélection Naturelle"],
        )

        nb_runs = st.sidebar.number_input(
            "Nombre de simulations", 1, 200, 10, 1
        )
        nb_generations = st.sidebar.number_input(
            "Nombre de générations", 1, 3000, 200, 10
        )
        p_A = st.sidebar.slider("Probabilité de l'allèle A", 0.0, 1.0, 0.5)
        st.sidebar.write(f"Probabilité de l'allèle A: {p_A:.2f}")
        st.sidebar.write(f"Probabilité de l'allèle a: {1-p_A:.2f}")

        if "Dérive génétique" in columns:
            """
            ## Dérive Génétique

            On modélise la dérive génétique en prenant en compte de petites populations.
            Plus la population est petite, plus le hasard a un impact fort sur la survie des allèles.

            Commencez par augmenter le `Nombre de générations` et diminuer la `Taille de la population` pour observer les effets de la dérive génétique.
            """
            pop_size = st.number_input(
                "Taille de la population", 0, 1000000, 1000, 1
            )
        else:
            pop_size = 1000000

        if "Mutation" in columns:
            """
            ## Mutation

            On modélise les mutations par un probabilité qu'une allèle se transforme en l'autre allèle.
            On peut choisir les probabilités de passer de `A` à `a` et inversement.
            Dans la vraie vie ces probabilités sont très faibles.

            Commencez par augmenter le `Taux de mutation a -> A` petit à petit pour voir les effects des mutations sur l'évolution de l'allèle A.
            """

            mutation_rate_a_to_A = st.number_input(
                "Taux de mutation a -> A", 0.0, 1.0, 0.0, 0.001, format="%.4f"
            )
            mutation_rate_A_to_a = st.number_input(
                "Taux de mutation A -> a", 0.0, 1.0, 0.0, 0.001, format="%.4f"
            )
        else:
            mutation_rate_a_to_A = 0.0
            mutation_rate_A_to_a = 0.0

        if "Sélection Naturelle" in columns:
            """
            ## Sélection Naturelle

            On modélise la sélection naturelle en tenant compte d'une chance de survie plus forte pour les individus d'un certain génotype.
            La valeur séléctive represente la chance de survie d'un génotype. Si tous les génotypes ont la même valeur sélective, il n'y a pas de pression évolutive.

            Commencez par augmenter petit a petit la `Valeur sélective AA` pour voir les effects de la sélection naturelle sur la fréquence de l'allèle A.
            Il est très interessant aussi d'observer ce qui se passe quand la valeur sélective des hétérozygotes est faible.
            """
            fitness_AA = st.slider("Valeur sélective (AA): ", 0.0, 2.0, 1.0)
            fitness_Aa = st.slider("Valeur sélective (Aa): ", 0.0, 2.0, 1.0)
            fitness_aa = st.slider("Valeur sélective (aa): ", 0.0, 2.0, 1.0)
        else:
            fitness_AA = 1.0
            fitness_Aa = 1.0
            fitness_aa = 1.0

        f"""
        ## Resultats

        Évolution des fréquences allèliques et des génotypes sur {nb_runs} populations.
        On observe les populations individuelles avec les lignes noires ainsi que les tendances moyennes
        representées par les lignes de couleurs.
        """

        st.markdown(
            """(en haut) Évolution de l'<span style="color:#630049">**allèle A**</span> et de l'<span style="color:#ffce00">**allèle a**</span>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """(en bas) Évolution des génotypes (<span style="color:#630049">**AA**</span>), (<span style="color:#d50032">**Aa**</span>), et (<span style="color:#ffce00">**aa**</span>)""",
            unsafe_allow_html=True,
        )

        multiple_runs = run_model(
            nb_runs,
            nb_generations,
            pop_size,
            p_A,
            mutation_rate_A_to_a,
            mutation_rate_a_to_A,
            fitness_AA,
            fitness_Aa,
            fitness_aa,
        )
        fig = model.display_multiple_runs(multiple_runs, pop_size)
        st.pyplot(fig)

        tests_chi2 = np.array(
            [
                chisquare(
                    ech,
                    f_exp=[
                        (p_A ** 2) * pop_size,
                        (2 * p_A * (1 - p_A)) * pop_size,
                        ((1 - p_A) ** 2) * pop_size,
                    ],
                )[1]
                for ech in multiple_runs[..., -1]
            ]
        )

        nb_pop_HW = np.sum(tests_chi2 >= 0.05)
        nb_pop_non_HW = np.sum(tests_chi2 < 0.05)

        f"""
        On compte parmis les {nb_runs} populations de la simulation:
         - {nb_pop_HW} populations qui suivent l'équilibre de Hardy Wienberg\*
         - {nb_pop_non_HW} populations qui ne suivent pas l'équilibre de Hardy Wienberg\*

        \*selon le critère du Chi2 avec moins de 5% de chance de se tromper

        """

    # elif page == "Exercice":
    #     """
    #     # Découvert du modèle de Hardy-Weinberg
    #
    #     ## Objectif
    #     Comparer deux populations panmictiques, d’effectif infini,
    #     sans mutations ni migration par rapport à l’évolution d’un gène qui code
    #     un caractère monogénique diallélique avec dominance de l’allèle A cercle
    #     bleu `Happy` sur l’allèle a cercle rouge `Grumpy`
    #
    #     ## Documents
    #
    #     On a un échantillon d'individus d'une **population 1** de taille infinie, panmictique,
    #     sans sélection, ni mutations, ni migration… avant la Covid.
    #     """
    #     st.image("exercice_pop1.png", width=400)
    #
    #     """
    #     On a aussi un échantillon d'individus d'une **population 2** de taille infinie, panmictique,
    #     sans mutations, ni migration… après la Covid.
    #     """
    #     st.image("exercice_pop2.png", width=400)
    #     """
    #     On fourni aussi les valeurs seuils du test du Chi2 associées à leurs probabilités d'erreurs.
    #     Ces valeurs sont obtenues pour un nombre de degrés de libertés de 2 car nous testons 3 génotypes
    #     `AA`, `Aa`, et `aa` toutes liés entre elles par :
    #
    #     $$
    #     N_{AA} + N_{Aa} + N_{aa} = N_{total}
    #     $$
    #
    #     | Probabilité | 0.9 | 0.5 | 0.3 | 0.2 | 0.1 | 0.05 | 0.02 | 0.01 | 0.001 |
    #     | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    #     | Valeur seuil Chi2 | 0.211 | 1.386 | 2.408 | 3.219 | 4.605 | **5.991** | 7.824 | 9.210 | 13.815 |
    #
    #     ## Questions
    #
    #     1. A partir de ces données, montrer que l’une de ces deux populations n’est pas à l’équilibre de Hardy-Weinberg en calculant:
    #        - la fréquence des deux allèles : $f(A) = p$ et $f(a) = q$
    #        - l’effectif théorique des deux populations si elles étaient panmictiques et à l’équilibre de Hardy-Weinberg
    #        - l’écart entre les effectifs observés O_i et calculés C_i en faisant le test de conformité à la panmixie du Chi2, pour une probabilité d’erreur de 0,05
    #
    #     $$
    #     Chi2 = \\sum{\\frac{(O_i - C_i)^2}{C_i}}
    #     $$
    #
    #     2. En tenant compte des données fournies dans l'énoncé, proposez une explication à vos résultats pour la population A et pour la population B.
    #     """


if __name__ == "__main__":
    main()
