import streamlit as st
import numpy as np
from scipy.stats import chisquare

import src.model
import src.plots


@st.cache_data
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
    consanguinite,
):
    return np.array(
        [
            src.model.run_simulation(
                nb_generations,
                pop_size,
                p_A,
                mutation_rate_A_to_a,
                mutation_rate_a_to_A,
                fitness_AA,
                fitness_Aa,
                fitness_aa,
                consanguinite,
            )
            for _ in range(nb_runs)
        ]
    )


def generate_page():
    columns = st.multiselect(
        label="Quelles forces evolutives ?",
        options=[
            "Dérive génétique",
            "Mutation",
            "Sélection Naturelle",
            "Consanguinité",
        ],
    )

    nb_runs = st.sidebar.number_input("Nombre de simulations", 1, 200, 10, 1)
    nb_generations = st.sidebar.number_input("Nombre de générations", 1, 3000, 200, 10)
    p_A = st.sidebar.slider("Probabilité de l'allèle A", 0.0, 1.0, 0.5)
    st.sidebar.write(f"Probabilité de l'allèle A: {p_A:.2f}")
    st.sidebar.write(f"Probabilité de l'allèle a: {1-p_A:.2f}")

    if "Dérive génétique" in columns:
        st.markdown(
            """
        ## Dérive Génétique

        On modélise la dérive génétique en prenant en compte de petites populations.
        Plus la population est petite, plus le hasard a un impact fort sur la survie des allèles.

        Commencez par augmenter le `Nombre de générations` et diminuer la `Taille de la population` pour observer les effets de la dérive génétique.
        """
        )
        pop_size = st.number_input("Taille de la population", 0, int(1e9), 1000, 1)
    else:
        pop_size = int(1e9)

    if "Mutation" in columns:
        st.markdown(
            """
        ## Mutation

        On modélise les mutations par un probabilité qu'une allèle se transforme en l'autre allèle.
        On peut choisir les probabilités de passer de `A` à `a` et inversement.
        Dans la vraie vie ces probabilités sont très faibles.

        Commencez par augmenter le `Taux de mutation a -> A` petit à petit pour voir les effects des mutations sur l'évolution de l'allèle A.
        """
        )

        mutation_rate_a_to_A = st.number_input("Taux de mutation a -> A", 0.0, 1.0, 0.0, 0.001, format="%.4f")
        mutation_rate_A_to_a = st.number_input("Taux de mutation A -> a", 0.0, 1.0, 0.0, 0.001, format="%.4f")
    else:
        mutation_rate_a_to_A = 0.0
        mutation_rate_A_to_a = 0.0

    if "Sélection Naturelle" in columns:
        st.markdown(
            """
        ## Sélection Naturelle

        On modélise la sélection naturelle en tenant compte d'une chance de survie plus forte pour les individus d'un certain génotype.
        La valeur séléctive represente la chance de survie d'un génotype. Si tous les génotypes ont la même valeur sélective, il n'y a pas de pression évolutive.

        Commencez par augmenter petit a petit la `Valeur sélective AA` pour voir les effects de la sélection naturelle sur la fréquence de l'allèle A.
        Il est très interessant aussi d'observer ce qui se passe quand la valeur sélective des hétérozygotes est faible.
        """
        )
        fitness_AA = st.slider("Valeur sélective (AA): ", 0.0, 2.0, 1.0)
        fitness_Aa = st.slider("Valeur sélective (Aa): ", 0.0, 2.0, 1.0)
        fitness_aa = st.slider("Valeur sélective (aa): ", 0.0, 2.0, 1.0)
    else:
        fitness_AA = 1.0
        fitness_Aa = 1.0
        fitness_aa = 1.0

    if "Consanguinité" in columns:
        st.markdown(
            """
        ## Consanguinité

        On modélise la consanguinité en modifiant les probabilités de rencontre entre individus avant la reproduction.
        Une valeur de consanguinité, nommée F, définie la similarité des génotypes de deux individus.
        Une valuer F de 1 signifie qu'ils ont tous les gènes en commun, comme les vrais jumeaux,
        alors qu'un F de 0.5 indique qu'il partage que la moitié des gènes, comme un parent avec son enfant.

        On peut voir que la consanguinité ne modifie pas la fréquence allèlique, mais qu'un fort
        taux de consanguinité favorise les homozygotes.
        """
        )
        consanguinite = st.slider("F de consanguinité: ", 0.0, 1.0, 0.0)
    else:
        consanguinite = 0.0
    st.markdown(
        f"""
    ## Résultats

    Évolution des fréquences allèliques et des génotypes sur {nb_runs} populations.
    On observe les populations individuelles avec les lignes noires ainsi que les tendances moyennes
    representées par les lignes de couleurs.
    """
    )

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
        consanguinite,
    )
    fig = src.plots.display_multiple_runs(multiple_runs, pop_size)
    st.pyplot(fig)

    taille_echantillon = 100
    tests_chi2 = np.array(
        [
            chisquare(
                (ech / pop_size) * taille_echantillon,
                f_exp=[
                    (p_A**2) * taille_echantillon,
                    (2 * p_A * (1 - p_A)) * taille_echantillon,
                    ((1 - p_A) ** 2) * taille_echantillon,
                ],
            )[1]
            for ech in multiple_runs[..., -1]
        ]
    )

    nb_pop_HW = np.sum(tests_chi2 >= 0.05)
    nb_pop_non_HW = np.sum(tests_chi2 < 0.05)

    st.markdown(
        f"""
    On compte parmis les {nb_runs} populations de la simulation:
     - {nb_pop_HW} populations qui suivent l'équilibre de Hardy Weinberg\\*
     - {nb_pop_non_HW} populations qui ne suivent pas l'équilibre de Hardy Weinberg\\*

    \\*selon le critère du Chi2 avec moins de 5% de chance de se tromper

    """
    )
