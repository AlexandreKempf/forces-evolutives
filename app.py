import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import model


@st.cache
def run_model(
    nb_runs,
    nb_generations,
    pop_size,
    pA,
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
            pA,
            mutation_rate_A_to_a,
            mutation_rate_a_to_A,
            fitness_AA,
            fitness_Aa,
            fitness_aa,
        )
    return multiple_runs


def main():

    page = st.sidebar.selectbox(
        "Choose a page", ["Hardy-Wienberg", "Dérive Génétique", "Modélisation"]
    )

    if page == "Hardy-Wienberg":
        """
        # L'équilibre d'Hardy-Wienberg

        Yo papa 😄
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
        pA = st.sidebar.slider("Probabilité de l'allèle A", 0.0, 1.0, 0.5)
        st.sidebar.write(f"Probabilité de l'allèle A: {pA:.2f}")
        st.sidebar.write(f"Probabilité de l'allèle a: {1-pA:.2f}")

        if "Dérive génétique" in columns:
            """
            ## Dérive Génétique

            On modélise la dérive génétique en prenant en compte de petites populations.
            Plus la population est petite, plus le hasard a un impact fort sur la survie des allèles.

            Commencez par augmenter le `Nombre de générations` et diminuer la `Taille de la population` pour observer les effets de la dérive génetique.
            """
            pop_size = st.number_input(
                "Taille de la population", 0, 10000, 10000, 1
            )
        else:
            pop_size = 10000

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
            fitness_AA = st.slider("Valeur sélective AA: ", 0.0, 2.0, 1.0)
            fitness_Aa = st.slider("Valeur sélective Aa: ", 0.0, 2.0, 1.0)
            fitness_aa = st.slider("Valeur sélective aa: ", 0.0, 2.0, 1.0)
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
            """(en bas) Évolution des génotypes <span style="color:#630049">**AA**</span>, <span style="color:#d50032">**Aa**</span>, et <span style="color:#ffce00">**aa**</span>""",
            unsafe_allow_html=True,
        )

        multiple_runs = run_model(
            nb_runs,
            nb_generations,
            pop_size,
            pA,
            mutation_rate_A_to_a,
            mutation_rate_a_to_A,
            fitness_AA,
            fitness_Aa,
            fitness_aa,
        )
        fig = model.display_multiple_runs(multiple_runs, pop_size)
        st.pyplot(fig)

    # elif page == 'Exploration':
    # st.title('Explore the Wine Data-set')
    # if st.checkbox('Show column descriptions'):
    #     st.dataframe(df.describe())
    #
    # st.markdown('### Analysing column relations')
    # st.text('Correlations:')
    # fig, ax = plt.subplots(figsize=(10,10))
    # sns.heatmap(df.corr(), annot=True, ax=ax)
    # st.pyplot(fig)
    # st.text('Effect of the different classes')
    # fig = sns.pairplot(df, vars=['magnesium', 'flavanoids', 'nonflavanoid_phenols', 'proline'], hue='alcohol')
    # st.pyplot(fig)
    # else:
    # st.title('Modelling')
    # model, accuracy = train_model(df)
    # st.write('Accuracy: ' + str(accuracy))
    # st.markdown('### Make prediction')
    # st.dataframe(df)
    # row_number = st.number_input('Select row', min_value=0, max_value=len(df)-1, value=0)
    # st.markdown('#### Predicted')
    # st.text(model.predict(df.drop(['alcohol'], axis=1).loc[row_number].values.reshape(1, -1))[0])


if __name__ == "__main__":
    main()
