import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import model


def main():

    page = st.sidebar.selectbox(
        "Choose a page", ["Hardy-Wienberg", "Dérive Génétique", "Modélisation"]
    )

    if page == "Hardy-Wienberg":
        """
        # L'équilibre d'Hardy-Wienberg

        Bla bla in the markdown format...
        """

    elif page == "Dérive Génétique":
        """
        # La dérive génétique

        Bla bla in the markdown format...
        """

    elif page == "Modélisation":

        columns = st.multiselect(
            label="Quelles forces evolutives ?", options=["Mutation"],
        )

        nb_runs = st.sidebar.number_input(
            "Nombre de simulations", 1, 200, 10, 1
        )
        pop_size = st.sidebar.number_input(
            "Taille de la population", 0, 10000, 1000, 1
        )
        nb_generations = st.sidebar.number_input(
            "Nombre de générations", 1, 3000, 200, 10
        )
        pA = st.sidebar.slider("Probabilité de l'allèle A", 0.0, 1.0, 0.5)
        st.sidebar.write(f"Probabilité de l'allèle A: {pA:.2f}")
        st.sidebar.write(f"Probabilité de l'allèle a: {1-pA:.2f}")

        if "Mutation" in columns:
            """
            ## Mutation
            """
            mutation_rate_A_to_a = st.number_input(
                "Taux de mutation A -> a", 0.0, 1.0, 0.0, 0.001, format="%.4f"
            )
            mutation_rate_a_to_A = st.number_input(
                "Taux de mutation a-> A", 0.0, 1.0, 0.001, 0.001, format="%.4f"
            )
        else:
            mutation_rate_A_to_a = 0.0
            mutation_rate_a_to_A = 0.0
        """
        ## Resultats
        """
        multiple_runs = np.zeros((nb_runs, 3, nb_generations))
        for i in range(nb_runs):
            multiple_runs[i] = model.run_simultation(
                nb_generations,
                pop_size,
                pA,
                mutation_rate_A_to_a,
                mutation_rate_a_to_A,
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
