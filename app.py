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

    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container{{max-width: 80%;}}

    </style>
    """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.selectbox(
        "Choisis une page", ["Hardy-Wienberg", "Modélisation"]
    )

    if page == "Hardy-Wienberg":
        """
        # L'équilibre d'Hardy-Wienberg

        Hardy Wienberg s'interessent aux allèles et aux génotypes d'une population au cours du temps.
        Ils nous dise que sans forces évolutives (mutation, sélection naturelle ...)
        la probabilité de chaque allèle, ou de chaque génotype, ne change pas au cours des générations.

        ## Démonstration

        On cherche à démontrer que :
         - la proportion de chaque allèle reste contante au cours des générations.
         - la proportion de chaque génotype reste constante au cours des générations.

        Pour cela on considère deux allèles `A` et `a` ce qui nous donne 3 génotypes `AA`, `Aa`, `aa`.

        On peut déduire le nombre d'individus total d'une population en additionant le nombre d'individus pour chaque génotype.
        Par exemple, si une population a 30 `AA`, 40 `Aa`, et 70 `aa`, on sait que la population totale est de $30 + 40 + 70 = 140$ individus

        ### Calcul de la proportion d'un génotype

        Pour calculer la proportion d'un génotype, il faut comparer le nombre d'individus de ce génotype ($N_{AA}$ par exemple pour le génotype `AA`) par rapport au nombre total d'individus dans la population ($N_{total}$).
        $$
        proportion\_AA = \\frac{N_{AA}}{N_{total}}
        $$
        Par exemple, si une population a 30 `AA`, 40 `Aa`, et 70 `aa`, la proportion de génotype `AA` est de $\\frac{30}{140} = 0.214$


        ### Calcul de la proportion d'un allèle

        Pour calculer la proportion de l'allèle A dans une population (que l'on appelle $p$),
        il suffit de compter le nombre d'allèle A par rapport à la quantité d'allèle totale.

        $$
        p = \\frac{nb\_allele\_A}{nb\_allele\_total}
        $$

        Les individus `AA` possédent deux fois l'allèle `A` et les individus `Aa` la possédent une fois.
        Du coup, le nombre d'allèle `A` est de : $2 . N_{AA} + 1 . N_{Aa}$.
        Le nombre d'allèle total est de $2 . N_{total}$ puisque chaque individus de la population contient 2 allèles.

        Du coup la fréquence allèlique de `A` est de :

        $$
        p = \\frac{2 . N_{AA} + 1 . N_{Aa}}{2 . N_{total}}
        $$

        Par exemple, si une population a 30 `AA`, 40 `Aa`, et 70 `aa`, la proportion de l'allèle `A` est de $\\frac{2 . 30 + 1 . 40}{2 . 140} = 0.357
        $

        Et ca veut dire qu'un individu de cette population a $35.7\%$ chance d'avoir l'allèle A.

        On peut procéder de la même manière pour calculer la proportion de l'allèle `a`, que l'on appelle `q` ou simplement faire $(1-A)$.

        ### Proportion des génotypes à la génération suivante

        Chaque nouvel individu est formé par la rencontre de deux gamètes.
        Chaque gamète possédent un seul allèle avec la probabilité $p\_A$ que cet allèle soit un `A`.
        Si deux gamètes se rencontrent, on peut faire l'arbre de probabilité suivant:
        """

        st.image("proba_tree.png", width=800)

        """
        On a donc $p^2$ chance d'obtenir un individu `AA`, $2pq$ chance d'avoir un individu `Aa`,
        et $q^2$ chance d'avoir un individu `aa`.
        Si on appelle $M$ le nombre d'individus de cette nouvelle génération,
        on aura $M . p^2$ individus `AA`, $M . 2 . p . q$ individus `Aa`,
        et $M . q^2$ individus `aa`.

        ### Proportion des allèles à la génération suivante

        On peut calculer, comme on l'a fait précedement la proportion d'allèle `A` dans cette nouvelle génération de M individus.

        $$
        p = \\frac{2 . N_{AA} + 1 . N_{Aa}}{2 . N_{total}} =
        \\frac{2 . M . p^2 + M . 2 . p . q}{2 . M}
        $$

        Qui une fois simplifié en "barrant" les $2M$ donne

        $$
        p^2 + p . q = p^2 + p. (1-p) = p^2 + p - p^2 = p
        $$

        La proportion d'allèle `A` dans la nouvelle génération est donc de $p$.


        ### Conclusion de la démonstration

        On a montré que, en l'absence de toute force évolutive,
        les proportion d'allèle `A` et `a` (respectivement $p$ et $q$)
        restaient les même d'une génération à l'autre.
        Nous avons aussi montré que la génération suivante a une proportion $p^2$ d'individus `AA`,
        $2.p.q$ d'individus `Aa`, et $q^2$ d'individus `aa`.
        Du coup de manière logique, toutes les générations auront les mêmes proportions de génotypes
        puisque la valeur de $p$ et $q$ ne change pas.

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
