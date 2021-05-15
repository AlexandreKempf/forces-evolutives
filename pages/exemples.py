import streamlit as st


def generate_page():
    st.markdown(
        """
    On vous prépare des exemples ! Patientez encore un peu ...
    """
    )
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
    #     st.image("images/exercice_pop1.png", width=400)
    #
    #     """
    #     On a aussi un échantillon d'individus d'une **population 2** de taille infinie, panmictique,
    #     sans mutations, ni migration… après la Covid.
    #     """
    #     st.image("images/exercice_pop2.png", width=400)
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
