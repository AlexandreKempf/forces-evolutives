import streamlit as st
import src.plots


def generate_page():
    st.markdown("""
    # L'équilibre d'Hardy-Weinberg
    
    Hardy Weinberg s'interessent à l'évolution de la fréquence des allèles
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
    proportion\\_AA = \\frac{N_{AA}}{N_{total}}
    $$
    Par exemple, si une population a 30 (`AA`), 40 (`Aa`), et 70 (`aa`), la proportion de génotype (`AA`) est de $\\frac{30}{140} = 0.214$


    ### Calcul de la proportion d'un allèle dans la population

    Pour calculer la proportion de l'allèle A dans une population (que l'on appelle $p$),
    il suffit de compter le nombre d'allèle A par rapport à la quantité totale d'allèles.

    $$
    p = \\frac{nb\\_allele\\_A}{nb\\_allele\\_total}
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

    Et ca veut dire qu'un individu de cette population à $35.7\\%$ chance d'avoir l'allèle A.

    On peut procéder de la même manière pour calculer la proportion de l'allèle `a`,
    que l'on appelle `q` ou simplement faire $q = (1-p)$.

    ### Proportion des génotypes à la génération suivante

    Dans une population panmictique, chaque nouvel individu est formé par la rencontre alêatoire de deux gamètes.
    Chaque gamète possède un seul allèle avec la probabilité $pA$ que cet allèle soit un `A`.
    Comme deux gamètes se rencontrent au hasard, on peut faire l'arbre de probabilité suivant:
    """)
    st.image("images/proba_tree.png", width=800)

    st.markdown("""
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

    ## Distribution des génotypes en fonction de la fréquence allélique

    On peut représenter, en pourcentage d'individus, les fréquences de chaque génotype en fonction de la fréquence
    de l'allèle A.
    """)

    fig = src.plots.plot_genotype_vs_frequence_allelique()
    st.pyplot(fig)
