import matplotlib.pyplot as plt
import numpy as np
import seaborn


colors_map = {
    "aa": "#ffce00",
    "other2": "#ff3803",
    "Aa": "#d50032",
    "other1": "#9e003e",
    "AA": "#630049",
}


def run_simultation(
    nb_gen,
    N,
    p_A,
    mutation_rate_A_to_a,
    mutation_rate_a_to_A,
    fitness_AA,
    fitness_Aa,
    fitness_aa,
    consanguinite,
):
    max_fitness = np.max([fitness_AA, fitness_Aa, fitness_aa])
    fitness_AA /= max_fitness
    fitness_Aa /= max_fitness
    fitness_aa /= max_fitness
    p_a = 1 - p_A
    genotypes = np.zeros((3, nb_gen))  # AA, Aa, aa
    N_AA = int(N * p_A ** 2)
    N_aa = int(N * p_a ** 2)
    N_Aa = N - (N_AA + N_aa)
    F = consanguinite

    for t in range(nb_gen):
        p_AA = N_AA / N
        p_Aa = N_Aa / N
        p_aa = N_aa / N

        # new generation
        p_AA = (
            (p_AA * (p_AA * (1 - F) + F))
            + (0.25 * p_Aa * (p_Aa * (1 - F) + F))
            + (p_AA * p_Aa * (1 - F))
        )

        p_aa = (
            (p_aa * (p_aa * (1 - F) + F))
            + (0.25 * p_Aa * (p_Aa * (1 - F) + F))
            + (p_aa * p_Aa * (1 - F))
        )

        N_AA = np.random.binomial(N, p_AA)
        N_aa = np.random.binomial(N, p_aa)
        N_Aa = N - N_AA - N_aa

        # selection
        if not fitness_AA == fitness_Aa == fitness_aa:
            N_AA = np.random.binomial(N_AA, fitness_AA)
            N_Aa = np.random.binomial(N_Aa, fitness_Aa)
            N_aa = np.random.binomial(N_aa, fitness_aa)

        # mutation
        if mutation_rate_A_to_a != 0 or mutation_rate_a_to_A != 0:
            mut_AA_to_Aa = np.random.binomial(N_AA, mutation_rate_A_to_a)
            mut_Aa_to_aa = np.random.binomial(N_Aa, mutation_rate_A_to_a / 2)
            mut_Aa_to_AA = np.random.binomial(N_Aa, mutation_rate_a_to_A / 2)
            mut_aa_to_Aa = np.random.binomial(N_aa, mutation_rate_a_to_A)
            N_AA += mut_Aa_to_AA - mut_AA_to_Aa
            N_Aa += mut_AA_to_Aa - mut_Aa_to_AA - mut_Aa_to_aa + mut_aa_to_Aa
            N_aa += mut_Aa_to_aa - mut_aa_to_Aa

        genotypes[0, t] = N_AA
        genotypes[1, t] = N_Aa
        genotypes[2, t] = N_aa
    return genotypes


def display_multiple_runs(multiple_runs, N):
    def draw_proba_allele(data, ax):
        ax.fill_between(
            np.arange(0, data.shape[1]),
            np.mean(data, 0),
            np.zeros(data.shape[1]),
            color=colors_map["AA"],
        )
        ax.fill_between(
            np.arange(0, data.shape[1]),
            np.ones(data.shape[1]),
            np.mean(data, 0),
            color=colors_map["aa"],
        )
        ax.plot(data.T, lw=1, color="black", alpha=0.3)
        # ax.plot(np.mean(data, 0), lw=3, color=colors_map["line"])
        return ax

    def draw_proba_genotypes(data, ax, N):
        mean_AA, mean_Aa, mean_aa = np.mean(data, 0)
        ax.fill_between(
            np.arange(0, data.shape[-1]),
            mean_AA,
            np.zeros(data.shape[-1]),
            color=colors_map["AA"],
        )
        ax.fill_between(
            np.arange(0, data.shape[-1]),
            mean_AA + mean_Aa,
            mean_AA,
            color=colors_map["Aa"],
        )
        ax.fill_between(
            np.arange(0, data.shape[-1]),
            np.ones(data.shape[-1]) * N,
            mean_AA + mean_Aa,
            color=colors_map["aa"],
        )
        ax.plot(data[:, 0, :].T, lw=1, color="black", alpha=0.3)
        ax.plot(
            data[:, 0, :].T + data[:, 1, :].T, lw=1, color="black", alpha=0.3
        )
        # ax.plot(mean_AA, lw=3, color=colors_map["line"])
        # ax.plot(mean_AA + mean_Aa, lw=3, color=colors_map["line"])
        # ax.plot(np.mean(data, 0), lw=3, color="#2f528e")
        # ax.plot(mean_AA, lw=3, color=colors_map["line"])
        return ax

    fig, axes = plt.subplots(2, 1)
    axes[0] = draw_proba_allele(
        (multiple_runs[:, 0, :] * 2 + multiple_runs[:, 1, :]) / (2 * N), axes[0]
    )
    axes[1] = draw_proba_genotypes(multiple_runs, axes[1], N)

    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0, N)

    axes[0].set_xlim(0, multiple_runs.shape[-1] - 1)
    axes[1].set_xlim(0, multiple_runs.shape[-1] - 1)

    axes[0].set_ylabel("% allèle")
    axes[1].set_ylabel("nombre d'ind.")

    axes[0].set_xticks([])
    axes[1].set_xlabel("nombre de générations")
    return fig


def display_echantillons(echantillons, truth):
    echantillons = np.concatenate(
        [[truth], [[0, 0, 0]], [[0, 0, 0]], echantillons]
    )
    echantillons_cum = echantillons.cumsum(axis=1)
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.set_xlabel("Nombre d'individus de chaque génotype")
    ax.set_xlim(0, np.sum(echantillons, axis=1).max())
    ax.tick_params(
        axis="y",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,
    )
    genotypes = ["AA", "Aa", "aa"]
    labels = ["Population totale", "", " ", "Échantillons"]
    labels += [" " * (e + 2) for e in range(len(echantillons) - len(labels))]
    for i in range(3):
        widths = echantillons[:, i]
        starts = echantillons_cum[:, i] - widths
        color = colors_map[genotypes[i]]
        rects = ax.barh(
            labels,
            widths,
            left=starts,
            height=0.5,
            label=f"({genotypes[i]})",
            color=color,
        )
    ax.legend(ncol=3, bbox_to_anchor=(0, 1), loc="lower left", fontsize="small")
    return fig
