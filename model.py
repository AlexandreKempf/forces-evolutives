import matplotlib.pyplot as plt
import numpy as np
import seaborn


def run_simultation(
    nb_gen,
    N,
    pA,
    mutation_rate_A_to_a,
    mutation_rate_a_to_A,
    fitness_AA,
    fitness_Aa,
    fitness_aa,
):
    max_fitness = np.max([fitness_AA, fitness_Aa, fitness_aa])
    fitness_AA /= max_fitness
    fitness_Aa /= max_fitness
    fitness_aa /= max_fitness
    pa = 1 - pA
    genepool = np.random.choice(["A", "a"], N * 2, p=[pA, pa])
    genotypes = np.zeros((3, nb_gen))  # AA, Aa, aa
    for t in range(nb_gen):
        mum = np.random.choice(genepool, N)
        dad = np.random.choice(genepool, N)

        AA = (mum == "A") & (dad == "A")
        Aa = ((mum == "A") & (dad == "a")) | ((mum == "a") & (dad == "A"))
        aa = (mum == "a") & (dad == "a")

        genotypes[0, t] = np.sum(AA)
        genotypes[1, t] = np.sum(Aa)
        genotypes[2, t] = np.sum(aa)

        idx_vivants_AA = np.random.choice(
            np.where(AA)[0], int(np.sum(AA) * fitness_AA), replace=False
        )
        idx_vivants_Aa = np.random.choice(
            np.where(Aa)[0], int(np.sum(Aa) * fitness_Aa), replace=False
        )
        idx_vivants_aa = np.random.choice(
            np.where(aa)[0], int(np.sum(aa) * fitness_aa), replace=False
        )
        idx_vivants = np.sort(
            np.concatenate([idx_vivants_AA, idx_vivants_Aa, idx_vivants_aa])
        )

        genepool = np.concatenate([mum[idx_vivants], dad[idx_vivants]])
        A = genepool == "A"
        mutants_A_to_a = A & (np.random.random(len(A)) < mutation_rate_A_to_a)
        a = genepool == "a"
        mutants_a_to_A = a & (np.random.random(len(a)) < mutation_rate_a_to_A)
        genepool[mutants_A_to_a] = "a"
        genepool[mutants_a_to_A] = "A"
    return genotypes


def display_multiple_runs(multiple_runs, N):
    colors_map = {
        "aa": "#ffce00",
        "Aa": "#d50032",
        "AA": "#630049",
        "line": "#aa0000",
    }

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
