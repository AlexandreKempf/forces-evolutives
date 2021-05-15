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
        axis="y", which="both", left=False, right=False,
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


def plot_genotype_vs_frequence_allelique():
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(4, 5)
    x = np.linspace(0, 1, 1000)
    ax[0].plot(x, x ** 2, lw=2, label="(AA)", color=colors_map["AA"])
    ax[0].plot(x, 2 * x * (1 - x), lw=2, label="(Aa)", color=colors_map["Aa"])
    ax[0].plot(x, (1 - x) ** 2, lw=2, label="(aa)", color=colors_map["aa"])
    ax[0].set_ylabel("fréquence\n génotypique")
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0, 1)

    ax[0].legend()

    ax[1].fill_between(
        x, 0, x ** 2, color=colors_map["AA"],
    )
    ax[1].fill_between(
        x, x ** 2, x ** 2 + 2 * x * (1 - x), color=colors_map["Aa"],
    )
    ax[1].fill_between(
        x, x ** 2 + 2 * x * (1 - x), 1, color=colors_map["aa"],
    )
    ax[1].set_ylabel("distribution\ngénotypique")
    ax[1].set_xlabel("fréquence de l'allèle A")
    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(0, 1)
    return fig


def chi2_curve():
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 2)
    x = np.linspace(0, 40, 1000)
    ax.plot(x, chi2.pdf(x, 2), "r-", lw=2, alpha=0.6, label="chi2")
    ax.vlines(32.977, 0, 0.5, linestyle="--")
    ax.set_xlabel("Valeur du Chi2")
    ax.set_ylabel("Probabilité")
    return fig
