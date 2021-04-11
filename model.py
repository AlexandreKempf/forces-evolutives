import matplotlib.pyplot as plt
import numpy as np
import seaborn

def run_simultation(nb_gen, N, pA, mutation_rate_A_to_a, mutation_rate_a_to_A):
    pa = 1- pA
    genepool = np.random.choice(["A", "a"], N*2, p=[pA, pa])
    genotypes = np.zeros((3, nb_gen)) # AA, Aa, aa
    for t in range(nb_gen):
        mum = np.random.choice(genepool, N, replace=False)
        dad = np.random.choice(genepool, N, replace=False)

        genotypes[0, t] = np.sum((mum=="A") & (dad=="A"))
        genotypes[1, t] = np.sum(((mum=="A") & (dad =="a")) | ((mum=="a") & (dad =="A")))
        genotypes[2, t] = np.sum((mum=="a") & (dad=="a"))

        genepool = np.concatenate([mum,dad])
        A = genepool=="A"
        mutants_A_to_a = A & (np.random.random(len(A))<mutation_rate_A_to_a)
        a = genepool == "a"
        mutants_a_to_A = a & (np.random.random(len(a))<mutation_rate_a_to_A)
        genepool[mutants_A_to_a] = "a"
        genepool[mutants_a_to_A] = "A"
    return genotypes


def display_multiple_runs(multiple_runs, N):
    colors_map = {
        "red": ("#ff8080", "#800000", "#ff0000"),
        "blue": ("#aaaaff", "#0044aa", "#0066ff"),
        "green": ("#aade87", "#44aa00", "#00d400"),
        "orange": ("#ffb380", "#aa4400", "#ff6600"),
        "black": ("#dddddd", "#bbbbbb", "#555555"),
    }

    def draw_plots(data, ax, color):
        back, main, single = colors_map[color]
        ax.fill_between(np.arange(0, data.shape[1]), np.max(data, 0), np.min(data, 0), color=back)
        ax.plot(data.T, lw=1, color=single)
        ax.plot(np.mean(data, 0), lw=3, color=main)
        return ax

    fig, axes = plt.subplots(4,1)
    axes[0] = draw_plots((multiple_runs[:,0,:] * 2 + multiple_runs[:,1,:]) / (2*N), axes[0], "orange")
    axes[1] = draw_plots(multiple_runs[:,0,:], axes[1], "red")
    axes[2] = draw_plots(multiple_runs[:,1,:], axes[2], "green")
    axes[3] = draw_plots(multiple_runs[:,2,:], axes[3], "blue")

    axes[0].set_ylim(0.0,1.0)
    axes[1].set_ylim(0,N)
    axes[2].set_ylim(0,N)
    axes[3].set_ylim(0,N)

    axes[0].set_ylabel("% allèle A")
    axes[1].set_ylabel("AA")
    axes[2].set_ylabel("Aa")
    axes[3].set_ylabel("aa")

    axes[0].set_xticks([])
    axes[1].set_xticks([])
    axes[2].set_xticks([])
    axes[3].set_xlabel("nombre de générations")
    return fig
