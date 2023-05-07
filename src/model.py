import matplotlib.pyplot as plt
import numpy as np
import seaborn


def run_simulation(
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
    N_Aa = np.maximum(N - (N_AA + N_aa), 0)
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
            total = N_AA + N_Aa + N_aa
            N_AA = N * N_AA / total
            N_Aa = N * N_Aa / total
            N_aa = N * N_aa / total

        # mutation
        if mutation_rate_A_to_a != 0 or mutation_rate_a_to_A != 0:
            mut_AA_to_Aa = np.random.binomial(N_AA, mutation_rate_A_to_a)
            mut_Aa_to_aa = np.random.binomial(N_Aa, mutation_rate_A_to_a / 2)
            mut_Aa_to_AA = np.random.binomial(N_Aa, mutation_rate_a_to_A / 2)
            mut_aa_to_Aa = np.random.binomial(N_aa, mutation_rate_a_to_A)
            N_AA += mut_Aa_to_AA - mut_AA_to_Aa
            N_Aa += mut_AA_to_Aa - mut_Aa_to_AA - mut_Aa_to_aa + mut_aa_to_Aa
            N_aa += mut_Aa_to_aa - mut_aa_to_Aa
            total = N_AA + N_Aa + N_aa
            N_AA = N * N_AA / total
            N_Aa = N * N_Aa / total
            N_aa = N * N_aa / total

        genotypes[0, t] = N_AA
        genotypes[1, t] = N_Aa
        genotypes[2, t] = N_aa
    return genotypes
