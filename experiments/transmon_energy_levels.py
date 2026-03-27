import numpy as np
import matplotlib.pyplot as plt

# def k_sorting(n, ng):
#     """
#
#     :param n: qubit energy level index
#     :param ng: offset / gate charge
#     :return:
#     """
#     term1 = -np.round(ng) # round(x) is the nearest integer to the real number x, it changes value at half whole numbers => Brillouin zone boundary
#     # floor(x + 0.5) is used to solve the round(x) problem that it rounds numbers to nearest even number
#     term2 = 0.25 * ( (-1) ** (np.floor(ng)) ) # floor(x) is the integer immediately below x, it changes value only at whole numbers
#     term3 = -1 + ( (-1) ** n) * (1 + 2*n)
#
#     return term1 + term2 * term3
#
# #
# # energy_raport = E_j / E_c
# # q = - 0.5 * energy_raport
#
# from scipy.special import mathieu_a
#
#
# def get_transmon_energies_mathieu(n, ng, EJ_EC_ratio):
#     """
#     Calculeaza energia nivelului n conform Eq. 2.2 din Koch et al.
#     E_C este setat conventional la 1 pentru calculul raportului.
#     """
#     EC = 1.0
#     # EJ = EJ_EC_ratio * EC
#     q = - 0.5 * EJ_EC_ratio
#
#     # Order v for mathieu characteristic value a_v
#     v = 2 * (ng + k_sorting(n, ng))
#     return EC * mathieu_a(v, q) # returns energy scaled about chosen unit (EC)
#     # mathieu_a(level to be calculated, potential form)
#     #a_v(q) = functia a de ordin v evaluata in q


from functions.transmon_hamiltonian import *


def get_transmon_energies(n_states_to_return, ng, EJ_EC_ratio):
    """
    Calculates eigenenergies
    """
    EC = 1.0
    EJ = EJ_EC_ratio * EC

    # define charge basis states:
    n_basis = np.arange(-15, 16)
    basis_size = len(n_basis)

    H = uncoupled_transmon_hamiltonian(EJ, EC, ng, basis_size)

    energies = np.linalg.eigvalsh(H)
    return energies[:n_states_to_return]


def main():
    ng_axis = np.linspace(-2, 2, 500) # 500 experimental points
    ratios = [1.0, 5.0, 10.0, 50.0, 300.0]
    levels = [0, 1, 2]
    colors = ['black', 'red', 'blue', 'green', 'brown', 'yellow', 'orange']
    n_states_to_return = len(levels)

    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # axes = axes.flatten()

    for ratio in ratios:
        plt.figure(figsize=(6, 5)) # new figure for each EJ/EC ratio

        # normalization factor E01 at ng = 0.5
        energies_ss = get_transmon_energies(n_states_to_return, 0.5, ratio)
        E01_ref = energies_ss[1] - energies_ss[0]

        all_energies = np.array([get_transmon_energies(n_states_to_return, ng, ratio) for ng in ng_axis])
        e0_min = np.min(all_energies[:, 0]) # will set the zero point energy as the bottom of m = 0 level

        # plot normalized energy levels:
        for m in levels:
            energies_raw = all_energies[:, m]
            # Normalization: (E - min(E0)) / E01_ref
            energies_norm = (energies_raw - e0_min) / E01_ref
            plt.plot(ng_axis, energies_norm, color=colors[m], label=f'm={m}', linewidth=1.5)

        plt.title(f'Transmon Energy Spectrum ($E_J/E_C = {ratio}$)')
        plt.xlabel('$n_g$ (Gate Charge)')
        plt.ylabel('$E_m / E_{01}$ (Normalized Energy)')

        plt.ylim(-0.5, 10 if ratio == 1.0 else 3) # adjusting Y scale to fit the regime (Charge vs Transmon)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()

        plt.show()

main()