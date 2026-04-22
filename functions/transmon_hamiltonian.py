from functools import reduce

import numpy as np

#
# def uncoupled_transmon_hamiltonian(EJ, EC, ng, basis_size=15): # maximum of 2*15+1 cooper pairs
#     """
#     Constructs Transmon Hamiltonian in charge basis (n) using DIA format
#
#     EJ: Josephson energy
#     EC: charging energy
#     note: EJ and EC need to be given in the same measuring units (ex: GHz)
#     ng: offset charge (adimensional)
#     n_levels_to_return: number of energy levels to return
#     basis_size: dimension of charge basis
#     """
#     n_basis = np.arange(-basis_size, basis_size + 1) # ex: [-15, -14, ..., 0, ..., 14, 15]
#     N = len(n_basis)
#     H = np.zeros((N, N))
#
#     # Diagonal: Electrostatic energy 4 * Ec * (n - ng)^2 on [n, n]
#     for i in range(N):
#         H[i, i] = 4 * EC * (n_basis[i] - ng) ** 2
#
#     # Off-diagonal: Josephson energy -EJ/2 on [n-1, n], [n, n+1]
#     for i in range(N - 1):
#         elem = - 0.5 * EJ
#         H[i, i + 1] = elem
#         H[i + 1, i] = elem
#
#
#     return H


import numpy as np
from scipy.sparse import dia_matrix


def uncoupled_transmon_hamiltonian_dia(EJ, EC, ng, basis_size=15):
    """
    Construiește Hamiltonianul Transmonului folosind formatul sparse DIA.

    EJ: Energia Josephson
    EC: Energia de încărcare
    ng: Sarcina de offset
    basis_size: Dimensiunea bazei de sarcină (n de la -basis_size la +basis_size)
    """
    n_basis = np.arange(-basis_size, basis_size + 1) # arrange creates an array that includes the left and excludes the right with default step 1
    D = len(n_basis) # matrix dimension

    diag_main = 4 * EC * (n_basis - ng) ** 2 # Diagonal: Electrostatic energy 4 * Ec * (n - ng)^2 on [n, n]

    # Off-diagonal: Josephson energy -EJ/2 on [n-1, n], [n, n+1]
    diag_off = np.full(D, -0.5 * EJ) # SciPy DIA necesită ca toate rândurile de date să aibă aceeași lungime D chiar dacă elementele de la capete vor fi ignorate pentru diagonalele secundare.

    data = np.array([diag_off, diag_main, diag_off])
    offsets = np.array([-1, 0, 1])

    H_dia = dia_matrix((data, offsets), shape=(D, D))

    return H_dia

print(uncoupled_transmon_hamiltonian_dia(4.0, 1.0, 0.1))