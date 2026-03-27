import numpy as np


def uncoupled_transmon_hamiltonian(EJ, EC, ng, basis_size=15):
    """
    Diagonalizes Transmon Hamiltonian in charge basis (n).

    EJ: Josephson energy
    EC: charging energy
    note: EJ and EC need to be given in the same measuring units (ex: GHz)
    ng: offset charge (adimensional)
    n_levels_to_return: number of energy levels to return
    basis_size: dimension of charge basis
    """
    n_basis = np.arange(-basis_size, basis_size + 1) # ex: [-15, -14, ..., 0, ..., 14, 15]
    N = len(n_basis)
    H = np.zeros((N, N))

    # Diagonal: Electrostatic energy 4 * Ec * (n - ng)^2 on [n, n]
    for i in range(N):
        H[i, i] = 4 * EC * (n_basis[i] - ng) ** 2

    # Off-diagonal: Josephson energy -EJ/2 on [n-1, n], [n, n+1]
    for i in range(N - 1):
        elem = - 0.5 * EJ
        H[i, i + 1] = elem
        H[i + 1, i] = elem

    return H
















