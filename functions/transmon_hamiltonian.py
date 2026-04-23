# from functools import reduce
#
# import numpy as np

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
from scipy import sparse
from scipy.sparse import dia_matrix


def get_uncoupled_transmon_hamiltonian_dia(EJ, EC, ng, basis_size=15): # basis size is |n> a.k.a. number of Cooper pairs relative to the ground state of the superconducting island
    """
    Constructs Transmon Hamiltonian using DIA sparse format.

    EJ: Josephson Energy
    EC: Charging Energy
    ng: Offset charge
    basis_size: Dimension of the charge basis(n from -basis_size to +basis_size)
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

# print(get_uncoupled_transmon_hamiltonian_dia(4.0, 1.0, 0.1))


def get_two_transmon_hamiltonian_csc(params, g_coupling, basis_size=15):
    """
    params: list [dict1, dict2] where each dict is {'EJ':..., 'EC':..., 'ng':...}
    returns H matrix in CSC format
    """
    d = 2 * basis_size + 1 # works only when gate charge ng is not too big, otherwise the 0 point of oscillation is shifted
    D_tot = d ** 2
    H_dok = sparse.dok_matrix((D_tot, D_tot), dtype=complex)

    H_locals = [] # will be a list of DIA sparse hamiltonians
    for transmon_i in params:
        H_locals.append(get_uncoupled_transmon_hamiltonian_dia(transmon_i['EJ'], transmon_i['EC'], transmon_i['ng'], basis_size))

    # H1 ⊗ I2
    for offset, data in zip(H_locals[0].offsets, H_locals[0].data):
        for r1, val in enumerate(data): # parcurge vectorul data[] si extrage perechea (pozitie/rand, valoare)
            c1 = r1 + offset
            if 0 <= c1 < d:
                for i in range(d):
                    H_dok[r1 * d + i, c1 * d + i] += val

    # I1 ⊗ H2
    for offset, data in zip(H_locals[1].offsets, H_locals[1].data):
        for r2, val in enumerate(data):
            c2 = r2 + offset
            if 0 <= c2 < d:
                for j in range(d):
                    H_dok[j * d + r2, j * d + c2] += val

    # Coupling g * n1 ⊗ n2
    n_vals = np.arange(-basis_size, basis_size + 1)
    for i in range(d):
        for j in range(d):
            H_dok[i * d + j, i * d + j] += g_coupling * n_vals[i] * n_vals[j]

    return H_dok.tocsc()  # CSC format for improved computation (sparse matrix addition, sparse matrix tensor product, using diagonalization algorithms that work with columns, etc.)


def get_transmon_chain_hamiltonian(qubit_params_list, couplings_list, basis_size=15):
    """
    qubit_params_list: [dict1, dict2, ..., dictN] (size N)
    couplings_list: [g12, g23, ..., g_{N-1,N}] (size N-1)
    """
    N = len(qubit_params_list)
    d = 2 * basis_size + 1
    I_single_transmon = sparse.eye(d, format='csc') # local identity matrix

    # Sum( I ⊗ ... ⊗ Hi ⊗ ... ⊗ I )
    H_total = None

    for i in range(N):
        p = qubit_params_list[i]
        Hi = get_uncoupled_transmon_hamiltonian_dia( p['EJ'], p['EC'], p['ng'], basis_size ).tocsc()

        H_ext = Hi # Hi in the global space

        for _ in range(i):
            H_ext = sparse.kron(I_single_transmon, H_ext, format='csc')

        for _ in range(N - 1 - i):
            H_ext = sparse.kron(H_ext, I_single_transmon, format='csc')

        if H_total is None: # only enters this branch after the first loop
            H_total = H_ext
        else:
            H_total += H_ext


    def __expand_to_global(op_local, pos, N, d):
        """Funcție utilitară pentru a pune un operator local în spațiul global"""
        I_single_transmon = sparse.eye(d, format='csc')
        op_global = op_local
        for _ in range(pos):
            op_global = sparse.kron(I_single_transmon, op_global, format='csc')
        for _ in range(N - 1 - pos):
            op_global = sparse.kron(op_global, I_single_transmon, format='csc')
        return op_global

    n_local = sparse.diags(np.arange(-basis_size, basis_size + 1), format='csc')

    # Interaction: Sum( g_i * n_i ⊗ n_{i+1} )
    for i in range(N - 1):
        g_i = couplings_list[i]

        ni_glob = __expand_to_global(n_local, i, N, d)
        n_next_glob = __expand_to_global(n_local, i + 1, N, d)

        H_total += g_i * (ni_glob @ n_next_glob)

    return H_total