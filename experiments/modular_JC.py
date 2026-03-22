from functools import reduce

import numpy as np
from math import *
#-----------------------------------------------------------------------------------------------------------------------
# Modular implementation of JC model for N independent systems
#-----------------------------------------------------------------------------------------------------------------------

def get_constants():
    """
        Retrieves fundamental physical constants required for the simulation.

        Returns:
            dict: A dictionary containing:
                - 'h': Planck's constant.
                - 'h_bar': Reduced Planck's constant (h / 2pi).
    """
    return{
        "h": 6.62607015*10**(-34),
        "h_bar": 6.62607015*10**(-34)/(2*pi)
    }


def get_system_lists(n_max):
    """
    Generates the basis state vectors for the atom, the electromagnetic field,
    and their composite system (Atom x Field).

    Args:
        n_max (int): The photon number cutoff (maximum Fock state considered).

    Returns:
        dict: A dictionary containing lists of state vectors:
            - 'atom': Basis states for the Two-Level Atom (|g>, |e>).
            - 'field': Fock states for the field (|0>, |1>, ... |n_max>).
            - 'system': Tensor product states for the composite system.
            - 'vacuum': The global ground state vector.
    """
    #-----atom list-----
    atom_states = []
    ground_state = np.array([0, 1])
    excited_state = np.array([1, 0])
    atom_states.append(ground_state)
    atom_states.append(excited_state)

    # -----field list-----
    field_states = []
    for i in range(0, n_max + 1):
        state = np.zeros(n_max+1)
        state[i] = 1
        field_states.append(state)

    # -----system list-----
    #|g,0>, |e,0>, |g,1>, |e,1> ...
    system_states = []
    for photon in field_states:
        for atom in atom_states:
            state = np.kron(atom, photon)
            system_states.append(state)

    global_vacuum = system_states[0]

    return {
        "atom": atom_states,
        "field": field_states,
        "system": system_states,
        "vacuum": global_vacuum
    }


def get_each_system_operators_as_tensors(n_max):
    """
    Constructs the quantum operators for a single Jaynes-Cummings system
    in the composite Hilbert space (Atom x Field).

    Args:
        n_max (int): The photon number cutoff.

    Returns:
        dict: A dictionary of operators as numpy arrays (matrices):
            - 'r_plus', 'r_minus': Atomic raising/lowering operators.
            - 'a', 'a_dagger': Field annihilation/creation operators.
            - 'X', 'Y', 'Z': Pauli spin matrices expanded in the full space of the specified system.
    """
    # -----raising and lowering operators for atom-----
    r_plus = np.array([[0, 1],
                       [0, 0]])
    r_minus = np.array([[0, 0],
                        [1, 0]])
    I_field = np.eye(n_max + 1, dtype=complex)
    r_plus_tensor = np.kron(r_plus, I_field)
    r_minus_tensor = np.kron(r_minus, I_field)

    #-----annihilation & creation operators for field-----
    a = np.zeros((n_max + 1, n_max + 1), dtype=complex)
    for i in range(1, n_max + 1):
        a[i - 1, i] = np.sqrt(i)
    a_dagger = a.T.conjugate()
    I_atom = np.eye(2, dtype=complex)
    a_tensor = np.kron(I_atom, a)
    adag_tensor = np.kron(I_atom, a_dagger)

    #-----Pauli operators------
    X = np.array([[0, 1],
                  [1, 0]])
    Y = np.array([[0, -1j],
                  [1j, 0]])
    Z = np.array([[1, 0],
                  [0, -1]])
    X_tensor = np.kron(X, I_field)
    Y_tensor = np.kron(Y, I_field)
    Z_tensor = np.kron(Z, I_field)

    return {
        "r_plus": r_plus_tensor,
        "r_minus": r_minus_tensor,
        "a": a_tensor,
        "a_dagger": adag_tensor,
        "X": X_tensor,
        "Y": Y_tensor,
        "Z": Z_tensor
    }


def get_local_hamiltonian_and_identity_list(N, n_max_list, omega_atom_list, omega_field_list, g_list):
    """
    Computes the Jaynes-Cummings Hamiltonian and Identity matrix for each of the N
    independent subsystems.

    This function generates the local Hamiltonian H_k for each system k,
    which acts on the local Hilbert space of dimensions 2*(n_max_k + 1).

    Args:
        N (int): Number of independent atom-field systems.
        n_max_list (list[int]): List of photon cutoffs for each system.
        omega_atom_list (list[float]): Atomic transition frequencies.
        omega_field_list (list[float]): Field mode frequencies.
        g_list (list[float]): Atom-field coupling strengths (Rabi frequency).

    Returns:
        dict: Contains two lists:
            - 'hamiltonian': List of local Hamiltonian matrices [H1, H2, ... HN].
            - 'identity': List of local Identity matrices [I1, I2, ... IN].
    """
    I_atom = np.eye(2, dtype=complex)
    h_bar = get_constants()["h_bar"]
    sum_term = 0
    H_list = []
    I_list = []

    for system in range(0, N, 1):
        I_field = np.eye(n_max_list[system] + 1, dtype=complex)
        g = g_list[system]
        omega_atom = omega_atom_list[system]
        omega_field = omega_field_list[system]
        ops = get_each_system_operators_as_tensors(n_max_list[system])
        a_dagger = ops["a_dagger"]
        a = ops["a"]
        Z = ops["Z"]
        r_plus = ops["r_plus"]
        r_minus = ops["r_minus"]

        H_atom = h_bar * omega_atom * 0.5 * Z
        H_field = h_bar * omega_field * (a_dagger @ a + 0.5 * np.kron(I_atom, I_field))
        H0 = H_atom + H_field
        H_AF = h_bar * g * ((r_plus @ a) + (r_minus @ a_dagger))

        H_system = H0 + H_AF
        I_system = np.kron(I_atom, I_field)

        H_list.append(H_system)
        I_list.append(I_system)

    return{
        "hamiltonian": H_list,
        "identity": I_list
    }


def get_total_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list):
    """
    Constructs the total Hamiltonian for the global Hilbert space of N systems.

    The total Hamiltonian is constructed by tensorizing the local Hamiltonian
    of each system with the identity operators of all other systems:
    H_total = sum(I x ... x H_k x ... x I).

    Args:
        N (int): Number of systems.
        n_max_list (list[int]): Photon cutoffs per system.
        omega_atom_list (list[float]): Atomic frequencies.
        omega_field_list (list[float]): Field frequencies.
        g_list (list[float]): Coupling constants.

    Returns:
        np.ndarray: The full Hamiltonian matrix representing the coupled N-system assembly.
                    Note: The dimension grows exponentially with N.
    """
    summation_terms = []
    system_data = get_local_hamiltonian_and_identity_list(N, n_max_list, omega_atom_list, omega_field_list, g_list)
    for k in range(0, N, 1):
        sum_term = system_data["identity"].copy()
        sum_term[k] = system_data["hamiltonian"][k]
        summation_terms.append(reduce(np.kron,sum_term))

    H = reduce(np.add, summation_terms)

    return H


def test_hamiltonian():
    """
        Unit test to validate the Hamiltonian construction.

        Verifies two properties:
        1. Hermiticity: Checks if H == H_dagger.
        2. Trace Consistency: Verifies if the trace of the total Hamiltonian equals
           the weighted sum of traces of individual subsystems (validating the
           tensor product structure and energy additivity).

        Raises:
            AssertionError: If the calculated properties do not match within numerical tolerance.
    """
    N = 3
    n_max_list = [1, 2, 3]
    omega_atom_list = [1.0, 1.2, 1.5]
    omega_field_list = [0.8, 1.0, 1.4]
    g_list_coupled = [0.10, 0.05, 0.15]  # RWA: weak coupling - g << omega_atom, omega_field
    H = get_total_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_coupled)

    assert np.allclose(H, H.T.conjugate()) == True #verif if hermitian

    #g_list = [0] * N
    eigvals = np.linalg.eigvals(H)
    total_energy = np.sum(eigvals) #Trace of H

    sum_of_energies = 0
    dim_hilbert_space = np.prod([2 * (i+1) for i in n_max_list]) #each atom has only two states

    for system in range(0, N, 1):
        H_system = get_local_hamiltonian_and_identity_list(N, n_max_list, omega_atom_list, omega_field_list, g_list_coupled)["hamiltonian"][system] #g_list_coupled could also be g_list, interacțiunea doar rotește
        # baza, nu schimbă suma totală a elementelor diagonale (urma).
        eigvals_k = np.linalg.eigvals(H_system)

        dim_system = 2 * (n_max_list[system] + 1)
        dim_rest_of_space = dim_hilbert_space // dim_system #integer division bcs float looses precision for big numbers

        sum_of_energies += np.sum(eigvals_k) * dim_rest_of_space #Tr(H) = Tr(H_system1) (tensor_product) Tr(I_2 tp T_3 tp I_4 tp...tp I_N)

    assert np.allclose(sum_of_energies, total_energy) == True #verif if sum of partial energies is equal to total energy, for independent uncoupled systems

test_hamiltonian()