import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from functions.transmon_hamiltonian import get_transmon_chain_hamiltonian



def sweep_two_transmons_in_g(qubits, g_vals, basis_size=8, n_levels=6):
    """
    qubit_params: list of dicts [{'EJ':..., 'EC':..., 'ng':...}, {...}]
    g_vals: array
    basis_size: basis dimension for each transmon
    n_levels: number of levels to keep
    """
    energies_all = []

    for g in g_vals:
        couplings = [g]  # 2 transmons --> one coupling term

        H = get_transmon_chain_hamiltonian(qubits, couplings, basis_size) # charge basis

        # diagonalization
        evals = eigsh(H, k=n_levels, which='SA', return_eigenvectors=False)
        evals = np.sort(evals)
        evals -= evals[0]  # reference to ground energy

        energies_all.append(evals)

    return np.array(energies_all)


def main():
    # 2 slightly detuned transmons
    qubits = [
        {'EJ': 50.0, 'EC': 1.0, 'ng': 0.0}, # initially ng = 0 (uncoupled transmons)
        {'EJ': 48.0, 'EC': 1.0, 'ng': 0.0}
    ]

    # sweep in capacitive coupling g
    g_vals = np.linspace(0.0, 0.4, 60) # GHz

    basis_size = 8   # (2*basis_size+1)^2 = 17^2 = 289
    n_levels = 6

    energies = sweep_two_transmons_in_g(
        qubits,
        g_vals,
        basis_size=basis_size,
        n_levels=n_levels
    )

    # Plot: energy levels in terms of g
    plt.figure(figsize=(7, 5))
    for i in range(energies.shape[1]): # energies.shape = (nr of g value, nr of levels)
        plt.plot(g_vals, energies[:, i], label=f"Level {i}") # energies[:,i] = E_i(g1), E_i(g2), ...

    plt.xlabel("g")
    plt.ylabel("E - E0")
    plt.title("2 coupled transmons: hybridization vs g")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()