from functions.modular_JC import *
import matplotlib.pyplot as plt


def main():
    N = 1
    n_max_list = [3]

    #------------UNCOUPLED VS COUPLED SYSTEM, omega_atom != omega_field------------
    omega_atom_list = [1.0]
    omega_field_list = [0.8]

    g_list_uncoupled = [0.0]
    test_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_uncoupled)
    H0 = get_total_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_uncoupled)

    g_list_coupled = [0.10]  # RWA: weak coupling - g << omega_atom, omega_field
    test_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_coupled)
    H = get_total_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_coupled)

    print("UNCOUPLED VS COUPLED SYSTEM, omega_atom != omega_field\n")
    'Eigenvalues, Eigenvectors of H0 and H'
    eigvals0, eigvecs0 = np.linalg.eigh(H0)
    print("Energies for uncoupled system: ", eigvals0)

    eigvals, eigvecs = np.linalg.eigh(H)
    print("Energies for coupled system: ", eigvals)

    plt.plot(eigvals0, 'o', label="uncoupled system")
    plt.plot(eigvals, 'o', label="coupled system")
    plt.legend()
    plt.xlabel("Indexul nivelului de energie (k)")
    plt.ylabel("Energy")
    plt.show()

    # ------------UNCOUPLED VS COUPLED SYSTEM, omega_atom = omega_field (resonance) ------------
    omega_atom_list = [1.0]
    omega_field_list = [1.0]

    g_list_uncoupled = [0.0]
    test_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_uncoupled)
    H0 = get_total_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_uncoupled)

    g_list_coupled = [0.10]  # RWA: weak coupling - g << omega_atom, omega_field
    test_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_coupled)
    H = get_total_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_coupled)

    print("\n\nUNCOUPLED VS COUPLED SYSTEM, omega_atom = omega_field (resonance)\n")
    'Eigenvalues, Eigenvectors of H0 and H'
    eigvals0, eigvecs0 = np.linalg.eigh(H0)
    print("Energies for uncoupled system: ", eigvals0)

    eigvals, eigvecs = np.linalg.eigh(H)
    print("Energies for coupled system: ", eigvals)

    plt.plot(eigvals0, 'o', label="uncoupled system")
    plt.plot(eigvals, 'o', label="coupled system")
    plt.legend()
    plt.xlabel("Indexul nivelului de energie (k)")
    plt.ylabel("Energy")
    plt.show()

    """
    Se observa ca, la rezonanta:
        - sistemul necuplat are energii degenerate
        - sistemul cuplat are energiile sunt nedegenerate
    """

    # ------------UNCOUPLED VS COUPLED SYSTEM, detuning ------------
    omega_atom_list = [1.0]

    detunings = []
    energies_uncoupled = []
    energies_coupled = []

    for field_frequency in np.arange(-5, 5.1, 0.5):
        omega_field_list[0] = field_frequency
        detuning_parameter = omega_atom_list[0] - omega_field_list[0]  # ESTE MEREU omega_atom MAI MARE??
        detunings.append(detuning_parameter)

        H0 = get_total_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_uncoupled)
        H = get_total_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_coupled)

        eigvals0, eigvecs0 = np.linalg.eigh(H0)
        energies_uncoupled.append(eigvals0)

        eigvals, eigvecs = np.linalg.eigh(H)
        energies_coupled.append(eigvals)

    for k in range(len(energies_uncoupled[0])):  # fiecare nivel de energie dintre toate cele n_max+1 nivele
        plt.plot(detunings, [level[k] for level in energies_uncoupled], 'o', color='blue')
        plt.plot(detunings, [level[k] for level in energies_coupled], 'o', color='orange')

    plt.xlabel("Detuning")
    plt.ylabel("Energy")
    plt.show()





    detunings = []
    energies_uncoupled = []
    energies_coupled = []

    for field_frequency in np.arange(1.0, 2.1, 0.1):
        omega_field_list[0] = field_frequency
        detuning_parameter = omega_atom_list[0] - omega_field_list[0]  # ESTE MEREU omega_atom MAI MARE??
        detunings.append(detuning_parameter)

        H0 = get_total_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_uncoupled)
        H = get_total_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_coupled)

        eigvals0, eigvecs0 = np.linalg.eigh(H0)
        energies_uncoupled.append(eigvals0)

        eigvals, eigvecs = np.linalg.eigh(H)
        energies_coupled.append(eigvals)

    for k in range(len(energies_uncoupled[0])):  # fiecare nivel de energie dintre toate cele n_max+1 nivele
        plt.plot(detunings, [level[k] for level in energies_uncoupled], 'o', color='blue')
        plt.plot(detunings, [level[k] for level in energies_coupled], 'o', color='orange')

    plt.xlabel("Detuning")
    plt.ylabel("Energy")
    plt.show()



    detunings = []
    energies_uncoupled = []
    energies_coupled = []

    for field_frequency in np.arange(0.0, 1.1, 0.1):
        omega_field_list[0] = field_frequency
        detuning_parameter = omega_atom_list[0] - omega_field_list[0]  # ESTE MEREU omega_atom MAI MARE??
        detunings.append(detuning_parameter)

        H0 = get_total_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_uncoupled)
        H = get_total_hamiltonian(N, n_max_list, omega_atom_list, omega_field_list, g_list_coupled)

        eigvals0, eigvecs0 = np.linalg.eigh(H0)
        energies_uncoupled.append(eigvals0)

        eigvals, eigvecs = np.linalg.eigh(H)
        energies_coupled.append(eigvals)

    for k in range(len(energies_uncoupled[0])):  # fiecare nivel de energie dintre toate cele n_max+1 nivele
        plt.plot(detunings, [level[k] for level in energies_uncoupled], 'o', color='blue')
        plt.plot(detunings, [level[k] for level in energies_coupled], 'o', color='orange')

    plt.xlabel("Detuning")
    plt.ylabel("Energy")
    plt.show()

    """
    Se observa ca, cu cat frecventa atomului se indeparteaza de cea a cavitatii (detuningul creste in modul),
    sistemul cuplat se comporta tot mai mult ca un sistem necuplat
    """

main()