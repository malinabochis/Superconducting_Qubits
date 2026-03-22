import numpy as np
from math import *
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------------------------
# Initial hardcoded implementation of JC model
#-----------------------------------------------------------------------------------------------------------------------
'constants'

h = 6.62607015*10**(-34)
h_bar = h/(2*pi)
n_max = 3 #nr maxim de fotoni din cavitate (=> n_max+1 nivele, pt ca luam de la nivelul 0)
omega_atom = 1.0 #atomic transition frequency
omega_field = 0.8 #frequency associated to the field mode
g = 0.1 #coupling constant = the strength of the interaction between atom and EM field
#g << omega_atom, omega_field => RWA (weak coupling)

#-----------------------------------------------------------------------------------------------------------------------
'raising and lowering operators for atom'

r_plus = np.array([[0,1],
                   [0,0]])
r_minus = np.array([[0,0],
                    [1,0]])
I_field = np.eye(n_max+1, dtype = complex)
r_plus_tensor = np.kron(r_plus, I_field) # 2*(n_max+1) x 2*(n_max+1) ; r+ only acts on atom, leaving field the same for the Hilbert space: atom x field
r_minus_tensor = np.kron(r_minus, I_field) #r- only acts on atom, leaving field the same for the Hilbert space: atom x field

#-----------------------------------------------------------------------------------------------------------------------
'annihhilation & creation operators for field'
'''
a = np.array([[0,1],
              [0,0]], dtype = complex)
a_dagger = np.array([[0,0],
                     [1,0]], dtype = complex)
I_atom = np.eye(2)
a_tensor = np.kron(I_atom, a) #a only acts on field, leaving atom the same for the Hilbert space: atom x field
adag_tensor = np.kron(I_atom, a_dagger) #a+ only acts on field, leaving atom the same for the Hilbert space: atom x field
'''
a = np.zeros((n_max+1, n_max+1), dtype = complex)
for i in range(1, n_max+1):
    a[i-1,i] = np.sqrt(i)
a_dagger = a.T.conjugate()
I_atom = np.eye(2, dtype = complex)
a_tensor = np.kron(I_atom, a) #a only acts on field, leaving atom the same for the Hilbert space: atom x field
adag_tensor = np.kron(I_atom, a_dagger) #a+ only acts on field, leaving atom the same for the Hilbert space: atom x field

#-----------------------------------------------------------------------------------------------------------------------
'Pauli operators'

X = np.array([[0,1],
              [1,0]])
Y = np.array([[0,-1j],
              [1j, 0]])
Z = np.array([[1, 0],
              [0,-1]])
X_tensor = np.kron(X, I_field)
Y_tensor = np.kron(Y, I_field)
Z_tensor = np.kron(Z, I_field)

#-----------------------------------------------------------------------------------------------------------------------
'atom states'
atom_state = []
ground_state = np.array([0,1])
excited_state = np.array([1,0])
atom_state.append(ground_state)
atom_state.append(excited_state)

#-----------------------------------------------------------------------------------------------------------------------
'field states'
n_state = []
for i in range(0, n_max+1):
    state = np.eye(1,n_max+1,i) #matrice 2D cu 1 linie si n_max+1 coloane
    state = state.flatten() #vector 1D
    n_state.append(state) #lista de vectori 1D
#np.eye(N,M,k), N=nr de linii, M=nr de coloane, k=pune 1 pe diagonala cu koffset fata de diag principala

vacuum = [0]
for i in range(1, n_max+1): vacuum = vacuum+[0] #niciun foton in cavitate |0,0,0,0,0>
'''
n0 = n_state[0] #1 foton in cavitate  |1,0,0,0,0>
n1 = n_state[1] #2 fotoni in cavitate |0,1,0,0,0>
n2 = n_state[2] #3 fotoni in cavitate |0,0,1,0,0>
n3 = n_state[3] #4 fotoni in cavitate |0,0,0,1,0>
n4 = n_state[4] #5 fotoni in cavitate |0,0,0,0,1>
'''

#-----------------------------------------------------------------------------------------------------------------------
'system states'
system_state = []
for i in range(0, n_max+1):
    for j in range(0, 2):
        system_state.append(np.kron(atom_state[j],n_state[i]))

#-----------------------------------------------------------------------------------------------------------------------
'commutation relations'

comm_a_a_dagger = a_tensor@adag_tensor - adag_tensor@a_tensor #[a,a_dagger] = I_atom X I_field
comm_r_plus_r_minus = r_plus_tensor@r_minus_tensor - r_minus_tensor@r_plus_tensor #[r+, r-] = Z X I_field
comm_Z_r_plus = Z_tensor@r_plus_tensor - r_plus_tensor@Z_tensor #[Z,r+] = 2r+ X I_field
comm_Z_r_minus = Z_tensor@r_minus_tensor - r_minus_tensor@Z_tensor #[Z,r-] = -2r- X I_field

#-----------------------------------------------------------------------------------------------------------------------
'Hamiltonian'
H_atom = h_bar * omega_atom * 0.5 * Z_tensor #actionez pe atom si extind la tot spatiul
H_field = h_bar * omega_field * (adag_tensor @ a_tensor + 0.5 * np.kron(I_atom, I_field) ) #actionez pe camp si extind la tot spatiul
H0 = H_atom + H_field
H_AF = h_bar * g * (np.kron(r_plus, a) + np.kron(r_minus, a_dagger) ) #actioneaza simultan pe atom si camp, deci deja lucreaza in spatiul extins, deci nu mai tensorizez
H = H0 + H_AF

#print(np.allclose(H, H.T.conjugate())) #verif if hermitian

#-----------------------------------------------------------------------------------------------------------------------
'Eigenvalues, Eigenvectors of H0 and H'
eigvals0, eigvecs0 = np.linalg.eigh(H0)
print("Energies for uncoupled system: ", eigvals0)

eigvals, eigvecs = np.linalg.eigh(H)
print("Energies for coupled system: ", eigvals)

plt.plot(eigvals0, 'o', label="uncoupled system")
plt.plot(eigvals, 'o', label="coupled system")
plt.legend()
plt.ylabel("Energy")
plt.show()

#-----------------------------------------------------------------------------------------------------------------------

detunings = []
energies_uncoupled = []
energies_coupled = []

for i in np.arange(-5, 5.1, 0.5):
    omega_field = i
    detuning_parameter = omega_atom - omega_field  # ESTE MEREU omega_atom MAI MARE??
    detunings.append(detuning_parameter)

    H_field = h_bar * omega_field * (adag_tensor @ a_tensor + 0.5 * np.kron(I_atom, I_field))
    H0 = H_atom + H_field
    H = H0 + H_AF

    eigvals0, eigvecs0 = np.linalg.eigh(H0)
    energies_uncoupled.append(eigvals0)

    eigvals, eigvecs = np.linalg.eigh(H)
    energies_coupled.append(eigvals)

for k in range(len(energies_uncoupled[0])): #fiecare nivel de energie dintre toate cele n_max+1 nivele
    plt.plot(detunings, [level[k] for level in energies_uncoupled], 'o', color='blue')
    plt.plot(detunings, [level[k] for level in energies_coupled], 'o', color='orange')

plt.xlabel("Detuning")
plt.ylabel("Energy")
plt.show()