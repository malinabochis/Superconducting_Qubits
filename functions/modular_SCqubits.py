import numpy as np
from anyio import sleep_until
from scipy.sparse import dia_matrix
from dataclasses import dataclass

class Constant:
    @staticmethod
    def h(): return 6.62607015e-34 # J*s

    @staticmethod
    def h_reduced(): return 6.62607015e-34 / (2 * np.pi)

    @staticmethod
    def e(): return 1.602176634e-19 # C

#
# @dataclass(frozen=True)
# class Constant:
#     h: float = 6.62607015e-34 # J*s
#     e: float = 1.602176634e-19 # C

class Notation:
    h = Constant.h()
    e = Constant.e()

    @classmethod
    def Phi_0(cls): return cls.h / (2 * cls.e)

    @classmethod
    def ng(cls, Cg, Vg): return (Cg * Vg) / (2 * cls.e)


# gate charge period = Vg necessary to change ng by 1
# delta_ng = Cg*delta_Vg/2e = 1 => gate_charge_period = 1/delta_Vg

class ExpData:
    def __init__(self, freq01=None, anh=None, ng=None, Cg=None, Vg=None, gate_charge_period=None, EJ=None, EC=None):
        self.freq01 = freq01
        self.anh = anh
        self.ng = ng
        self.Cg = Cg
        self.Vg = Vg
        self.gate_charge_period = gate_charge_period
        self.EJ = EJ
        self.EC = EC

class Transmon:
    def __init__(self, **kwargs):
        self.exp_data = ExpData(**kwargs)
# ----------------------------------------------------
    @property
    def freq01(self): return self.exp_data.freq01

    @property
    def anh(self): return self.exp_data.anh

    @property
    def EC(self):
        if self.exp_data.EC is not None: return self.exp_data.EC
        else: return -self.exp_data.anh  # EC = -alpha

    @property
    def EJ(self):
        if self.exp_data.EJ is not None: return self.exp_data.EJ
        else:
            freq01 = self.exp_data.freq01
            EC = self.EC
            return ((freq01 * Constant.h_reduced() + EC) ** 2) / (8 * EC)

    @property
    def ng(self):
        if self.exp_data.ng is not None:
            return self.exp_data.ng
        else:
            Vg = self.exp_data.Vg
            gate_charge_period = self.exp_data.gate_charge_period
            if not gate_charge_period:
                raise ValueError("gate charge period cannot be 0")
            else:
                return Vg / gate_charge_period # + ng(Vg=0) ??? rezidual exp value
# ----------------------------------------------------
    @freq01.setter
    def freq01(self, value): self.exp_data.freq01 = value

    @anh.setter
    def anh(self, value): self.exp_data.anh = value

    @EJ.setter
    def EJ(self, value): self.exp_data.EJ = value

    @EC.setter
    def EC(self, value): self.exp_data.EC = value

    @ng.setter
    def ng(self, value):
        self.exp_data.ng = value
# ----------------------------------------------------
    def TransmonPotential(self):
        EJ = self.EJ



    def UncoupledTransmonHamiltonianDia(self, EJ, EC, ng, basis_size=15): # basis size is |n> a.k.a. number of Cooper pairs relative to the ground state of the superconducting island
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