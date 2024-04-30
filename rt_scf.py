import numpy as np
from pyscf import gto, dft, scf
import rt_integrators
import rt_observables
import rt_output
import rt_cap

'''
Real-time SCF main driver
'''

class rt_scf:
    def __init__(self, mf, timestep, frequency, total_steps, filename, CAP=None, prop=None, orth=None):
        self.timestep = timestep
        self.frequency = frequency
        self.total_steps = total_steps
        self.filename = filename
        self._scf = mf
        self.CAP = CAP
        self.mag = False
        self.fragments = []
        self.ovlp = self._scf.get_ovlp()
        self.bfield = False
        self.delta_field = False
        self.magnus_tolerance = 1e-7
        if prop is None: self.prop = "magnus_interpol"
        if orth is None: self.orth = scf.addons.canonical_orth_(self.ovlp)


        # Get number of molecular orbitals/electrons
        nmo = self._scf.mol.nao_nr()
        nelec_alpha, nelec_beta = self._scf.mol.nelec[0], self._scf.mol.nelec[1]

        occ_alpha = np.concatenate((np.ones(nelec_alpha), np.zeros(nmo-nelec_alpha)))
        occ_beta = np.concatenate((np.ones(nelec_beta), np.zeros(nmo-nelec_beta)))

        # Determine number of matrices: 1 for closed shell/generalized, 2 for open shell
        match mf:
            case dft.rks.RKS() | scf.rhf.RHF():
                self.nmat = 1
                self.occ = occ_alpha + occ_beta
            case dft.uks.UKS() | scf.uhf.UHF():
                self.nmat = 2
                self.occ = np.stack((occ_alpha,occ_beta))
            case dft.gks.GKS() | scf.ghf.GHF():
                self.mag = True
                self.nmat = 1
                self.occ = np.concatenate((np.ones(nelec_alpha+nelec_beta), np.zeros(2*nmo-nelec_alpha-nelec_beta)))
            case _:
                raise Exception('unknown scf method')

        if self.nmat == 1:
            self.dim = np.array([nmo, nmo])
        else:
            self.dim = np.array([self.nmat, nmo, nmo])

        self.den_ao = self._scf.make_rdm1(mo_occ=self.occ)

    def get_fock_orth(self, den_ao):

        fock = self._scf.get_fock(h1e=self.hcore, dm=den_ao)

        if self.CAP:
            if self.nmat == 1:
                fock_orth = rt_cap.mocap(self, fock)
            else:
                fock_orth = np.zeros(self.dim).astype(np.complex128)
                fock_orth[0] = rt_cap.mocap(self, fock[0])
                fock_orth[1] = rt_cap.mocap(self, fock[1])
        else:
            if self.nmat == 1:
                fock_orth = np.dot(self.orth.T,np.dot(fock,self.orth))
            else:
                fock_orth = np.zeros(self.dim).astype(np.complex128)
                fock_orth[0] = np.dot(self.orth.T,np.dot(fock[0],self.orth))
                fock_orth[1] = np.dot(self.orth.T,np.dot(fock[1],self.orth))

        return fock_orth

    def kernel(self, mo_coeff_print=None):

        rt_output.create_output_file(self)

        if mo_coeff_print is None: mo_coeff_print = self._scf.mo_coeff

        self.hcore = self._scf.get_hcore(self._scf.mol)

        if self.prop == "magnus_step":
            mo_coeff_old = self._scf.mo_coeff
        elif self.prop == "magnus_interpol":
            fock_orth_n12dt = self.get_fock_orth(self.den_ao)

        # Start propagation
        for i in range(0, self.total_steps):
        
            t = (i * self.timestep)
            if np.mod(i, self.frequency) == 0:
                rt_observables.get_observables(self, t, mo_coeff_print)

            match self.prop:
                case "magnus_step":
                    if self.nmat == 1:
                        den_ao_old = rt_integrators.prop_magnus_step(self, den_ao_old)
                    else:
                        den_ao_old = rt_integrators.prop_magnus_step_os(self, den_ao_old)
                case "magnus_interpol":
                    if self.nmat == 1:
                        fock_orth_n12dt = rt_integrators.prop_magnus_ord2_interpol(self, fock_orth_n12dt)
                    else:
                        fock_orth_n12dt = rt_integrators.prop_magnus_ord2_interpol_os(self, fock_orth_n12dt)
                case "rk4":
                    if self.nmat == 1:
                        rt_integrators.rk4(self)
                    else:
                        rt_integrators.rk4_os(self)
                case _:
                    raise Exception("unknown propagator")