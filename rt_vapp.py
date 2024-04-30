import numpy as np

'''
Real-time SCF Applied Potential
'''

def static_bfield(rt_mf, bfield):
    rt_mf.bfield = True
    x_bfield = bfield[0]
    y_bfield = bfield[1] 
    z_bfield = bfield[2]

    hcore = rt_mf._scf.get_hcore()
    Nsp = int(rt_mf.ovlp.shape[0]/2)

    ovlp = rt_mf.ovlp[:Nsp,:Nsp]
    hcore = hcore[:Nsp,:Nsp]

    hprime = np.zeros([2*Nsp,2*Nsp], dtype=complex)

    hprime[:Nsp,:Nsp] = hcore + 0.5 * z_bfield * ovlp
    hprime[Nsp:,Nsp:] = hcore - 0.5 * z_bfield * ovlp
    hprime[Nsp:,:Nsp] = 0.5 * (x_bfield + 1j * y_bfield) * ovlp
    hprime[:Nsp,Nsp:] = 0.5 * (x_bfield - 1j * y_bfield) * ovlp
    rt_mf._scf.get_hcore = lambda *args: hprime

def delta_field(rt_mf, field):
    pass