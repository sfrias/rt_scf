# rt_pyscf
Real-Time Electronic Structure Package for PySCF

Currently includes:
  1. Real-time propagation of molecular orbital (MO) coefficient matrix
  2. Supports restricted, unrestricted, and generalized hf/dft methods.
  3. Observables:
        - MO occupations
        - Charge
        - Energy
        - Dipole
        - Magnetization
  4. Propagators:
        - 2nd Order Magnus Step (MMUT)
        - 2nd Order Interpolated Magnus
        - Runge-Kutta 4
  5. Functionality:
        - Localized "noscf" basis
        - Static bfield
        - Excitation
        - Complex Absorbing Potential (CAP)


How to use:
1. Initialize system using hf/dft method of choice in pyscf.
2. Create rt_scf class from the static scf object, enter propagation parameters.
        - rt_scf class must be given the following (mf, timestep, frequency, total_steps, filename)
3. Call the kernel() function to start propagation.
