import numpy as np

def hamiltonian(j_coupling= 1, spin_i, spin_j, magnetic= False):
    if magnetic == True:
        raise NotImplementedError('Not implemented')
    
    H = -j_coupling #* np.sum()