import numpy as np
import matplotlib.pyplot as plt



def section(title: str):
    """print to console in nice format"""
    width = 60
    print("\n" + "=" * width)
    print(f"{title.upper():^{width}}")
    print("=" * width + "\n")


def spacer(n: int = 2):
    """print to console in nice format"""
    print("\n" * n, end="")


def hamiltonian(j_coupling= 1, spin_i, spin_j, magnetic= False):
    if magnetic == True:
        raise NotImplementedError('Not implemented')
    
    H = -j_coupling #* np.sum()