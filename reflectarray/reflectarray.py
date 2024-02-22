import numpy as np
import scipy.constants
from reflectarray import toolbox as tb
tb.set_font(fontsize=15)

C = scipy.constants.c
EPS_0 = scipy.constants.epsilon_0
MU_0 = scipy.constants.mu_0

class Reflectarray:
    '''
    Defines a physical, fixed reflectarray antenna in terms of dipole positions, dipole complex magnitudes, and frequencies.
    Handles various methods of supplying dipole positions, ultimately converted to (# positions) x 3 array.
    '''

    def __init__(self, **kwargs):