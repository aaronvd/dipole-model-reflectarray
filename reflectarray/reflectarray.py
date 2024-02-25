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

    def __init__(self, element, **kwargs):

        self.element = element
        self.f = self.element.f

        self.r = kwargs.get('r', None)
        self.x = kwargs.get('x', None)
        self.y = kwargs.get('y', None)
        self.quiet = kwargs.get('quiet', False)
        if self.r is None:
            if ((self.x is not None) and (self.y is None)):
                self.y = np.zeros_like(self.x)
                self.z = np.zeros_like(self.x)
            elif ((self.y is not None) and (self.x is None)):
                self.x = np.zeros_like(self.y)
                self.z = np.zeros_like(self.x)
            elif self.x is not None:
                self.z = np.zeros_like(self.x)
            elif (any([i in kwargs for i in ['delta_x', 'Nx', 'Lx']])) or (any([i in kwargs for i in ['delta_y', 'Ny', 'Ly']])):
                if sum([i in kwargs for i in ['delta_x', 'Nx', 'Lx']]) == 1:
                    raise Exception('Must supply two out of three of delta_x, Nx, or Lx.')
                elif ('delta_x' in kwargs) and ('Nx' in kwargs):
                    self.delta_x = kwargs.get('delta_x')
                    self.Nx = kwargs.get('Nx')
                    self.Lx = (self.Nx - 1) * self.delta_x
                    self.x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)
                elif ('delta_x' in kwargs) and ('Lx' in kwargs):
                    self.delta_x = kwargs.get('delta_x')
                    self.Lx = kwargs.get('Lx')
                    self.Nx = int(self.Lx/self.delta_x) + 1
                    self.x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)
                elif ('Lx' in kwargs) and ('Nx' in kwargs):
                    self.Lx = kwargs.get('Lx')
                    self.Nx = kwargs.get('Nx')
                    self.delta_x = self.Lx/(self.Nx - 1)
                    self.x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)
                if sum([i in kwargs for i in ['delta_y', 'Ny', 'Ly']]) == 1:
                    raise Exception('Must supply two out of three of delta_y, Ny, or Ly.')
                elif ('delta_y' in kwargs) and ('Ny' in kwargs):
                    self.delta_y = kwargs.get('delta_y')
                    self.Ny = kwargs.get('Ny')
                    self.Ly = (self.Ny - 1) * self.delta_y
                    self.y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)
                elif ('delta_y' in kwargs) and ('Ly' in kwargs):
                    self.delta_y = kwargs.get('delta_y')
                    self.Ly = kwargs.get('Ly')
                    self.Ny = int(self.Ly/self.delta_y + 1)
                    self.y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)
                elif ('Ly' in kwargs) and ('Ny' in kwargs):
                    self.Ly = kwargs.get('Ly')
                    self.Ny = kwargs.get('Ny')
                    self.delta_y = self.Ly/(self.Ny - 1)
                    self.y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)
                if ((self.x is not None) and (self.y is None)):
                    self.y = np.zeros_like(self.x)
                    self.z = np.zeros_like(self.x)
                elif ((self.y is not None) and (self.x is None)):
                    self.x = np.zeros_like(self.y)
                    self.z = np.zeros_like(self.y)
                else:
                    self.x, self.y = np.meshgrid(self.x, self.y, indexing='ij')
                    self.x = self.x.flatten()
                    self.y = self.y.flatten()
                    self.z = np.zeros_like(self.x)
            else:
                self.x = np.array([0])
                self.y = np.array([0])
                self.z = np.array([0])
            self.r = np.stack((self.x, self.y, self.z), axis=1)
        elif self.r is not None:
            self.x = self.r[:,0]
            self.y = self.r[:,1]
            self.z = self.r[:,2]
        self.N = self.r.shape[0]

    def deform(self, mode='cylindrical', **kwargs):
        if mode == 'cylindrical':
            axis = kwargs.get('axis', 'x')
            R_cylinder = kwargs.get('R_cylinder', 10*C/self.f)