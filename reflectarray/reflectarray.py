import numpy as np
from matplotlib import pyplot as plt
import scipy.constants
from reflectarray import toolbox as tb
from reflectarray import transformations
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

        self.n_hat = np.transpose(np.array([0, 0, 1])[:,None])      # Array is in xy plane, so normal is z

        self.element = element
        self.lattice_vectors = np.copy(self.element.lattice_vectors)[:,None,:]
        self.f = self.element.f

        self.r = kwargs.get('r', None)
        self.x = kwargs.get('x', None)
        self.y = kwargs.get('y', None)
        self.quiet = kwargs.get('quiet', False)
        if self.r is None:
            if (any([i in kwargs for i in ['delta_x', 'Nx', 'Lx']])) or (any([i in kwargs for i in ['delta_y', 'Ny', 'Ly']])):
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
            elif (self.x is not None) and (self.y is not None):
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

        self.deformed = False

    def deform(self, mode='cylindrical', **kwargs):
        self.deformed = True
        self.deform_mode = mode
        
        x = np.unique(self.x)
        y = np.unique(self.y)
        Lx = x.max() - x.min()
        Ly = y.max() - y.min()
        delta_x = x[1] - x[0]
        delta_y = y[1] - y[0]

        if mode == 'cylindrical':            
            self.deform_axis = kwargs.get('axis', 'x')
            R_cylinder = kwargs.get('R_cylinder', 10*C/self.f)

            if self.deform_axis == 'x':
                delta_psi = delta_y / R_cylinder
                L_psi = Ly / R_cylinder
                psi = np.arange(-L_psi/2, L_psi/2, delta_psi)
                self.x, Psi = np.meshgrid(x, psi, indexing='ij')
                self.y = R_cylinder * np.sin(Psi)
                self.z = R_cylinder * np.cos(Psi)
                Psi = -Psi                          # not sure why this is necessary only for 'x'

            elif self.deform_axis == 'y':
                delta_psi = delta_x / R_cylinder
                L_psi = Lx / R_cylinder
                psi = np.arange(-L_psi/2, L_psi/2, delta_psi)
                Psi, self.y = np.meshgrid(psi, y, indexing='ij')
                self.x = R_cylinder * np.sin(Psi)
                self.z = R_cylinder * np.cos(Psi)

            self.n_hat = transformations.rotate_vector(self.n_hat, np.degrees(Psi.flatten()), self.deform_axis)
            lattice_vectors_rotated = []
            for i in range(self.lattice_vectors.shape[0]):
                lattice_vectors_rotated.append(transformations.rotate_vector(self.lattice_vectors[i,:,:], np.degrees(Psi.flatten()), self.deform_axis))
            self.lattice_vectors = np.stack(lattice_vectors_rotated, axis=0)

            self.z = self.z - self.z.max()
            self.r = np.stack((self.x.flatten(), self.y.flatten(), self.z.flatten()), axis=1)

    def plot(self, ax=None, plot_type=None, **kwargs):
        L_ap = np.maximum(self.x.max() - self.x.min(), self.y.max() - self.y.min())
        if plot_type is None:
            plot_type = '2D'
            if self.deformed:
                plot_type = '3D'

        if ax is None:
            fig = plt.figure()
            if plot_type=='2D':
                ax = fig.add_subplot()
            elif plot_type=='3D':
                ax = fig.add_subplot(projection='3d')

        show_lattice_vectors = kwargs.get('show_lattice_vectors', False)

        if plot_type == '2D':
            ax.scatter(self.r[:,0], self.r[:,1], marker='o', facecolors='none', color='tab:blue')
            if show_lattice_vectors:
                for i in range(self.element.lattice_vectors.shape[0]):
                    ax.quiver(self.r[:,0].flatten(), self.r[:,1].flatten(),
                              self.lattice_vectors[i,:,0], self.lattice_vectors[i,:,1],
                              scale=10, color='tab:red', pivot='middle')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_title('Reflectarray Dipole Positions')
        
        elif plot_type == '3D':
            show_normals = kwargs.get('show_normals', False)
            ax.scatter(self.r[:,0], self.r[:,1], self.r[:,2], marker='o', facecolors='none', color='tab:blue', label='Patch Positions')
            if show_normals:
                ax.quiver(self.r[:,0].flatten(), self.r[:,1].flatten(), self.r[:,2].flatten(),
                        self.n_hat[:,0], self.n_hat[:,1], self.n_hat[:,2], length=0.03, color='k', label='Surface Normals')
            if show_lattice_vectors:
                for i in range(self.element.lattice_vectors.shape[0]):
                    ax.quiver(self.r[:,0].flatten(), self.r[:,1].flatten(), self.r[:,2].flatten(),
                              self.lattice_vectors[i,:,0], self.lattice_vectors[i,:,1], self.lattice_vectors[i,:,2],
                              length=0.03, color='tab:red', pivot='middle')
            if kwargs.get('legend', True):
                ax.legend(frameon=False)
            ax.set_xlabel('$x$ (m)')
            ax.set_ylabel('$y$ (m)')
            ax.set_zlabel('$z$ (m)')
            ax.set_xlim(self.x.min(), self.x.max())
            ax.set_ylim(self.y.min(), self.y.max())
            ax.set_zlim(np.mean(self.r[:,2])-L_ap/2, np.mean(self.r[:,2])+L_ap/2)
        ax.set_aspect('equal')
        plt.tight_layout()
                