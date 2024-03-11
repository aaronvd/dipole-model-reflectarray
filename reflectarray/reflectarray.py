import numpy as np
from matplotlib import pyplot as plt
import scipy.constants
from reflectarray import toolbox as tb
from reflectarray import transformations
tb.set_font(fontsize=15)

C = scipy.constants.c
EPS_0 = scipy.constants.epsilon_0
MU_0 = scipy.constants.mu_0

mm = 1E-3

class Reflectarray:
    '''
    Defines a physical, fixed reflectarray antenna in terms of dipole positions, dipole complex magnitudes, and frequencies.
    Handles various methods of supplying dipole positions, ultimately converted to (# positions) x 3 array. Coordinates must be defined on a grid.
    Operating frequency corresponds to element operating frequency.

    args:
        element: Element instance defining the reflectarray element.
        eps_r: relative permittivity of the substrate (default: 1)
        d: substrate thickness (default: 0.762 mm)
    kwargs:
        x: x-coordinates of dipole positions (default: None)
        y: y-coordinates of dipole positions (default: None)
        delta_x: x-spacing of dipole positions (default: None)
        Nx: number of x-dipole positions (default: None)
        Lx: length of x-dipole positions (default: None)
        delta_y: y-spacing of dipole positions (default: None)
        Ny: number of y-dipole positions (default: None)
        Ly: length of y-dipole positions (default: None)
    '''

    def __init__(self, element, eps_r=1, d=0.762*mm, **kwargs):
        
        self.quiet = kwargs.get('quiet', False)

        self.n_hat = np.transpose(np.array([0, 0, 1])[:,None])      # Array is in xy plane, so normal is z
        self.element = element
        self.lattice_vectors = np.copy(self.element.lattice_vectors)[:,None,:]
        self.f = self.element.f
        self.eps_r = eps_r
        self.tan_delta = kwargs.get('tan_delta', 0)
        self.d = d

        self.x = kwargs.get('x', None)
        self.y = kwargs.get('y', None)
        self.z = 0
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
        
        if self.x is None:
            self.x = 0
        if self.y is None:
            self.y = 0
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        Z = np.zeros(X.shape)
        
        self.r = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
        self.N = self.r.shape[0]

        self.deformed = False

    def deform(self, mode='cylindrical', **kwargs):
        '''
        Deforms the reflectarray surface according to the specified mode (currently only cylindrical).

        args:
            mode: deformation mode (default: 'cylindrical')
        kwargs:
            axis: axis of cylindrical deformation (default: 'x')
            R_cylinder: radius of cylindrical deformation (default: 10*C/self.f)
        '''
        self.deformed = True
        self.deform_mode = mode
        
        x = np.copy(self.x)
        y = np.copy(self.y)
        Lx = x.max() - x.min()
        Ly = y.max() - y.min()
        delta_x = x[1] - x[0]
        delta_y = y[1] - y[0]

        if mode == 'cylindrical':            
            self.deform_axis = kwargs.get('axis', 'x')
            self.R_cylinder = kwargs.get('R_cylinder', 10*C/self.f)

            if self.deform_axis == 'x':
                delta_psi = delta_y / self.R_cylinder
                L_psi = Ly / self.R_cylinder
                self.psi = np.arange(-L_psi/2, L_psi/2+delta_psi, delta_psi)
                X, Psi = np.meshgrid(x, self.psi, indexing='ij')
                Y = self.R_cylinder * np.sin(Psi)
                Z = self.R_cylinder * np.cos(Psi)
                Psi = -Psi                          # not sure why this is necessary only for 'x'

            elif self.deform_axis == 'y':
                delta_psi = delta_x / self.R_cylinder
                L_psi = Lx / self.R_cylinder
                self.psi = np.arange(-L_psi/2, L_psi/2+delta_psi, delta_psi)
                Psi, Y = np.meshgrid(self.psi, y, indexing='ij')
                X = self.R_cylinder * np.sin(Psi)
                Z = self.R_cylinder * np.cos(Psi)

            self.n_hat = transformations.rotate_vector(self.n_hat, np.degrees(Psi.flatten()), self.deform_axis)
            lattice_vectors_rotated = []
            for i in range(self.lattice_vectors.shape[0]):
                lattice_vectors_rotated.append(transformations.rotate_vector(self.lattice_vectors[i,:,:], np.degrees(Psi.flatten()), self.deform_axis))
            self.lattice_vectors = np.stack(lattice_vectors_rotated, axis=0)

            Z = Z - Z.max()
            self.r = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)

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
                