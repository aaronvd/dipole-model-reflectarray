import numpy as np
from matplotlib import pyplot as plt
import scipy.constants
from copy import deepcopy
from reflectarray import toolbox as tb
from reflectarray import transformations
from reflectarray.compute import Compute
tb.set_font(fontsize=15)

C = scipy.constants.c
EPS_0 = scipy.constants.epsilon_0
MU_0 = scipy.constants.mu_0
ETA_0 = np.sqrt(MU_0/EPS_0)

mm = 1E-3

class Feed:
    '''
    Base class for reflectarray feed. Defaults to point dipole. 
    Accepts various keyword combinations for defining feed positions.
    Supply corresponding electric and/or magnetic currents as (N x 3) arrays.
    Approach: define 2D feed antenna centered at origin, and provide offset and rotation vectors.
    '''

    def __init__(self, f=None, **kwargs):

        self.quiet = kwargs.get('quiet', False)
        self.compute = Compute(quiet=self.quiet)
        self.f = f
        if self.f is None:
            if not self.quiet:
                print('No frequency vector provided, defaulting to 10 GHz')
            self.f = 10E9

        self.r_offset = np.array(kwargs.get('r_offset', (0, 0, 10*C/self.f)))
        self.rotation = np.array(kwargs.get('rotation', (0, 0, 0)))

        self.make(**kwargs)
        self.transform()
        
    def make(self, **kwargs):
        self.x = kwargs.get('x', None)
        self.y = kwargs.get('y', None)
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

        self.J_e_origin = kwargs.get('J_e', None)
        self.J_m_origin = kwargs.get('J_m', None)
        if (self.J_e_origin is None) and (self.J_m_origin is None):
            self.J_e_origin = np.tile(np.array([[1, 0, 0]]).astype(np.complex128), (self.r_origin.shape[0], 1))
        self.J_e = deepcopy(self.J_e_origin)
        self.J_m = deepcopy(self.J_m_origin)

    def transform(self):
        self.r = transformations.rotate_vector(self.r, 180, 'y')         ### flip feed so that it's facing in -z direction
        self.r = transformations.rotate_vector(self.r, self.rotation[0], 'x')
        self.r = transformations.rotate_vector(self.r, self.rotation[1], 'y')
        self.r = transformations.rotate_vector(self.r, self.rotation[2], 'z')
        self.r += self.r_offset[None,:]

        if self.J_e is not None:
            self.J_e = transformations.rotate_vector(self.J_e, 180, 'y')         ### flip feed electric currents
            self.J_e = transformations.rotate_vector(self.J_e, self.rotation[0], 'x')
            self.J_e = transformations.rotate_vector(self.J_e, self.rotation[1], 'y')
            self.J_e = transformations.rotate_vector(self.J_e, self.rotation[2], 'z')
        
        if self.J_m is not None:
            self.J_m = transformations.rotate_vector(self.J_m, 180, 'y')         ### flip feed magnetic currents
            self.J_m = transformations.rotate_vector(self.J_m, self.rotation[0], 'x')
            self.J_m = transformations.rotate_vector(self.J_m, self.rotation[1], 'y')
            self.J_m = transformations.rotate_vector(self.J_m, self.rotation[2], 'z')

    def far_field_propagate(self, delta_theta, delta_phi, method='integration'):
        self.compute.far_field_propagate(self, delta_theta, delta_phi, 2*np.pi*self.f/C, method=method)

    def plot_fields(self, ax=None, plot_type='2D', **kwargs):
        E_int = np.sum(np.abs(self.compute.E_ff)**2, axis=1)
        E_plot = np.reshape(E_int, (self.compute.theta.size, self.compute.phi.size))

        dB_min = kwargs.get('dB_min', -20)
        dB_max = kwargs.get('dB_max', 0)

        if plot_type is None:
            plot_type = '2D'
        if ax is None:
            if plot_type=='1D':
                fig = plt.figure(figsize=(8,8))
                ax = plt.subplot(111, projection='polar')
            if plot_type=='2D':
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(7,7))

        if plot_type=='1D':
            phi_beam = kwargs.get('phi_slice', 0)
            phi_beam = np.radians(phi_beam)
            phi_indx1 = np.argmin(np.abs(self.compute.phi - phi_beam))
            phi_indx2 = np.argmin(np.abs(self.compute.phi - (phi_beam+np.pi)))
            E_plot_1D = np.concatenate((np.flip(E_plot[1:,phi_indx2][:,None]), E_plot[:,phi_indx1][:,None]))
            ax.plot(np.linspace(-np.pi/2, np.pi/2, E_plot_1D.size), 10*np.log10(E_plot_1D/np.amax(E_plot_1D)))
            ax.set_thetalim(-np.pi/2, np.pi/2)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylabel('dB', rotation=0)
            ax.set_ylim(dB_min, dB_max)
            ax.yaxis.set_label_coords(0.17, 0.15)
            ax.set_xlabel(r'$\theta$')
            ax.xaxis.set_label_coords(0.5, 0.86)
            ax.set_yticks(np.arange(dB_min, dB_max+10, 10))
        elif plot_type=='2D':
            cs = ax.contourf(self.compute.phi, self.compute.theta*180/np.pi, 10*np.log10(E_plot/np.amax(E_plot)), 
                        np.linspace(dB_min, dB_max, 100), 
                        cmap=plt.cm.hot_r)
            ax.grid(True)
            ax.set_rlabel_position(135)
            fig.colorbar(cs, ticks=np.linspace(dB_min, dB_max, 7))
            ax.set_xlabel('$\phi$')
        
    def plot(self, ax=None, plot_type='2D', **kwargs):
        L_ap = np.maximum(self.x.max() - self.x.min(), self.y.max() - self.y.min())
        buffer = np.maximum(0.1*L_ap, 0.1)
        if plot_type is None:
            plot_type = '2D'

        if ax is None:
            fig = plt.figure()
            if plot_type=='2D':
                ax = fig.add_subplot()
            elif plot_type=='3D':
                ax = fig.add_subplot(projection='3d')

        plot_dict_origin = {'J_e': np.real(self.J_e_origin), 'J_m': np.real(self.J_m_origin)}
        plot_dict = {'J_e': np.real(self.J_e), 'J_m': np.real(self.J_m)}
        component_dict = {'x': 0, 'y': 1, 'z': 2}
        plot_value = kwargs.get('plot_value', 'J_e')
        if (self.J_e is None) and (plot_value == 'J_e'):
            plot_value = 'J_m'
        component = kwargs.get('component', 'x')
        quiver = kwargs.get('quiver', False)

        plot_obj_origin = plot_dict_origin[plot_value]
        plot_obj = plot_dict[plot_value]
        component_index = component_dict[component]

        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        Z = np.zeros_like(X)
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()

        if plot_type == '2D':
            
            ax.scatter(X, Y, marker='o', facecolors='none', c=np.real(plot_obj)[:,component_index], label='Feed Positions')
            if quiver:    
                ax.quiver(X.flatten(), Y.flatten(),
                            plot_obj_origin[:,0], plot_obj_origin[:,1],
                            scale=10, color='tab:red', pivot='middle',
                            label='${}$'.format(plot_value))
            if kwargs.get('legend', True):
                ax.legend(frameon=False)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_title('Feed Fields')
        
        elif plot_type == '3D':
            ax.scatter(self.r[:,0], self.r[:,1], self.r[:,2], marker='o', facecolors='none', c=np.real(plot_obj)[:,component_index], label='Feed Positions')
            if quiver:
                ax.quiver(self.r[:,0].flatten(), self.r[:,1].flatten(), self.r[:,2].flatten(),
                    plot_obj[:,0], plot_obj[:,1], plot_obj[:,2], length=0.01, color='tab:red', label='${}$'.format(plot_value), pivot='middle')
            if kwargs.get('legend', True):
                ax.legend(frameon=False)
            ax.set_xlabel('$x$ (m)')
            ax.set_ylabel('$y$ (m)')
            ax.set_zlabel('$z$ (m)')
            ax.set_xlim(self.r[:,0].min()-buffer, self.r[:,0].max()+buffer)
            ax.set_ylim(self.r[:,1].min()-buffer, self.r[:,1].max()+buffer)
            ax.set_zlim(np.mean(self.r[:,2])-L_ap/2-buffer, np.mean(self.r[:,2])+L_ap/2+buffer)
        ax.set_aspect('equal')
        plt.tight_layout()

class PyramidalHorn(Feed):
    '''
    Defines a pyramidal horn, inheriting from the Feed class.
    E-polarization in y direction by default.
    Definition from Balanis, C. A. (2016). Antenna Theory: Analysis and Design. John Wiley & Sons.
    '''
    def __init__(self, f=None, **kwargs):
        self.E0 = kwargs.get('E0', 1)
        self.quiet = kwargs.get('quiet', False)
        self.compute = Compute(quiet=self.quiet)
        self.f = f
        if self.f is None:
            if not self.quiet:
                print('No frequency vector provided, defaulting to 10 GHz')
            self.f = 10E9

        self.r_offset = np.array(kwargs.get('r_offset', (0, 0, 10*C/self.f)))
        self.rotation = np.array(kwargs.get('rotation', (0, 0, 0)))

        self.make(**kwargs)

        self.gain = kwargs.get('gain', None)
        if self.gain is not None:
            gain_linear = 10**(self.gain/10)
            self.far_field_propagate(1, 2, method='integration')
            U_int = 1/(2*ETA_0) * np.sum(np.abs(self.compute.E_ff)**2, axis=1)
            self.E0 = np.sqrt(gain_linear/(4*np.pi*np.amax(U_int)))
            self.make(**kwargs)
            self.far_field_propagate(1, 2, method='integration')

        self.transform()

    def make(self, **kwargs):
        a = kwargs.get('a', 40.13*mm)
        b = kwargs.get('b', 29.2*mm)
        rho_1 = kwargs.get('rho_1', 66.09*mm)
        rho_2 = kwargs.get('rho_2', 100.155*mm)
        delta_x = kwargs.get('delta_x', C/(self.f*10))
        delta_y = kwargs.get('delta_y', C/(self.f*10))
        self.x = np.arange(-a/2, a/2+delta_x, delta_x)
        self.y = np.arange(-b/2, b/2+delta_y, delta_y)
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        Z = np.zeros_like(X)
        self.r = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
        self.N = self.r.shape[0]

        J_ey = -self.E0/ETA_0 * np.cos(np.pi*self.r[:,0]/a) * np.exp(-1j*(2*np.pi*self.f*(self.r[:,0]**2/rho_2 + self.r[:,1]**2/rho_1)/(2*C)))
        J_mx = self.E0 * np.cos(np.pi*self.r[:,0]/a) * np.exp(-1j*(2*np.pi*self.f*(self.r[:,0]**2/rho_2 + self.r[:,1]**2/rho_1)/(2*C)))
        self.J_e_origin = np.stack((np.zeros_like(J_ey), J_ey, np.zeros_like(J_ey)), axis=1)
        self.J_m_origin = np.stack((J_mx, np.zeros_like(J_mx), np.zeros_like(J_mx)), axis=1)
        self.J_e = deepcopy(self.J_e_origin)
        self.J_m = deepcopy(self.J_m_origin)