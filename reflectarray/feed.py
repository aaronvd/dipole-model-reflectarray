import numpy as np
from matplotlib import pyplot as plt
import scipy.constants
import scipy.io
from reflectarray import toolbox as tb
from reflectarray import transformations
tb.set_font(fontsize=15)

C = scipy.constants.c
EPS_0 = scipy.constants.epsilon_0
MU_0 = scipy.constants.mu_0

class Feed:
    '''
    Base class for reflectarray feed. Defaults to point dipole. 
    Supply (N x 3) array of positions and corresponding electric and/or magnetic currents as (N x 3) arrays.
    Approach: define 2D feed antenna centered at origin, and provide offset and rotation vectors.
    '''

    def __init__(self, f=None, **kwargs):

        self.quiet = kwargs.get('quiet', False)
        self.f = f
        if self.f is None:
            if not self.quiet:
                print('No frequency vector provided, defaulting to 10 GHz')
            self.f = 10E9

        self.r_offset = np.array(kwargs.get('r_offset', (0, 0, 10*C/self.f)))
        self.rotation = np.array(kwargs.get('rotation', (0, 0, 0)))

        self.make(**kwargs)
        self.transform()
        
        ### CALCULATE COEFFICIENT ACCORDING TO GAIN
            
    def make(self, **kwargs):
        self.r_origin = kwargs.get('r', None)
        if self.r_origin is None:
            self.r_origin = np.array([[0, 0, 0]])

        self.J_e_origin = kwargs.get('J_e', None)
        self.J_m_origin = kwargs.get('J_m', None)
        if (self.J_e_origin is None) and (self.J_m_origin is None):
            self.J_e_origin = np.tile(np.array([[1, 0, 0]]).astype(np.complex128), (self.r_origin.shape[0], 1))

    def transform(self):
        self.r = transformations.rotate_vector(self.r_origin, 180, 'y')         ### flip feed so that it's facing in -z direction
        self.r = np.copy(self.r_origin)
        self.r = transformations.rotate_vector(self.r, self.rotation[0], 'x')
        self.r = transformations.rotate_vector(self.r, self.rotation[1], 'y')
        self.r = transformations.rotate_vector(self.r, self.rotation[2], 'z')
        self.r += self.r_offset[None,:]

        if self.J_e_origin is not None:
            self.J_e = transformations.rotate_vector(self.J_e_origin, 180, 'y')         ### flip feed electric currents
            self.J_e = transformations.rotate_vector(self.J_e, self.rotation[0], 'x')
            self.J_e = transformations.rotate_vector(self.J_e, self.rotation[1], 'y')
            self.J_e = transformations.rotate_vector(self.J_e, self.rotation[2], 'z')
        else:
            self.J_e = None
        
        if self.J_m_origin is not None:
            self.J_m = transformations.rotate_vector(self.J_m_origin, 180, 'y')         ### flip feed magnetic currents
            self.J_m = transformations.rotate_vector(self.J_m, self.rotation[0], 'x')
            self.J_m = transformations.rotate_vector(self.J_m, self.rotation[1], 'y')
            self.J_m = transformations.rotate_vector(self.J_m, self.rotation[2], 'z')
        else:
            self.J_m = None
        
    def plot(self, ax=None, plot_type='2D', **kwargs):
        L_ap = np.maximum(self.r[:,0].max() - self.r[:,0].min(), self.r[:,1].max() - self.r[:,1].min())
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
        component = kwargs.get('component', 'x')
        quiver = kwargs.get('quiver', False)

        plot_obj_origin = plot_dict_origin[plot_value]
        plot_obj = plot_dict[plot_value]
        component_index = component_dict[component]

        if plot_type == '2D':
            
            ax.scatter(self.r_origin[:,0], self.r_origin[:,1], marker='o', facecolors='none', c=np.real(plot_obj)[:,component_index], label='Feed Positions')
            if quiver:    
                ax.quiver(self.r_origin[:,0].flatten(), self.r_origin[:,1].flatten(),
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

        