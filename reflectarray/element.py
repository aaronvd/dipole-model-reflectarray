import numpy as np
from matplotlib import pyplot as plt
import scipy.constants
import scipy.io
from reflectarray import toolbox as tb
tb.set_font(fontsize=15)

C = scipy.constants.c
EPS_0 = scipy.constants.epsilon_0
MU_0 = scipy.constants.mu_0

class Element:
    '''
    Defines element for reflectarray construction.

    args:
        lattice_vectors: 1x3 or 2x3 array of lattice vectors (default: np.array([[0,1,0]]))
    '''

    def __init__(self, lattice_vectors=None, f=None, alpha=None, filepath=None, f0=None, normalize=False, **kwargs):
        self.quiet = kwargs.get('quiet', False)

        self.lattice_vectors = lattice_vectors
        if self.lattice_vectors is None:
            self.lattice_vectors = np.array([[0,1,0]])
        
        self.f = f
        if self.f is None:
            if not self.quiet:
                if not self.quiet:
                    print('No frequency vector provided, defaulting to 10 GHz')
            self.f = 10E9
        
        if alpha is not None:
            self.alpha = alpha
        elif filepath is not None:
            filetype = filepath.split('.')[-1]
            if filetype == 'txt':
                self.alpha = np.loadtxt(filepath, delimiter=',')
            elif filetype == 'mat':
                self.alpha = scipy.io.loadmat(filepath)['alpha']
            elif filetype == 'pkl':
                self.alpha = tb.pickle_load(filepath)['alpha']
        else:
            if f0 is None:
                self.f0 = np.linspace(self.f - 0.1*self.f, 
                                      self.f + 0.1*self.f, 
                                      101)
            self.F = kwargs.get('F', 1)
            self.Q = kwargs.get('Q', 20)
            omega = 2*np.pi * self.f
            omega_0 = 2*np.pi * self.f0
            Gamma = omega / self.Q
            self.alpha = (self.F * omega**2) / (omega_0**2 - omega**2 + 1j*Gamma*omega)
        if normalize:
            self.alpha = self.alpha / np.max(np.abs(self.alpha))

    def plot(self, plot_dict=None):
        if plot_dict is None:
            plot_dict = ['magnitude', 'phase', 'complex']
        _, axes = plt.subplots(1, len(plot_dict), figsize=(len(plot_dict)*5, 5))
        if len(plot_dict) == 1:
            axes = [axes]
        for i, p in enumerate(plot_dict):
            if p == 'magnitude':
                axes[i].plot(np.abs(self.alpha))
                axes[i].set_title('Magnitude')
            elif p == 'phase':
                axes[i].plot(np.angle(self.alpha))
                axes[i].set_title('Phase')
            elif p == 'complex':
                axes[i].scatter(np.real(self.alpha), np.imag(self.alpha))
                axes[i].set_title('Complex')
            elif p == 'real':
                axes[i].plot(np.real(self.alpha))
                axes[i].set_title('Real')
            elif p == 'imag':
                axes[i].plot(np.imag(self.alpha))
                axes[i].set_title('Imaginary')
            plt.tight_layout()
            
class Patch(Element):
    '''
    Defines a ground-plane-backed patch element with magnetic dipole spacing W.
    '''
    def __init__(self, lattice_vectors=None, alpha=None, filepath=None, f0=None, **kwargs):
        
        super().__init__(lattice_vectors=lattice_vectors, alpha=alpha, filepath=filepath, f0=f0, **kwargs)
        
        self.element_type = 'patch'
        self.W = kwargs.get('W', C / (2*self.f))
