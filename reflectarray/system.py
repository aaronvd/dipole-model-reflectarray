import numpy as np
from matplotlib import pyplot as plt
import scipy.constants
from reflectarray import toolbox as tb
from reflectarray.compute import Compute
from reflectarray import transformations
tb.set_font(fontsize=15)

C = scipy.constants.c
EPS_0 = scipy.constants.epsilon_0
MU_0 = scipy.constants.mu_0
ETA_0 = np.sqrt(MU_0/EPS_0)

mm = 1E-3

class System:
    '''
    Establishes reflectarray system consisting of feed and reflectarray.
    '''

    def __init__(self, reflectarray, feed, **kwargs):
        self.quiet = kwargs.get('quiet', False)

        self.compute = Compute(quiet=self.quiet)

        self.reflectarray = reflectarray
        self.feed = feed
        self.f = self.reflectarray.f
        if self.f != self.feed.f:
            if not self.quiet:
                print('Warning: reflectarray and feed frequencies do not match')

    def compute_feed_fields(self):
        self.H_feed = self.compute.near_field_propagate(self.feed, self.reflectarray.r, 2*np.pi*self.f/C, field='H')

    def compute_total_fields(self):
        k = 2*np.pi*self.f/C
        
        # Layer 1 - Free Space
        eps_1 = EPS_0
        n1 = 1
        d1 = 0
        eta_1 = np.sqrt(MU_0/eps_1)

        # Layer 2 - Dielectric
        eps_r_2 = self.reflectarray.eps_r
        tan_delta_2 = self.reflectarray.tan_delta
        d2 = self.reflectarray.d
        eps_2 = EPS_0 * eps_r_2 * (1 - 1j*tan_delta_2)
        n2 = np.sqrt(eps_2/EPS_0)
        eta_2 = np.sqrt(MU_0/eps_2)

        # Layer 3 - Copper
        sigma = 6E7
        eps_3 = -1j*sigma/(2*np.pi*self.f)
        n3 = np.sqrt(eps_3/EPS_0)
        eta_3 = np.sqrt(MU_0/eps_3)

        R12 = (eta_2 - eta_1)/(eta_2 + eta_1)
        R23 = (eta_3 - eta_2)/(eta_3 + eta_2)

        R12_tilda = (R12 + R23*np.exp(-2*1j*n2*k*(d2 - d1)))/(1 + R12*R23*np.exp(-2*1j*n2*k*(d2 - d1)))

        self.H_t = (1 - R12_tilda) * self.H_feed

    def compute_polarizabilities(self, theta_beam, phi_beam, H_polarization=[0,1,0], R_far=10):
        delta_x = self.reflectarray.x[1] - self.reflectarray.x[0]
        delta_y = self.reflectarray.y[1] - self.reflectarray.y[0]
        
        k = 2*np.pi*self.f/C
        k_beam_spherical = np.transpose(np.array([k, 0, 0])[:,None])
        k_beam_cartesian = transformations.spherical_to_cartesian_vector(k_beam_spherical,
                                                                         theta_beam,
                                                                         phi_beam)
        k_hat = k_beam_cartesian / k

        H_beam = np.transpose(np.array(H_polarization)[:,None])

        A = (1j*4*np.pi*R_far)/(2*np.pi*self.f*EPS_0*np.exp(-1j*k*R_far))
        exp_term = np.exp(-1j*np.sum(k_beam_cartesian * self.reflectarray.r, 1))
        
        self.alpha_desired = np.empty((self.reflectarray.element.lattice_vectors.shape[0], self.reflectarray.r.shape[0]), dtype=np.complex64)
        for i in range(self.reflectarray.element.lattice_vectors.shape[0]):
            term1 = A * np.sum(H_beam * self.reflectarray.element.lattice_vectors[i,None,:], 1) * exp_term
            term2 = -ETA_0 * (np.sum(k_hat * self.H_t, 1) * np.sum(self.reflectarray.element.lattice_vectors[i,None,:] * self.reflectarray.n_hat, 1) - np.sum(k_hat * self.reflectarray.n_hat, 1) * np.sum(self.reflectarray.element.lattice_vectors[i,None,:] * self.H_t, 1))
            coeff = -1j*delta_x*delta_y/(2*np.pi*self.f*MU_0) * (1/np.sum(self.H_t * self.reflectarray.element.lattice_vectors[i,None,:], 1))
            self.alpha_desired[i,:] = coeff * (term1 + term2)

    def map_polarizabilities(self, mapping='ideal'):
        if mapping == 'ideal':
            self.alpha = self.alpha_desired
    
    def design(self, theta_beam, phi_beam, H_polarization=[0,1,0], R_far=10, mapping='ideal'):
        if not self.quiet:
            print('Computing feed fields...')
        self.compute_feed_fields()
        self.compute_total_fields()
        
        self.theta_beam = theta_beam
        self.phi_beam = phi_beam
        theta_beam = np.radians(theta_beam)
        phi_beam = np.radians(phi_beam)

        self.H_polarization = H_polarization
        if not self.quiet:
            print('Computing ideal polarizabilities...')
        self.compute_polarizabilities(theta_beam, phi_beam, H_polarization, R_far)
        
        if not self.quiet:
            print('Computing constrained polarizabilities...')
        self.map_polarizabilities(mapping=mapping)

    def propagate(self, delta_theta, delta_phi, method='integration'):
        delta_x = self.reflectarray.x[1] - self.reflectarray.x[0]
        delta_y = self.reflectarray.y[1] - self.reflectarray.y[0]

        self.reflectarray.J_e = np.cross(self.reflectarray.n_hat, self.H_t, axisa=1, axisb=1, axisc=1)
        alpha_tensor = np.zeros((self.alpha.shape[1], 3, 3), dtype=np.complex64)
        for i in range(self.alpha.shape[0]):
            alpha_tensor += self.alpha[i,:,None,None] * (self.reflectarray.lattice_vectors[i,:,:,None] @ self.reflectarray.lattice_vectors[i,:,None,:])
        self.reflectarray.J_m = (1j*2*np.pi*self.f*MU_0/(delta_x*delta_y) * (alpha_tensor @ self.H_t[:,:,None]))[:,:,0]
        self.compute.far_field_propagate(self.reflectarray, delta_theta, delta_phi, 2*np.pi*self.f/C, method=method)
    
    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

        self.reflectarray.plot(ax=ax, plot_type='3D')
        self.feed.plot(ax=ax, plot_type='3D')

        xmin = np.minimum(self.reflectarray.r[:,0].min(), self.feed.r[:,0].min())
        rmin = np.minimum(np.min(self.reflectarray.r, axis=0), np.min(self.feed.r, axis=0))
        rmax = np.maximum(np.max(self.reflectarray.r, axis=0), np.max(self.feed.r, axis=0))
        buffer = np.maximum(0.1*np.max(rmax - rmin), 0.1)
        ax.set_xlim(rmin[0] - buffer, rmax[0] + buffer)
        ax.set_ylim(rmin[1] - buffer, rmax[1] + buffer)
        ax.set_zlim(rmin[2] - buffer, rmax[2] + buffer)
