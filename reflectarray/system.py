import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections  as mc
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

        self.reflectarray.R12_tilda = (R12 + R23*np.exp(-2*1j*n2*k*(d2 - d1)))/(1 + R12*R23*np.exp(-2*1j*n2*k*(d2 - d1)))

        self.H_t = (1 - self.reflectarray.R12_tilda) * self.H_feed

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

    def map_polarizabilities(self, mapping='ideal', normalize=True, scale=1):
        self.alpha_library = np.copy(self.reflectarray.element.alpha)
        self.alpha_library = (1 - self.reflectarray.R12_tilda) * self.alpha_library        # accounts for image dipole
        if self.reflectarray.element.element_type == 'patch':
            self.alpha_library = 2 * self.alpha_library                       # accounts for patch array factor in the limit of an infinitesimal patch
        if normalize:
            self.alpha_library = self.alpha_library / np.max(np.abs(self.alpha_library)) * np.max(np.abs(self.alpha_desired)) * scale
        
        if mapping == 'ideal':
            self.alpha = self.alpha_desired
        if mapping == 'euclidean':
            self.alpha_constrained_index = np.empty(self.alpha_desired.shape, dtype=int)
            alpha_constrained = np.empty(self.alpha_desired.shape, dtype=np.complex64)
            for i in range(self.reflectarray.element.lattice_vectors.shape[0]):
                self.alpha_constrained_index[i,:] = np.argmin(np.sqrt(np.abs(np.real(self.alpha_desired[i,:,None]) - np.real(self.alpha_library[None,:]))**2 +
                                            np.abs(np.imag(self.alpha_desired[i,:,None]) - np.imag(self.alpha_library[None,:]))**2), axis=1)
                alpha_constrained[i,:] = self.reflectarray.element.alpha[self.alpha_constrained_index[i,:]]

            self.alpha = alpha_constrained
        
        if mapping == 'phase':
            self.alpha_constrained_index = np.empty(self.alpha_desired.shape, dtype=int)
            alpha_constrained = np.empty(self.alpha_desired.shape, dtype=np.complex64)
            for i in range(self.reflectarray.element.lattice_vectors.shape[0]):
                self.alpha_constrained_index[i,:] = np.argmin(np.sqrt(np.abs(np.angle(self.alpha_desired[i,:,None]) - np.angle(self.alpha_library[None,:]))**2), axis=1)
                alpha_constrained[i,:] = self.reflectarray.element.alpha[self.alpha_constrained_index[i,:]]

            self.alpha = alpha_constrained
    
    def design(self, theta_beam, phi_beam, H_polarization=[0,1,0], R_far=10, mapping='ideal', normalize=True, scale=1):
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
        self.map_polarizabilities(mapping=mapping, normalize=normalize, scale=scale)

    def propagate(self, delta_theta, delta_phi, method='integration'):
        delta_x = self.reflectarray.x[1] - self.reflectarray.x[0]
        delta_y = self.reflectarray.y[1] - self.reflectarray.y[0]

        self.reflectarray.J_e = np.cross(self.reflectarray.n_hat, self.H_t, axisa=1, axisb=1, axisc=1)
        alpha_tensor = np.zeros((self.alpha.shape[1], 3, 3), dtype=np.complex64)
        for i in range(self.alpha.shape[0]):
            alpha_tensor += self.alpha[i,:,None,None] * (self.reflectarray.lattice_vectors[i,:,:,None] @ self.reflectarray.lattice_vectors[i,:,None,:])
        self.reflectarray.J_m = (1j*2*np.pi*self.f*MU_0/(delta_x*delta_y) * (alpha_tensor @ self.H_t[:,:,None]))[:,:,0]
        self.compute.far_field_propagate(self.reflectarray, delta_theta, delta_phi, 2*np.pi*self.f/C, method=method)

    def calculate_beam_metrics(self, **kwargs):
        self.compute.calculate_beam_metrics(**kwargs)

    def plot_mapping(self, ax=None, lattice_index=0, scale=1):

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5,5))
        
        x1 = np.real(self.alpha_desired[lattice_index,:])
        y1 = np.imag(self.alpha_desired[lattice_index,:])
        x2 = np.real(self.alpha_library[self.alpha_constrained_index[lattice_index,:]])
        y2 = np.imag(self.alpha_library[self.alpha_constrained_index[lattice_index,:]])
        
        ax.scatter(x1, y1, label='Ideal')
        ax.scatter(x2, y2, label='Constrained')
        lines = np.transpose(np.array([(x1, y1), (x2, y2)]), (2,0,1))
        lc = mc.LineCollection(lines, linewidths=.3, colors='black')
        ax.add_collection(lc)
        ax.set_xlim(scale*np.amin(np.real(self.alpha_desired[lattice_index,:])), scale*np.amax(np.real(self.alpha_desired[lattice_index,:])))
        ax.set_ylim(scale*np.amin(np.imag(self.alpha_desired[lattice_index,:])), scale*np.amax(np.imag(self.alpha_desired[lattice_index,:])))
        ax.set_xlabel(r'Re{$\alpha$}')
        ax.set_ylabel(r'Im{$\alpha$}')
        plt.legend(frameon=False, loc='lower right')

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
