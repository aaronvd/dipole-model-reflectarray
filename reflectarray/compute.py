import numpy as np
import scipy.constants
from reflectarray import toolbox as tb
from reflectarray import transformations
tb.set_font(fontsize=15)

C = scipy.constants.c
EPS_0 = scipy.constants.epsilon_0
MU_0 = scipy.constants.mu_0
ETA_0 = np.sqrt(MU_0/EPS_0)

mm = 1E-3

### NEED TO GENERALIZE ARRAY FACTOR
### CURRENTLY ONLY VALID FOR PATCH EXCITED BY MAGNETIC FIELD ALONG Y DIRECTION

class Compute:
    '''
    Class for computing near and far field radiation patterns of a reflectarray.
    Additionally handles calculation of various beam metrics.
    '''
    def __init__(self, quiet=False):
        self.quiet = quiet

    def near_field_propagate(self, source, r_target, k, field='H'):

        if source.__module__.split('.')[-1] == 'reflectarray':
            if source.deformed:
                if source.deform_axis == 'x':
                    x_source = source.x
                    y_source = source.R_cylinder * source.psi
                elif source.deform_axis == 'y':
                    x_source = source.R_cylinder * source.psi
                    y_source = source.y
            else:
                x_source = source.x
                y_source = source.y
        else:
            x_source = source.x
            y_source = source.y
        
        if field == 'H':
            H_A = np.zeros(r_target.shape, dtype=np.complex64)
            H_F = np.zeros(r_target.shape, dtype=np.complex64)
            for i in range(r_target.shape[0]):
                R_vec = r_target[i,:][None,:] - source.r
                R_norm = np.linalg.norm(R_vec, axis=1, keepdims=True)
                R_hat = R_vec/R_norm
                
                G1 = -(1 + 1j*k*R_norm - k**2 * R_norm**2)/(R_norm**3)
                G2 = (3 + 3*1j*k*R_norm - k**2 * R_norm**2)/(R_norm**5)
                
                if source.J_e is not None:
                    H_A_integrand = np.reshape((-1/(4*np.pi)) 
                                    * np.cross(R_hat, source.J_e, axisa=1, axisb=1, axisc=1)
                                    * (1 + 1j*k*R_norm)/R_norm**2
                                    * np.exp(-1j*k*R_norm), (x_source.size, y_source.size, 3))
                    H_A[i,:] = tb.trapz(H_A_integrand, [x_source, y_source])
                    # H_A[i,:] = np.trapz(np.trapz(H_A_integrand, x_int, axis=0), y_int, axis=0)
                if source.J_m is not None:
                    H_F_integrand = np.reshape((-1j/(4*np.pi*k*ETA_0))
                                            * (G1 * source.J_m + 
                                                G2 * R_vec * np.sum(R_vec * source.J_m, axis=1, keepdims=True))
                                            * np.exp(-1j*k*R_norm), (x_source.size, y_source.size, 3))
                    H_F[i,:] = tb.trapz(H_F_integrand, [x_source, y_source])
                    # H_F[i,:] = np.trapz(np.trapz(H_F_integrand, x_int, axis=0), y_int, axis=0)
                
            return H_A + H_F

    def far_field_propagate(self, source, delta_theta, delta_phi, k, method='integration'):

        if source.__module__.split('.')[-1] == 'reflectarray':
            if source.deformed:
                if source.deform_axis == 'x':
                    x_source = source.x
                    y_source = source.R_cylinder * source.psi
                elif source.deform_axis == 'y':
                    x_source = source.R_cylinder * source.psi
                    y_source = source.y
            else:
                x_source = source.x
                y_source = source.y

        else:
            x_source = source.x
            y_source = source.y
        
        if method == 'integration':
            
            self.theta = np.radians(np.arange(0, 90+delta_theta, delta_theta))
            self.phi = np.radians(np.arange(0, 360+delta_phi, delta_phi))
            Theta, Phi = np.meshgrid(self.theta, self.phi, indexing='ij')
            Theta = np.reshape(Theta, -1)
            Phi = np.reshape(Phi, -1)

            r_hat = transformations.spherical_to_cartesian_vector(np.transpose(np.array([1, 0, 0])[:,None]),
                                                                  Theta, 
                                                                  Phi)
            k_far_vec = k * r_hat

            if source.J_e is not None:
                J_e_grid = np.reshape(source.J_e, (source.x.size, source.y.size, 3))
                N_theta = np.empty((Theta.size), dtype=np.complex64)
                N_phi = np.empty((Theta.size), dtype=np.complex64)
                for i in range(Theta.size):
                    integrand_theta = ((J_e_grid[:,:,0]*np.cos(Theta[i])*np.cos(Phi[i]) + 
                                    J_e_grid[:,:,1]*np.cos(Theta[i])*np.sin(Phi[i]) -
                                    J_e_grid[:,:,2]*np.sin(Theta[i])) *
                                    np.reshape(np.exp(1j*np.sum(k_far_vec[i] * source.r, 1)), (x_source.size, y_source.size)))
                    integrand_phi = ((-J_e_grid[:,:,0]*np.sin(Phi[i]) + 
                                    J_e_grid[:,:,1]*np.cos(Phi[i])) *
                                    np.reshape(np.exp(1j*np.sum(k_far_vec[i] * source.r, 1)), (x_source.size, y_source.size)))
                    N_theta[i] = tb.trapz(integrand_theta, [x_source, y_source])
                    N_phi[i] = tb.trapz(integrand_phi, [x_source, y_source])
                    # N_theta[i] = np.trapz(np.trapz(integrand_theta, x_int, axis=0), y_int, axis=0)
                    # N_phi[i] = np.trapz(np.trapz(integrand_phi, x_int, axis=0), y_int, axis=0)
            else:
                N_theta = 0
                N_phi = 0
            
            if source.J_m is not None:
                J_m_grid = np.reshape(source.J_m, (source.x.size, y_source.size, 3))
                L_theta = np.empty((Theta.size), dtype=np.complex64)
                L_phi = np.empty((Theta.size), dtype=np.complex64)
                for i in range(Theta.size):
                    integrand_theta = ((J_m_grid[:,:,0]*np.cos(Theta[i])*np.cos(Phi[i]) + 
                                    J_m_grid[:,:,1]*np.cos(Theta[i])*np.sin(Phi[i]) -
                                    J_m_grid[:,:,2]*np.sin(Theta[i])) *
                                    np.reshape(np.exp(1j*np.sum(k_far_vec[i] * source.r, 1)), (x_source.size, y_source.size)))
                    integrand_phi = ((-J_m_grid[:,:,0]*np.sin(Phi[i]) + 
                                    J_m_grid[:,:,1]*np.cos(Phi[i])) *
                                    np.reshape(np.exp(1j*np.sum(k_far_vec[i] * source.r, 1)), (x_source.size, y_source.size)))
                    L_theta[i] = tb.trapz(integrand_theta, [x_source, y_source])
                    L_phi[i] = tb.trapz(integrand_phi, [x_source, y_source])
                    # L_theta[i] = np.trapz(np.trapz(integrand_theta, x_source, axis=0), y_source, axis=0)
                    # L_phi[i] = np.trapz(np.trapz(integrand_phi, x_source, axis=0), y_source, axis=0)
            else:
                L_theta = 0
                L_phi = 0

            self.AF = 1
            if source.__module__.split('.')[-1] == 'reflectarray':
                if source.element.element_type == 'patch':
                    self.AF = (1 - source.R12_tilda) * 2 * np.cos(k*source.element.W/2 * np.sin(Theta)*np.cos(Phi))

            R_far = 1

            E_theta = -(1j*k*np.exp(-1j*k*R_far))/(4*np.pi*R_far) * (self.AF * L_phi + ETA_0 * N_theta)
            E_phi = (1j*k*np.exp(-1j*k*R_far))/(4*np.pi*R_far) * (self.AF * L_theta - ETA_0 * N_phi)

            self.E_ff = np.stack((np.zeros_like(E_theta), E_theta, E_phi), axis=1)

    def calculate_beamwidth(self, **kwargs):
        E_int = np.sum(np.abs(self.E_ff)**2, axis=1)
        E_plot = np.reshape(E_int, (self.theta.size, self.phi.size))
        phi_beam = kwargs.get('phi_slice', 0)
        phi_beam = np.radians(phi_beam)
        phi_indx1 = np.argmin(np.abs(self.phi - phi_beam))
        phi_indx2 = np.argmin(np.abs(self.phi - (phi_beam+np.pi)))
        E_plot_1D = np.concatenate((np.flip(E_plot[1:,phi_indx2][:,None]), E_plot[:,phi_indx1][:,None]))
        log_pattern = 10*np.log10(E_plot_1D[:,0]/np.amax(E_plot_1D))
        angles = np.linspace(-np.pi/2, np.pi/2, log_pattern.size)*180/np.pi
        angles_interp = np.linspace(angles[0], angles[-1], 10000)
        log_pattern_interp = np.interp(angles_interp, angles, log_pattern)
        peak_indx = np.argmax(log_pattern_interp)
        log_pattern_left = log_pattern_interp[:peak_indx]
        log_pattern_right = log_pattern_interp[peak_indx:]
        beam_left = np.where(log_pattern_left < (log_pattern_interp[peak_indx]-3))[0][-1]
        beam_right = np.where(log_pattern_right < (log_pattern_interp[peak_indx]-3))[0][0]
        bw = angles_interp[peak_indx+beam_right] - angles_interp[beam_left]
        if not self.quiet:
            print('Beamwidth: {:.2f} degrees'.format(bw))
        return bw
    
    def calculate_directivity(self, **kwargs):
        quiet = kwargs.get('quiet', self.quiet)
        Theta, Phi = np.meshgrid(self.theta, self.phi, indexing='ij')
        E_int = np.sum(np.abs(self.E_ff)**2, axis=1)
        P_tot_num = np.trapz(np.trapz(np.reshape(E_int, (self.theta.size, self.phi.size)) * np.sin(np.reshape(Theta, (self.theta.size, self.phi.size))), self.theta, axis=0), self.phi, axis=0)
        directivity = 10*np.log10(4*np.pi*np.amax(E_int)/P_tot_num)
        if not quiet:
            print('Directivity: {:.2f} dB'.format(directivity))
        return directivity
    
    def calculate_gain(self, **kwargs):
        quiet = kwargs.get('quiet', self.quiet)
        E_int = np.sum(np.abs(self.E_ff)**2, axis=1)
        U_int = 1/(2*ETA_0) * E_int
        G = 10*np.log10(4*np.pi*np.amax(U_int))
        if not quiet:
            print('Gain: {:.2f} dB'.format(G))
        return G
    
    def calculate_efficiency(self):
        directivity = self.calculate_directivity(quiet=True)
        gain = self.calculate_gain(quiet=True)
        efficiency = 10**((gain - directivity)/10)
        if not self.quiet:
            print('Efficiency: {:.2f}%'.format(efficiency*100))
        return efficiency
    
    def calculate_beam_metrics(self, **kwargs):
        self.bw = self.calculate_beamwidth(**kwargs)
        self.directivity = self.calculate_directivity()
        self.gain = self.calculate_gain()
        self.efficiency = self.calculate_efficiency()