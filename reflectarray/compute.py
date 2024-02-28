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

class Compute:

    def __init__(self, quiet=False):
        self.quiet = quiet

    def near_field_propagate(self, source, r_target, k, field='H'):
        x_source = np.unique(source.r[:,0])
        y_source = np.unique(source.r[:,1])
        
        if field == 'H':
            H_A = np.empty(r_target.shape, dtype=np.complex64)
            H_F = np.empty(r_target.shape, dtype=np.complex64)
            for i in range(r_target.shape[0]):
                R_vec = r_target[i,:][None,:] - source.r
                R_norm = np.linalg.norm(R_vec, axis=1, keepdims=True)
                R_hat = R_vec/R_norm
                
                G1 = -(1 + 1j*k*R_norm - k**2 * R_norm**2)/(R_norm**3)
                G2 = (3 + 3*1j*k*R_norm - k**2 * R_norm**2)/(R_norm**5)
                
                H_A_integrand = np.reshape((-1/(4*np.pi)) 
                                * np.cross(R_hat, source.J_e, axisa=1, axisb=1, axisc=1)
                                * (1 + 1j*k*R_norm)/R_norm**2
                                * np.exp(-1j*k*R_norm), (x_source.size, y_source.size, 3))
                H_F_integrand = np.reshape((-1j/(4*np.pi*k*ETA_0))
                                        * (G1 * source.J_m + 
                                            G2 * R_vec * np.sum(R_vec * source.J_m, axis=1, keepdims=True))
                                        * np.exp(-1j*k*R_norm), (x_source.size, y_source.size, 3))
                H_A[i,:] = np.trapz(np.trapz(H_A_integrand, x_source, axis=0), y_source, axis=0)
                H_F[i,:] = np.trapz(np.trapz(H_F_integrand, x_source, axis=0), y_source, axis=0)
                
            return H_A + H_F

    def far_field_propagate(self, source, delta_theta, delta_phi, k, method='integration'):
        
        x_source = np.unique(source.r[:,0])
        y_source = np.unique(source.r[:,1])
        
        if method == 'integration':
            
            theta = np.radians(np.arange(0, 90+delta_theta, delta_theta))
            phi = np.radians(np.arange(0, 360+delta_phi, delta_phi))
            Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
            Theta = np.reshape(Theta, -1)
            Phi = np.reshape(Phi, -1)

            r_hat = transformations.spherical_to_cartesian_vector(np.transpose(np.array([1, 0, 0])[:,None]),
                                                                  Theta, 
                                                                  Phi)
            k_far_vec = k * r_hat

            if source.J_e is not None:
                J_e_grid = np.reshape(source.J_e, (x_source.size, y_source.size, 3))
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
                    
                    N_theta[i] = np.trapz(np.trapz(integrand_theta, x_source, axis=0), y_source, axis=0)
                    N_phi[i] = np.trapz(np.trapz(integrand_phi, x_source, axis=0), y_source, axis=0)
            else:
                N_theta = 0
                N_phi = 0
            
            if source.J_m is not None:
                J_m_grid = np.reshape(source.J_m, (x_source.size, y_source.size, 3))
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
                    
                    L_theta[i] = np.trapz(np.trapz(integrand_theta, x_source, axis=0), y_source, axis=0)
                    L_phi[i] = np.trapz(np.trapz(integrand_phi, x_source, axis=0), y_source, axis=0)
            else:
                L_theta = 0
                L_phi = 0

            R_far = 1

            E_theta = -(1j*k*np.exp(-1j*k*R_far))/(4*np.pi*R_far) * (L_phi + ETA_0 * N_theta)
            E_phi = (1j*k*np.exp(-1j*k*R_far))/(4*np.pi*R_far) * (L_theta - ETA_0 * N_phi)

            return np.stack((np.zeros_like(E_theta), E_theta, E_phi), axis=1)