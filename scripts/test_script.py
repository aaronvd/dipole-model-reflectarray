# taken from "binary_test_3.ipynb" and "metasurface_feed_test.ipynb", with plots removed

import numpy as np
from matplotlib import pyplot as plt
import scipy.constants
import scipy.io
import sys
sys.path.append('..')
from reflectarray import element
from reflectarray.reflectarray import Reflectarray
from reflectarray.feed import Feed
from reflectarray.feed import PyramidalHorn
from reflectarray.system import System

# %matplotlib qt                    # plots in an interactive window

#
C = scipy.constants.c
f = 10E9
lam = C/f

alpha_on_filepath = './data/alpha_patch.mat'
alpha_off_filepath = './data/alpha_patch_short.mat'
patch_on = scipy.io.loadmat(alpha_on_filepath)
patch_off = scipy.io.loadmat(alpha_off_filepath)
patch_on_f = patch_on['f'][:-1,0]
patch_off_f = patch_off['f_short'][:-1,0]
alpha_on = patch_on['a0'][:,0][np.argmin(np.abs(patch_on_f - f))] / 2       # isolating single slot dipole. Divya included image dipole and used total field.
alpha_off = patch_off['a_short'][:,0][np.argmin(np.abs(patch_off_f - f))] / 2

alpha = np.array([alpha_on, alpha_off])
patch1 = element.Patch(f=f, alpha=alpha, lattice_vectors=np.array([[0, 1, 0]]), W=lam/4)

# Second Cell
L = 10*lam
delta_x = lam/4
delta_y = lam/4
x = np.arange(-L/2, L/2+delta_x, delta_x)
y = np.arange(-L/2, L/2+delta_y, delta_y)

array = Reflectarray(patch1, x=x, y=y)

#

Lx_ms = 3*lam
Ly_ms = 3*lam
delta_x_ms = lam/2
delta_y_ms = lam/2
x_ms = np.arange(-Lx_ms/2, Lx_ms/2+delta_x_ms, delta_x_ms)
y_ms = np.arange(-Ly_ms/2, Ly_ms/2+delta_y_ms, delta_y_ms)
X_ms, Y_ms = np.meshgrid(x_ms, y_ms, indexing='ij')
Z_ms = np.zeros_like(X_ms)
r_ms = np.stack((X_ms.flatten(), Y_ms.flatten(), Z_ms.flatten()), axis=1)
J_m = np.random.randn(r_ms.shape[0]) + 1j*np.random.randn(r_ms.shape[0])
J_m = 1 + 0 * J_m
J_m = np.stack((np.zeros_like(J_m), J_m, np.zeros_like(J_m)), axis=1)


feed = Feed(x=x_ms, y=y_ms, J_m=J_m, f=f, rotation=(0, 0, 0), r_offset=(0, 0, 0.01*lam))

#
system1 = System(array, feed)

#
system1.design(theta_beam=25, phi_beam=0, mapping='ideal', R_far=10, scale=.5)

#
system1.propagate(delta_theta=1, delta_phi=2*90)

#
system1.calculate_beam_metrics()
system1.plot_fields(plot_type='1D', phi_slice=0, dB_min=-40)

plt.show()




