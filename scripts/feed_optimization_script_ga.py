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
from reflectarray.ga2 import geneticalgorithm as ga

# %matplotlib qt                    # plots in an interactive window

def setup_ga(max_iterations, population_size):
    global array
    dimension = 3
    varbound = np.array([[1, 5], [1, 5], [3, 8]])
    variable_type = 'real' # Can also be "int" or "bool"
    max_iterations_without_improvement = 5
    algorithm_param = {'max_num_iteration': max_iterations,
                       'population_size': population_size,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': max_iterations_without_improvement}
    ga_model = ga(function=fitness_func, dimension=dimension,
                  variable_type=variable_type, variable_boundaries=varbound,
                  algorithm_parameters=algorithm_param, convergence_curve=False)
    return ga_model


def create_feed(inputs):
    lx = inputs[0]
    ly = inputs[1]
    z_position = inputs[2]

    Lx_ms = lx * lam
    Ly_ms = ly * lam
    delta_x_ms = lam / 2
    delta_y_ms = lam / 2
    x_ms = np.arange(-Lx_ms / 2, Lx_ms / 2 + delta_x_ms, delta_x_ms)
    y_ms = np.arange(-Ly_ms / 2, Ly_ms / 2 + delta_y_ms, delta_y_ms)
    X_ms, Y_ms = np.meshgrid(x_ms, y_ms, indexing='ij')
    Z_ms = np.zeros_like(X_ms)
    r_ms = np.stack((X_ms.flatten(), Y_ms.flatten(), Z_ms.flatten()), axis=1)
    J_m = np.random.randn(r_ms.shape[0]) + 1j * np.random.randn(r_ms.shape[0])
    J_m = 1 + 0 * J_m
    J_m = np.stack((np.zeros_like(J_m), J_m, np.zeros_like(J_m)), axis=1)
    feed = Feed(x=x_ms, y=y_ms, J_m=J_m, f=f, rotation=(0, 0, 0), r_offset=(0, 0, z_position * lam))
    return feed


def create_and_run_system(array, feed):
    system1 = System(array, feed, quiet=True)
    system1.design(theta_beam=25, phi_beam=0, mapping='phase', R_far=10, scale=.5)
    system1.propagate(delta_theta=1, delta_phi=2 * 90)
    system1.calculate_beam_metrics()
    return system1


def fitness_func(inputs):
    global array
    feed = create_feed(inputs)
    system1 = create_and_run_system(array, feed)
    return -system1.compute.directivity


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

#
L = 10*lam
delta_x = lam/4
delta_y = lam/4
x = np.arange(-L/2, L/2+delta_x, delta_x)
y = np.arange(-L/2, L/2+delta_y, delta_y)

array = Reflectarray(patch1, x=x, y=y)

ga_model = setup_ga(max_iterations=20, population_size=20)
ga_model.run()
print('\n', ga_model.output_dict)

solution = ga_model.output_dict['variable']
feed = create_feed(solution)
system1 = create_and_run_system(array, feed)

ax0 = plt.subplot(121)
ax0.plot(-np.array(ga_model.report))
ax0.set_title('Optimization Process')
ax0.set_xlabel('Iteration')
ax0.set_ylabel('Reflectarray Directivity (dBi)')
ax0.grid()
ax1 = plt.subplot(122, projection='polar')
system1.plot_fields(plot_type='1D', phi_slice=0, dB_min=-40, ax=ax1)
ax1.set_title('Resulting Radiation Pattern')
plt.show()



