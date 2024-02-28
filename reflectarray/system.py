import numpy as np
from matplotlib import pyplot as plt
import scipy.constants
from reflectarray import toolbox as tb
from reflectarray.compute import Compute
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

    def design(self):
        pass

    def compute_farfield(self, delta_theta, delta_phi, method='integration'):
        self.E_farfield = self.compute.far_field_propagate(self, self.reflectarray, delta_theta, delta_phi, 2*np.pi*self.f/C, method=method)
    
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
