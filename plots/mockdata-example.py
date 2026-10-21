import numpy as np
import utilities.basic.coordinate as wcoord
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import healpy as hp
import warnings
import sys

sys.path.append("home/code/")
import dipole

# NOTE: not good practice but the healpy plotting warnings are annoying
warnings.filterwarnings("ignore")

plt.style.use('home/plots/4bads-talk/scripts/presentation.mplstyle')

# parameters and output file
mean = 0
sig = 10

# no kick
# metapars = np.array([0, 0, np.pi, mean, mean, mean, sig, sig, sig])
# outfile = 'home/plots/4bads-talk/plots/fakedata_nokick.pdf'

# kick straight up
# metapars = np.array([50, 0, np.pi, mean, mean, mean, sig, sig, sig])
# outfile = 'home/plots/4bads-talk/plots/fakedata_straightup.pdf'

# kick at angle
metapars = np.array([50, np.pi/3, np.pi, mean, mean, mean, sig, sig, sig])
outfile = 'home/plots/4bads-talk/plots/fakedata_angled.pdf'

# convert to likelihood pars
pars = metapars.copy()
pars[:3] = wcoord.get_positions_in_coordinate_system(pars[:3], 'spherical',
                                                     'cartesian')
pars[6:] = np.log10(pars[6:])

# generate mock data
xyz, v_sph = dipole.generate_fake_data(pars, size=(100000, 3), seed=42)

# for healpy plots
nside = 32
sph = wcoord.get_positions_in_coordinate_system(xyz, 'cartesian', 'spherical')
dist, lat, lon = sph.T
hpx_maps = dipole.hpx_velocities(lon, lat, v_sph, nside)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))
indices = [(0, 1), (1, 2), (0, 2)]
labels = [r'$v_r$', r'$v_\theta$', r'$v_\phi$']
# vmax = np.max(np.abs(v_sph))
vmax = 75

# combo plots
for i, ii in zip(range(3), indices):
    axes[0, i].hist2d(v_sph[:, ii[0]], v_sph[:, ii[1]],
                      norm=LogNorm(), bins=100)
    axes[0, i].set_xlim(-vmax, vmax)
    axes[0, i].set_ylim(-vmax, vmax)
    axes[0, i].set_xlabel(labels[ii[0]], fontsize='xx-large')
    axes[0, i].set_ylabel(labels[ii[1]], fontsize='xx-large')
    axes[0, i].set_aspect('equal')

# 1-D histograms
for i in range(3):
    axes[1, i].hist(v_sph[:, i], bins=50, histtype='step')
    axes[1, i].get_yaxis().set_visible(False)
    axes[1, i].set_xlabel(labels[i], fontsize='xx-large')
    axes[1, i].set_xlim(-vmax, vmax)

# healpy maps
for i in range(3):
    plt.axes(axes[2, i])
    hp.mollview(hpx_maps[i+1], hold=True, title=None, min=-50, max=50)

# fig.suptitle(metapars)
fig.tight_layout()
fig.savefig(outfile, bbox_inches='tight', facecolor='white')
