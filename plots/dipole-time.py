import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
import utilities.basic.coordinate as wcoord

sys.path.append("home/code/")
import dipole

plt.style.use('home/plots/dda-talk/scripts/presentation.mplstyle')


def get_plot_data(filename, orbit, distance_bin_index=False):
    # load in saved fits
    with open(filename, "rb") as f:
        data = pickle.load(f)

    # get data into a format we like
    snapshots = data['snapshots']
    centers = data['centers']

    solns_orig = []
    for ii in range(len(snapshots)):
        solns_orig.append([soln.x for soln in data['solutions'][ii]])
    solns_orig = np.array(solns_orig)
    solutions = [dipole.transform_model_outputs(soln) for soln in solns_orig]
    solutions = np.array(solutions)

    if distance_bin_index:
        means = solutions[:, distance_bin_index, :]
    else:
        # average over >200 kpc bins
        distance_mask = centers > 150
        means = np.mean(solutions[:, distance_mask, :], axis=1)

    # dot products
    snap_mask = np.isin(orbit['snapshots'], data['snapshots'])
    xyz_masked = orbit['xyz'][snap_mask]
    soln_masked = solns_orig[:, :, :3]

    # normalize vectors
    xyz_masked_unit = (xyz_masked.T / np.linalg.norm(xyz_masked, axis=1)).T
    soln_masked_unit = soln_masked.transpose(2, 0, 1)
    soln_masked_unit = soln_masked_unit / np.linalg.norm(soln_masked, axis=-1)
    soln_masked_unit = soln_masked_unit.transpose(1, 2, 0)

    # calculate dot product
    dotprods = np.array([[np.dot(xyz_masked_unit[ii], soln_masked_unit[ii, jj]) for jj in range(soln_masked_unit.shape[1])] for ii in range(soln_masked_unit.shape[0])])
    if distance_bin_index:
        mean_dotprod = dotprods[:, distance_bin_index]
    else:
        mean_dotprod = np.mean(dotprods[:, distance_mask], axis=1)

    return snapshots, means, mean_dotprod


# files for this script (bads version)
# sim_directory = 'data/latte_metaldiff/m12f_res7100'
# orbitfile = 'home/data/final_bads_week/m12f_lmc_orbit.pickle'
# filename_star = 'home/data/final_bads_week/m12f_star_time.pickle'
# filename_dark = 'home/data/final_bads_week/m12f_dark_time.pickle'
# outfile_orbit = 'home/plots/4bads-talk/plots/dipole_lmc_orbit.pdf'
# outfile_sky = 'home/plots/4bads-talk/plots/dipole_sky_time_3.pdf'

# files for this script (after reassigning stars)
sim = 'm12f'
orbitfile = 'home/data/m12f_lmc_orbit.pickle'
filename_star = 'home/data/m12f_star_time_starsreassigned_binned.pickle'
filename_dark = 'home/data/m12f_star_time_starsreassigned_complementary.pickle'
outfile_orbit = 'home/plots/dda-talk/plots//m12f_dipole_lmc_orbit_onlybinned.pdf'
outfile_sky = 'home/plots/dda-talk/plots/dipole_sky_time_starsreassigned.pdf'

# sim_directory = dipole.get_simulation_directory(sim)

# LMC orbit
with open(orbitfile, "rb") as f:
    orbit = pickle.load(f)

snapshots, means, mean_dotprod = get_plot_data(filename_star, orbit)
snapshots_dark, means_dark, mean_dotprod_dark = get_plot_data(filename_dark, orbit, distance_bin_index=1)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

sel = (orbit['snapshots'] <= snapshots.max()) & (orbit['snapshots'] >= snapshots.min())

# convert to time
t_463 = dipole.snapshot_to_time(sim, 463)
orbit['time'] = dipole.snapshot_to_time(sim, orbit['snapshots']) - t_463
time = dipole.snapshot_to_time(sim, snapshots) - t_463
time_dark = dipole.snapshot_to_time(sim, snapshots_dark) - t_463

xlow = time.min()
xhigh = time.max()

# lmc distance vs time
lmc_dist = np.sqrt(np.sum(orbit['xyz']**2, axis=1))
axes[0].plot(orbit['time'][sel], lmc_dist[sel], c='k')
axes[0].set_ylabel(r'$d_{\mathrm{LMC}}$ $\mathrm{[kpc]}$')

# v_travel vs time
axes[1].plot(time, means[:, 0])
# axes[1].plot(time_dark, means_dark[:, 0])
# axes[1].set_ylim(0, 65)
axes[1].set_ylabel(r'$v_{\mathrm{travel}}$ $\mathrm{[km/s]}$')
# axes[1].set_ylim(0, 100)

# dotproduct vs time
axes[2].plot(time, mean_dotprod, label='binned')
# axes[2].plot(time_dark, mean_dotprod_dark, label='cumulative')
axes[2].set_ylim(-1, 1)
axes[2].set_ylabel(r'$\vec{v}_{\mathrm{travel}} \cdot \vec{r}_{\mathrm{LMC}}$')
# axes[2].legend()

for ax in axes:
    ax.set_xlabel(r'$\mathrm{Time\ [Gyr]}$')
    ax.set_xlim(xlow, xhigh)
    ax.axvline(0, color='k', ls='--')

fig.tight_layout()
plt.savefig(outfile_orbit, bbox_inches='tight', facecolor='white')

# track along sky

# dipole formatting
# dipole_lon = means[:, 2] - np.pi
# dipole_lat = means[:, 1] - np.pi/2

# dipole_lon_dark = means_dark[:, 2] - np.pi
# dipole_lat_dark = means_dark[:, 1] - np.pi/2

# # lmc orbit
# lmc_sph = wcoord.get_positions_in_coordinate_system(orbit['xyz'],
#                                                     'cartesian', 'spherical')
# lmc_lon = lmc_sph[sel, 2] - np.pi
# lmc_lat = lmc_sph[sel, 1] - np.pi/2

# fig = plt.figure(figsize=(6, 6))
# ax = plt.subplot(projection="aitoff")
# kwargs = {'c': time, 'cmap': 'bwr', 'edgecolors': 'k'}
# ax.plot(lmc_lon, lmc_lat, 'k-', zorder=-1)
# sc = ax.scatter(lmc_lon, lmc_lat, marker='D', **kwargs)
# ax.scatter(dipole_lon_dark, dipole_lat_dark, marker='s', **kwargs)
# ax.scatter(dipole_lon, dipole_lat, marker='o', **kwargs)
# ax.grid(True)
# # ax.legend(loc='best')
# fig.colorbar(sc, orientation='horizontal', label=r'$\mathrm{Time\ [Gyr]}$', pad=0.01)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# fig.tight_layout()
# fig.savefig(outfile_sky, bbox_inches='tight', facecolor='w')
