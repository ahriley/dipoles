import halo_analysis as halo
import numpy as np
import sys
import utilities.basic.coordinate as wcoord
import pickle

sys.path.append('home/code/')
import dipole

# parameters for the fits
simulation_directory = 'data/latte_metaldiff/m12f_res7100'
rotation_matrix = dipole.get_z0_rotation_matrix(simulation_directory)
outfile = 'home/data/final_bads_week/m12f_subhaloes_time.pickle'

# TODO: is this the snapshot range we want?
snap_center = 463
snap_range = 20
snapshot_arr = np.arange(snap_center-snap_range, snap_center+snap_range+1, 1)

solutions = [[], []]
for snapshot in snapshot_arr:
    # load in the halo catalog
    hal = halo.io.IO.read_catalogs('snapshot', snapshot, simulation_directory)

    # positions and velocities (same rotation matrix)
    kwargs = {'system_from': 'cartesian', 'system_to': 'spherical'}
    xyz = hal.prop('host.distance')
    xyz = wcoord.get_coordinates_rotated(xyz, rotation_matrix)
    v_xyz = hal.prop('host.velocity')
    v_xyz = wcoord.get_coordinates_rotated(v_xyz, rotation_matrix)
    v_sph = wcoord.get_velocities_in_coordinate_system(v_xyz, xyz, **kwargs)
    dist = np.sqrt(np.sum(xyz**2, axis=1))

    # mask data we don't want
    mask_base = np.full_like(hal['mass'], True, dtype=bool)
    mask_base &= dist != 0
    mask_base &= dist < 300
    mask_base &= dist > 100

    # mask by mass or stellar mass
    mask_lum = hal['star.mass'] > 0
    mask_mass = hal['mass'] > 10**7

    initial = [0, 0, 0, 0, 0, 0, np.log10(50), np.log10(50), np.log10(50)]
    for mask_tracer, solution in zip([mask_lum, mask_mass], solutions):
        mask = mask_base & mask_tracer
        soln = dipole.optimize_pars_global(initial, xyz[mask], v_sph[mask])
        solution.append(soln)

# save to file (this is fast so not worried about failures)
data = {'snapshots': snapshot_arr, 'solutions_luminous': solutions[0],
        'solutions_1e7': solutions[1]}
with open(outfile, "wb") as f:
    pickle.dump(data, f)
