import numpy as np
import pickle
import utilities.coordinate as wcoord
import dipole

# in case we need an RNG
rng = np.random.default_rng()

# parameters for the fits
analysis = 'complementary'
z0matrix = True

# baseline (last ~1/2 Gyr)
# sim = 'm12i'
# start = 570
# stop = 600
# snapshot_arr = np.linspace(start, stop, 20, dtype=int)

sim = 'm12f'
snap_center = 463
snap_range = 20
snapshot_arr = np.arange(snap_center-snap_range, snap_center+snap_range+1, 1)

# sim = 'm12b'
# snap_center = 385
# snap_range = 25
# snapshot_arr = np.arange(snap_center-snap_range, snap_center+snap_range+1, 1)

# m12r has many LMCs at once, probably bad example
# sim = 'm12r'
# snap_center = 562
# snap_range = 25
# snapshot_arr = np.arange(snap_center-snap_range, snap_center+snap_range+1, 1)

if z0matrix:
    rotation_matrix = dipole.get_z0_rotation_matrix(sim)
    style = 'z0matrix'
else:
    style = 'rotmatrixchanges'

width = 5
centers = np.arange(30, 301, 10)
outfile = f'home/data/{sim}-star-time-{analysis}-{style}.pickle'
part_type = 'star'
low = 25
high = 305

solutions_arr = []
for snapshot in snapshot_arr:
    part, hal = dipole.load_data(sim, snapshot)

    # positions and velocities
    if z0matrix:
        # apply same rotation matrix to everything
        k = {'system_from': 'cartesian', 'system_to': 'spherical'}
        xyz = part[part_type].prop('host.distance')
        xyz = wcoord.get_coordinates_rotated(xyz, rotation_matrix)
        v_xyz = part[part_type].prop('host.velocity')
        v_xyz = wcoord.get_coordinates_rotated(v_xyz, rotation_matrix)
        v_sph = wcoord.get_velocities_in_coordinate_system(v_xyz, xyz, **k)
    else:
        # rotation matrix varies by snapshot
        xyz = part[part_type].prop('host.distance.principal')
        v_sph = part[part_type].prop('host.velocity.principal.spherical')
    dist = np.sqrt(np.sum(xyz**2, axis=1))

    # distance shells
    if analysis == 'binned':
        bnds = np.column_stack([centers - width/2, centers + width/2])
        masks = [dipole.mask_distance_shell(dist, bd[0], bd[1])
                 for bd in bnds]
    elif analysis == 'cumulative':
        masks = [dipole.mask_distance_shell(dist, low, c) for c in centers]
    elif analysis == 'complementary':
        masks = [dipole.mask_distance_shell(dist, c, high) for c in centers]

    # mask bound stars (or random fraction for DM)
    if part_type == 'star':
        hal = dipole.assign_particles_to_halo(sim=sim, snapshot=snapshot,
                                              part=part, halos=hal)
        mask_bound = dipole.mask_bound_stars(part['star'], hal)
        masks = [mask & mask_bound for mask in masks]
    elif part_type == 'dark':
        frac = 0.005
        mask_random = rng.choice([True, False], p=[frac, 1-frac],
                                 size=len(xyz), replace=True)
        masks = [mask & mask_random for mask in masks]

    # looping over distances
    print("Fitting {1} dipoles for snapshot: {0}".format(snapshot, analysis))
    initial = [0, 0, 0, 0, 0, 0, np.log10(50), np.log10(50), np.log10(50)]
    solutions = []
    for mask, center in zip(masks, centers):
        print(center, np.sum(mask))
        soln = dipole.optimize_pars_global(initial, xyz[mask], v_sph[mask])
        solutions.append(soln)
    solutions_arr.append(solutions)

    # save to file after every loop (protects against failures)
    data = {'centers': centers, 'width': width,
            'snapshots': snapshot_arr, 'solutions': solutions_arr}
    with open(outfile, "wb") as f:
        pickle.dump(data, f)

print("Results saved to {0}".format(outfile))
