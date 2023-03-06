import numpy as np
import pickle
# import utilities.coordinate as wcoord
import dipole
import os

# parameters for the fits
sim = 'm12f'
snapshot = 463
analysis = 'binned'
tracer = 'rrl'
# matrix = 'z0'
# outfile = f'home/data/{sim}-{snapshot}-{analysis}-bootstrapped-'
# outfile += f'{tracer}-{matrix}matrix.pickle'
data_dir = dipole.get_data_directory()
outfile = f'{sim}-{snapshot}-{analysis}-bootstrapped-{tracer}.pickle'
outfile = os.path.join(data_dir, outfile)

width = 5
centers = np.arange(30, 301, 10)
part_type = 'star'
low = 25
high = 305
n_boots = 100

# read in particle data
part, hal = dipole.load_data(sim, snapshot)

# positions and velocities
xyz_full = part[part_type].prop('host.distance.principal')
v_sph_full = part[part_type].prop('host.velocity.principal.spherical')

# if matrix == 'z':
#     # rotation matrix from snapshot
#     xyz_full = part[part_type].prop('host.distance.principal')
#     v_sph_full = part[part_type].prop('host.velocity.principal.spherical')
# elif matrix == 'z0':
#     # rotation matrix from z=0
#     kwargs = {'system_from': 'cartesian', 'system_to': 'spherical'}
#     rotation_matrix = dipole.get_z0_rotation_matrix(sim)
#     xyz = part[part_type].prop('host.distance')
#     xyz_full = wcoord.get_coordinates_rotated(xyz, rotation_matrix)
#     v_xyz = part[part_type].prop('host.velocity')
#     v_xyz = wcoord.get_coordinates_rotated(v_xyz, rotation_matrix)
#     v_sph_full = wcoord.get_velocities_in_coordinate_system(v_xyz, xyz_full,
#                                                             **kwargs)

# need to precompute bound substructure mask b/c based on star indices
if part_type == 'star':
    hal = dipole.assign_particles_to_halo(sim=sim, snapshot=snapshot,
                                          part=part, halos=hal)
    mask_bound_full = dipole.mask_bound_stars(part['star'], hal)

n_stars = len(xyz_full)

solutions_arr = []
for ii in range(n_boots):
    rng = np.random.default_rng()
    boot_indices = rng.choice(n_stars, size=n_stars, replace=True)
    xyz = xyz_full[boot_indices]
    v_sph = v_sph_full[boot_indices]
    dist = np.sqrt(np.sum(xyz**2, axis=1))

    if tracer:
        errs = dipole.sample_uncertainties(dist, tracer=tracer)
        xyz = (errs[0] * xyz.T).T
        dist = np.sqrt(np.sum(xyz**2, axis=1))
        v_sph[:, 0] += errs[1]
        v_sph[:, 1] += errs[2]
        v_sph[:, 2] += errs[3]

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
        mask_bound = mask_bound_full[boot_indices]
        masks = [mask & mask_bound for mask in masks]
        # mask_faint = dipole.mask_faint_tracer s(dist, tracer=tracer)
        # masks = [mask & mask_bound & mask_faint for mask in masks]
    elif part_type == 'dark':
        frac = 0.005
        rng = np.random.default_rng()
        mask_random = rng.choice([True, False], p=[frac, 1-frac],
                                 size=len(xyz), replace=True)
        masks = [mask & mask_random for mask in masks]

    # looping over distances
    rootstring = f"Fitting {analysis} dipoles for {sim} snapshot {snapshot}"
    print(rootstring + f": {ii} / {n_boots}")
    initial = [0, 0, 0, 0, 0, 0, np.log10(50), np.log10(50), np.log10(50)]
    solutions = []
    for mask, center in zip(masks, centers):
        print(center, np.sum(mask))
        soln = dipole.optimize_pars_global(initial, xyz[mask], v_sph[mask])
        solutions.append(soln)
    solutions_arr.append(solutions)

    # save to file after every loop (protects against failures)
    data = {'centers': centers, 'width': width,
            'n_boots': n_boots, 'solutions': solutions_arr}
    with open(outfile, "wb") as f:
        pickle.dump(data, f)

print("Results saved to {0}".format(outfile))
