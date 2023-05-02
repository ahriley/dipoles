import numpy as np
import pickle
# import utilities.coordinate as wcoord
import dipole
import os

# NOTE: I ripped this code from bootstrapping.py to really quickly get m12f 463
# and m12i 600 without any bootstrapping.

# TODO: SERIOUSLY CLEAN THIS UP!!!!!!

# parameters for the fits
sims = ['m12i', 'm12f']
snapshots = [600, 463]
analysis_list = ['binned', 'complementary']
data_dir = dipole.get_data_directory()

width = 5
centers = np.arange(30, 301, 10)
part_type = 'star'
low = 25
high = 305

for sim, snapshot in zip(sims, snapshots):
    # read in particle data
    part, hal = dipole.load_data(sim, snapshot)

    # positions and velocities
    xyz = part[part_type].prop('host.distance.principal')
    v_sph = part[part_type].prop('host.velocity.principal.spherical')

    # need to precompute bound substructure mask b/c based on star indices
    if part_type == 'star':
        hal = dipole.assign_particles_to_halo(sim=sim, snapshot=snapshot,
                                              part=part, halos=hal)
        mask_bound = dipole.mask_bound_stars(part['star'], hal)

    dist = np.sqrt(np.sum(xyz**2, axis=1))

    for analysis in analysis_list:
        outfile = f'{sim}-{snapshot}-{analysis}.pickle'
        outfile = os.path.join(data_dir, outfile)

        # distance shells
        if analysis == 'binned':
            bnds = np.column_stack([centers - width/2, centers + width/2])
            masks = [dipole.mask_distance_shell(dist, bd[0], bd[1]) for bd in bnds]
        elif analysis == 'cumulative':
            masks = [dipole.mask_distance_shell(dist, low, c) for c in centers]
        elif analysis == 'complementary':
            masks = [dipole.mask_distance_shell(dist, c, high) for c in centers]

        # mask bound stars (or random fraction for DM)
        if part_type == 'star':
            masks = [mask & mask_bound for mask in masks]
            # mask_faint = dipole.mask_faint_tracer s(dist, tracer=tracer)
            # masks = [mask & mask_bound & mask_faint for mask in masks]
        elif part_type == 'dark':
            frac = 0.005
            rng = np.random.default_rng()
            mask_random = rng.choice([True, False], p=[frac, 1-frac], size=len(xyz),
                                     replace=True)
            masks = [mask & mask_random for mask in masks]

        # looping over distances
        rootstring = f"Fitting {analysis} dipoles for {sim} snapshot {snapshot}"
        print(rootstring)
        initial = [0, 0, 0, 0, 0, 0, np.log10(50), np.log10(50), np.log10(50)]
        solutions = []
        for mask, center in zip(masks, centers):
            print(center, np.sum(mask))
            soln = dipole.optimize_pars_global(initial, xyz[mask], v_sph[mask])
            solutions.append(soln)

        # save to file after every loop (protects against failures)
        data = {'centers': centers, 'width': width, 'solutions': solutions}
        with open(outfile, "wb") as f:
            pickle.dump(data, f)

        print("Results saved to {0}".format(outfile))
