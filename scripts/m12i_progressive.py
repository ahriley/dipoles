import numpy as np
import pickle
import dipole
import os

# parameters for fits
sim = 'm12i'
snapshot = 600
data_dir = dipole.get_data_directory()
outfile = f'{sim}-{snapshot}-progressive.pickle'
outfile = os.path.join(data_dir, outfile)

width = 5
centers = np.arange(30, 301, 10)
part_type = 'star'
low = 25
high = 305

# read in particle data
part, hal = dipole.load_data(sim, snapshot)

# positions and velocities
xyz = part[part_type].prop('host.distance.principal')
v_sph = part[part_type].prop('host.velocity.principal.spherical')
dist = np.sqrt(np.sum(xyz**2, axis=1))

# precompute masks
hal = dipole.assign_particles_to_halo(sim=sim, snapshot=snapshot,
                                      part=part, halos=hal)
mask_bound = dipole.mask_bound_stars(part['star'], hal)
mask_emily = dipole.mask_m12i_messy_stream(part['star'])
mask_nondh = dipole.mask_nondh_streams_stars(part['star'], sim)
mask_danny = dipole.mask_danny_streams_stars(part['star'], sim)

# different configs
keys = ['basic', 'bound', 'emily', 'danny', 'nondh']
configs = {'basic': {'mask-bound': False, 'mask-emily': False,
                     'mask-danny': False, 'mask-nondh': False},
           'bound': {'mask-bound': True, 'mask-emily': False,
                     'mask-danny': False, 'mask-nondh': False},
           'emily': {'mask-bound': True, 'mask-emily': True,
                     'mask-danny': False, 'mask-nondh': False},
           'danny': {'mask-bound': True, 'mask-emily': True,
                     'mask-danny': True, 'mask-nondh': False},
           'nondh': {'mask-bound': True, 'mask-emily': True,
                     'mask-danny': True, 'mask-nondh': True},
           }

analyses = ['binned', 'complementary']
output = {'centers': centers, 'width': width, 'low': low, 'high': high}
for key in keys:
    config = configs[key]
    output[key] = {'config': config}

    # add non-distance-related masks together
    mask_config = np.ones_like(mask_bound, dtype=bool)
    if config['mask-bound']:
        mask_config &= mask_bound
    if config['mask-emily']:
        mask_config &= mask_emily
    if config['mask-danny']:
        mask_config &= mask_danny
    if config['mask-nondh']:
        mask_config &= mask_nondh

    for analysis in analyses:
        # distance-based mask
        if analysis == 'binned':
            bnds = np.column_stack([centers - width/2, centers + width/2])
            masks = [dipole.mask_distance_shell(dist, bd[0], bd[1])
                     for bd in bnds]
        elif analysis == 'cumulative':
            masks = [dipole.mask_distance_shell(dist, low, c) for c in centers]
        elif analysis == 'complementary':
            masks = [dipole.mask_distance_shell(dist, c, high)
                     for c in centers]

        # combine all masks
        masks = [mask & mask_config for mask in masks]

        # dipole analysis
        rootstring = f"Fitting {analysis} dipoles for {sim} snapshot "
        print(rootstring + f" {snapshot} with config {key}")
        initial = [0, 0, 0, 0, 0, 0, np.log10(50), np.log10(50), np.log10(50)]
        solutions = []
        for mask, center in zip(masks, centers):
            print(center, np.sum(mask))
            soln = dipole.optimize_pars_global(initial, xyz[mask], v_sph[mask])
            solutions.append(soln)

        # save after each analysis
        output[key][analysis] = solutions
        with open(outfile, "wb") as f:
            pickle.dump(output, f)

print("Results saved to {0}".format(outfile))
