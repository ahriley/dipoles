import numpy as np
import sys
import pickle

sys.path.append("home/code/")
import dipole

# in case we need an RNG
rng = np.random.default_rng()

# parameters for the fits
# NOTE: assumes coordinate rotation matrix at z=0 for each sim
outfile = 'home/data/all_halos_600_starsreassigned_nondhstreams_metalpoor.pickle'
snapshot = 600
width = 5
low = 25
centers = np.arange(30, 301, 10)
part_types = ['star']
remove_lmc_debris = False
cumulative = False
kathryn = False
metals = 'metalpoor'

sims = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12r', 'm12w']
data = {'centers': centers, 'width': width, 'low': low}

solutions_arr = []
for sim in sims:
    if remove_lmc_debris:
        if sim != 'm12f':
            print(f'{sim} does not have LMC pre-assigned')
            continue
        lmc_cat_index = 56220
        part_past, hal_past = dipole.load_data(sim, 463, assign_pointers=True)
        hal_past = dipole.assign_particles_to_halo(sim=sim, snapshot=463,
                                                   part=part_past,
                                                   halos=hal_past)
    part, hal = dipole.load_data(sim, snapshot)

    solution_dict = {}
    for part_type in part_types:
        xyz = part[part_type].prop('host.distance.principal')
        v_sph = part[part_type].prop('host.velocity.principal.spherical')
        dist = np.sqrt(np.sum(xyz**2, axis=1))

        # distance shells
        if cumulative:
            if kathryn:
                print('CUMULATIVE LARGE BINS NOT IMPLEMENTED')
                exit()
            masks = [dipole.mask_distance_shell(dist, low, c) for c in centers]
        elif kathryn:
            lows = np.array([low, 50, 150])
            highs = np.array([50, 150, 300])
            centers = (lows + highs) / 2
            masks = [dipole.mask_distance_shell(dist, lo, hi) for lo, hi in
                     zip(lows, highs)]
        else:
            bnds = np.column_stack([centers - width/2, centers + width/2])
            masks = [dipole.mask_distance_shell(dist, bd[0], bd[1])
                     for bd in bnds]
        if part_type == 'star':
            hal = dipole.assign_particles_to_halo(sim=sim, snapshot=snapshot,
                                                  part=part, halos=hal)
            mask_bound = dipole.mask_bound_stars(part['star'], hal)
            mask_nondh = dipole.mask_nondh_streams_stars(part['star'], sim)
            # mask_danny = dipole.mask_danny_streams_stars(part['star'], sim)
            # masks = [mask & mask_bound for mask in masks]
            masks = [mask & mask_bound & mask_nondh for mask in masks]
            # masks = [mask & mask_bound & mask_nondh
            #          & mask_danny for mask in masks]
            if remove_lmc_debris:
                mask_debris = dipole.mask_halo_debris_at_z0(lmc_cat_index,
                                                            part_past,
                                                            hal_past,
                                                            part_z0=part)
                masks = [mask & mask_debris for mask in masks]
            if metals:
                limits = {'metalpoor': [-np.inf, -2],
                          'metalmid': [-2, -1],
                          'metalrich': [-1, np.inf]
                         }
                fe_h = part['star'].prop('metallicity.iron')
                low, high = limits[metals]
                mask_metals = (fe_h > low) & (fe_h < high)
                masks = [mask & mask_metals for mask in masks]
        elif part_type == 'dark':
            frac = 0.005
            mask_random = rng.choice([True, False], p=[frac, 1-frac],
                                     size=len(xyz), replace=True)
            masks = [mask & mask_random for mask in masks]

        # looping over distances
        print("Fitting for {0} {1}".format(sim, part_type))
        initial = [0, 0, 0, 0, 0, 0, np.log10(50), np.log10(50), np.log10(50)]
        solutions = []
        for mask, center in zip(masks, centers):
            print(center, np.sum(mask))
            soln = dipole.optimize_pars_global(initial, xyz[mask], v_sph[mask])
            solutions.append(soln)
        solution_dict[part_type] = solutions

    # save to file after every loop (protects against failures)
    data[sim] = solution_dict
    with open(outfile, "wb") as f:
        pickle.dump(data, f)
