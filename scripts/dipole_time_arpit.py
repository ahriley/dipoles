import numpy as np
import pickle
import utilities.coordinate as wcoord
import dipole
import gizmo_analysis as gizmo
import halo_analysis as halo

# parameters for the fits
analysis = 'binned'         # other options 'complementary' or 'cumulative'
part_type = 'star'          # part_type used
sim = 'm12b'                # sim name (mostly for file outputs)
out_dir = 'home/data/'      # where output goes (see below)
sim_dir = 'dm_alternative/dSIDM_aarora/m12b7e3_sidm1/'  # simulation directory

# snapshot range used for analysis
snap_center = 401
snap_range = 20
snapshot_arr = np.arange(snap_center-snap_range, snap_center+snap_range+1, 1)

# radial bins to use for analysis
centers = np.arange(30, 301, 10)    # radial bin array (x-axis for plots)
width = 5                           # width of radial bins if 'binned' analysis
low = 25                    # cumulative uses particles from [low, center]
high = 305                  # complementary uses particles from [center, high]

# whether to fix rotation matrix to z=0
z0matrix = True
if z0matrix:
    part = gizmo.io.Read.read_snapshots('star', 'redshift', 0, sim_dir)
    rotation_matrix = part.host['rotation'][0]
    style = 'z0matrix'
else:
    style = 'rotmatrixchanges'

outfile = out_dir + f'{sim}-{part_type}-time-{analysis}-{style}.pickle'

# in case we need an RNG
rng = np.random.default_rng()

solutions_arr = []
for snapshot in snapshot_arr:
    # reading in data
    part = gizmo.io.Read.read_snapshots(part_type,
                                        snapshot_values=snapshot,
                                        snapshot_value_kind='index',
                                        simulation_directory=sim_dir,
                                        assign_pointers=False,
                                        sort_dark_by_id=False)
    hal = halo.io.IO.read_catalogs('snapshot', snapshot, sim_dir,
                                   rockstar_directory='halo/rockstar_dm/')

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
        # uncomment if you want to use my function to re-do halo assignment
        # hal = dipole.assign_particles_to_halo(sim=sim, snapshot=snapshot,
        #                                       part=part, halos=hal,
        #                                       data_dir=out_dir)
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
