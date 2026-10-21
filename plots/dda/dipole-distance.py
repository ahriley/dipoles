import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt

sys.path.append("home/code/")
import dipole

plt.style.use('home/plots/dda-talk/scripts/presentation.mplstyle')

sim = 'm12f'
snap = 463

outfile = f'home/plots/dda-talk/plots/{sim}_{snap}_dipole_distance_onlybinned.pdf'
fig, ax = plt.subplots()

for analysis in ['binned']:
    infile = f'home/data/{sim}_star_time_starsreassigned_{analysis}.pickle'

    # load in data
    with open(infile, "rb") as f:
        result = pickle.load(f)
    centers = result['centers']

    # use fits vs time
    if 'm12f' in infile:
        i_snap = np.where(result['snapshots'] == 463)[0][0]
    elif 'm12i' in infile:
        i_snap = np.where(result['snapshots'] == 600)[0][0]
    solution = np.array([soln.x for soln in result['solutions'][i_snap]])
    solution = dipole.transform_model_outputs(solution)
    vtravel = solution[:, 0]

    # plot
    label = 'cumulative' if analysis == 'complementary' else analysis
    ax.plot(centers, vtravel, label=label)

# ax.legend()
ax.set_xlim(30, 300)
ax.set_ylim(0, 100)
ax.set_xlabel(r'$\mathrm{Distance}$ $\mathrm{[kpc]}$')
ax.set_ylabel(r'$v_{\mathrm{travel}}$ $\mathrm{[km/s]}$')
fig.tight_layout()

fig.savefig(outfile, bbox_inches='tight', facecolor='white')
