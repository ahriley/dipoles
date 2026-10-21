import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt

sys.path.append("home/code/")
import dipole

plt.style.use('home/plots/4bads-talk/scripts/presentation.mplstyle')


# load in saved fits
# filenames = ['home/data/final_bads_week/m12f_star_time.pickle',
#              'home/data/final_bads_week/m12f_dark_time.pickle']
# outfile = 'home/plots/4bads-talk/plots/m12f_463_dipole_distance.pdf'

filenames = ['home/data/final_bads_week/m12i_600_star.pickle',
             'home/data/final_bads_week/m12i_600_dark.pickle']
outfile = 'home/plots/4bads-talk/plots/m12i_600_dipole_distance.pdf'

# load in data
labels = ['star', 'dark']
fmts = ['o-', 's-']
data_list = []
for filename in filenames:
    with open(filename, "rb") as f:
        data_list.append(pickle.load(f))

centers = data_list[0]['centers']

# use fits vs time
if 'm12f' in filenames[0]:
    vtravels = []
    for data in data_list:
        i_snap = np.where(data['snapshots'] == 463)[0][0]
        solution = np.array([soln.x for soln in data['solutions'][i_snap]])
        solution = dipole.transform_model_outputs(solution)
        vtravels.append(solution[:, 0])
elif 'm12i' in filenames[0]:
    vtravels = []
    for data in data_list:
        solution = np.array([soln.x for soln in data['solutions']])
        solution = dipole.transform_model_outputs(solution)
        vtravels.append(solution[:, 0])

fig, ax = plt.subplots()
for vtravel, label, fmt in zip(vtravels, labels, fmts):
    # format data properly
    ax.plot(centers, vtravel, fmt, label=label)

ax.legend()
ax.set_xlim(50, 300)
ax.set_ylim(0, 100)
ax.set_xlabel(r'$d_{\mathrm{GC}}$ $\mathrm{[kpc]}$')
ax.set_ylabel(r'$v_{\mathrm{travel}}$ $\mathrm{[km/s]}$')
fig.tight_layout()

fig.savefig(outfile, bbox_inches='tight', facecolor='white')
