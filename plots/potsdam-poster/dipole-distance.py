import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.append("home/code/")
import dipole

dark_version = True

if dark_version:
    plt.style.use('dark_background')
    params = {'font.size': 14,
              'figure.titlesize': 'x-large',
              'axes.titlesize': 'large',
              'axes.labelsize': 'large',
              'xtick.labelsize': 'medium',
              'ytick.labelsize': 'medium',
              'axes.axisbelow': True,
              'axes.linewidth': 1.5,
              'xtick.direction': 'in',
              'ytick.direction': 'in',
              'xtick.top': True,
              'ytick.right': True,
              'lines.linewidth': 2,
              'mathtext.fontset': 'cm'
              }
    rcParams.update(params)
    fstring = '-dark'
else:
    plt.style.use('home/dipoles/plots/presentation.mplstyle')
    fstring = ''

sims = ['m12f', 'm12i']
snapshots = [463, 600]

for sim, snapshot in zip(sims, snapshots):
    outfile = f'home/dipoles/plots/{sim}-{snapshot}-dipole-distance{fstring}.pdf'

    fig, ax = plt.subplots()
    for style, color in zip(['binned', 'complementary'], ['C0', 'C1']):
        filename = f'home/dipoles/data/{sim}-{snapshot}-{style}.pickle'
        with open(filename, "rb") as f:
            result = pickle.load(f)
        centers = result['centers']
        solution = np.array([s.x for s in result['solutions']])
        solution = dipole.transform_model_outputs(solution)
        kwargs = {'c': color, 'lw': 4}

        label = 'cumulative' if style == 'complementary' else style
        ax.plot(centers, solution[:, 0], label=label, **kwargs)

    if sim == 'm12f':
        ax.axhline(53, ls='--', lw=2, label=r'$v^{\mathrm{lim}}_{\mathrm{dipole}} = 53$ $\mathrm{km/s}$')

    ax.legend()
    ax.set_xlim(30, 300)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r'$\mathrm{Distance \ [kpc]}$')
    ax.set_ylabel(r'$v_{\mathrm{dipole}}$ $\mathrm{[km/s]}$')
    fig.tight_layout()

    fig.savefig(outfile, bbox_inches='tight')
