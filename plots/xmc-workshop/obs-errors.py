import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
import numpy as np
import dipole
import astropy.coordinates as coord
import astropy.units as u
from matplotlib import rcParams

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
              'mathtext.fontset': 'cm',
              'figure.facecolor': 'black',
              "figure.edgecolor": "black",
              "savefig.facecolor": "black",
              "savefig.edgecolor": "black"
              }
    rcParams.update(params)
    dashcolor = 'w'
    fstring = '-dark'
else:
    plt.style.use('home/dipoles/plots/presentation.mplstyle')
    dashcolor = 'k'
    fstring = ''

sim = 'm12f'
snapshot = 463
tracer = 'rgb'
outfile = f'home/dipoles/plots/{sim}-{snapshot}-bootstrapped-{tracer}{fstring}.pdf'

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[9.5, 4], sharex=True)

labels = ['v_travel [km/s]', 'theta0 / pi', 'phi0 / pi']

custom_lines = [Line2D([0], [0], color='C0', lw=4),
                Line2D([0], [0], color='C1', lw=4),
                Line2D([0], [0], color=dashcolor, lw=2, ls='dashed')]

for style, color in zip(['binned', 'complementary'], ['C0', 'C1']):
    infile = f'home/dipoles/data/{sim}-{snapshot}-{style}-bootstrapped-{tracer}.pickle'
    with open(infile, "rb") as f:
        result = pickle.load(f)

    solutions = result['solutions']
    centers = result['centers']
    labeled = False

    transformed = []
    for ii in range(len(solutions)):
        soln = solutions[ii]
        part_type = 'star'
        solution = np.array([s.x for s in soln])
        solution = dipole.transform_model_outputs(solution)
        transformed.append(solution)
    transformed = np.array(transformed)

    low, med, high = np.percentile(transformed, q=[16, 50, 84], axis=0)
    axes[0].fill_between(centers, low[:, 0], high[:, 0], color=color, alpha=0.5)

    filename = f'home/dipoles/data/{sim}-{snapshot}-{style}.pickle'
    with open(filename, "rb") as f:
        result = pickle.load(f)
    solution = np.array([s.x for s in result['solutions']])
    solution = dipole.transform_model_outputs(solution)
    kwargs = {'c': color, 'lw': 4, 'ls': '--'}

    axes[0].plot(centers, solution[:, 0], **kwargs)

    lat = coord.Latitude((transformed[:, :, 1] - np.pi/2)*u.rad)
    lon = coord.Longitude(transformed[:, :, 2]*u.rad)
    sc = coord.Galactic(l=lon, b=lat)

    lat = coord.Latitude((solution[:, 1] - np.pi/2)*u.rad)
    lon = coord.Longitude(solution[:, 2]*u.rad)
    sc_noerr = coord.Galactic(l=lon, b=lat)

    sep = sc.separation(sc_noerr).deg
    low, high = np.percentile(sep, q=[16, 84], axis=0)

    axes[1].fill_between(centers, low, high, color=color, alpha=0.5)

for ax in axes:
    ax.set_xlabel(r'$\mathrm{Distance}$ $\mathrm{[kpc]}$')
axes[0].set_ylabel(r'$v_{\mathrm{dipole}}$ $\mathrm{[km/s]}$')
axes[1].set_ylabel(r'$\mathrm{Offset \ relative \ to \ no \ error}$ $\mathrm{[deg]}$')

axes[0].legend(custom_lines, ['binned', 'cumulative', 'no error'], loc='upper left')
axes[0].set_xlim(30, 300)
axes[0].set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(outfile, bbox_inches='tight')
