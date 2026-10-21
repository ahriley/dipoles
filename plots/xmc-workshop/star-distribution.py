import gizmo_analysis as gizmo
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams

# no LMC analog
# simulation_directory = 'data/latte_metaldiff/m12i_res7100'
# snapshot = 600
# outfile = 'home/dipoles/plots/m12i-600-star-xyz.pdf'

# good LMC analog
simulation_directory = 'data/latte_metaldiff/m12f_res7100'
snapshot = 463
outfile = 'home/dipoles/plots/m12f-463-star-xyz.pdf'

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
              }
    rcParams.update(params)
    fstring = '-dark'
else:
    plt.style.use('home/dipoles/plots/presentation.mplstyle')
    fstring = ''

part_type = 'star'
part = gizmo.io.Read.read_snapshots(part_type, 'snapshot', snapshot,
                                    simulation_directory)
xyz = part[part_type].prop('host.distance.principal')

sel = (xyz[:, 0] < 100) & (xyz[:, 0] > -100)
sel &= (xyz[:, 1] < 100) & (xyz[:, 1] > -100)
sel &= (xyz[:, 2] < 100) & (xyz[:, 2] > -100)

xyz = xyz[sel]

fig, ax = plt.subplots(figsize=(6, 6))
ax.hist2d(xyz[:, 0], xyz[:, 2], norm=LogNorm(), bins=300, range=[[-100, 100], [-100, 100]], cmap='bone')
ax.set_xlabel(r'$x$ $\mathrm{[kpc]}$')
ax.set_ylabel(r'$z$ $\mathrm{[kpc]}$')
ax.set_aspect('equal')
fig.tight_layout()

fig.savefig(outfile, bbox_inches='tight')
