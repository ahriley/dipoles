import gizmo_analysis as gizmo
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.style.use('home/plots/4bads-talk/scripts/presentation.mplstyle')

# no LMC analog
simulation_directory = 'data/latte_metaldiff/m12i_res7100'
snapshot = 600
outfile = 'home/plots/4bads-talk/plots/m12i_600_star_xyz.pdf'

# good LMC analog
# simulation_directory = 'data/latte_metaldiff/m12f_res7100'
# snapshot = 463
# outfile = 'home/plots/4bads-talk/plots/m12f_463_star_xyz.pdf'

part_type = 'star'
part = gizmo.io.Read.read_snapshots(part_type, 'snapshot', snapshot,
                                    simulation_directory)
xyz = part[part_type].prop('host.distance.principal')

sel = (xyz[:, 0] < 100) & (xyz[:, 0] > -100)
sel &= (xyz[:, 1] < 100) & (xyz[:, 1] > -100)
sel &= (xyz[:, 2] < 100) & (xyz[:, 2] > -100)

xyz = xyz[sel]

fig, ax = plt.subplots(figsize=(6, 6))
ax.hist2d(xyz[:, 0], xyz[:, 2], norm=LogNorm(), bins=1000)
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_xlabel(r'$x$ $\mathrm{[kpc]}$')
ax.set_ylabel(r'$z$ $\mathrm{[kpc]}$')
ax.set_aspect('equal')
fig.tight_layout()

fig.savefig(outfile, bbox_inches='tight', facecolor='white')
