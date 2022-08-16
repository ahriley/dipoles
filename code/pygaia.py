import numpy as np

# from https://github.com/agabrown/PyGaia, modest syntax changes for flake8

_scaling_for_proper_motions = {'dr3': {'Total': 0.96, 'AlphaStar': 1.03,
                                       'Delta': 0.89},
                               'dr4': {'Total': 0.54, 'AlphaStar': 0.58,
                                       'Delta': 0.50},
                               'dr5': {'Total': 0.27, 'AlphaStar': 0.29,
                                       'Delta': 0.25}}

_science_margin = 1.1
_t_factor = {'dr3': 1.0, 'dr4': 0.749, 'dr5': 0.527}
_default_release = 'dr4'
_bright_floor_star_plx = 13.0


def proper_motion_uncertainty(gmag, release=_default_release):
    plx_unc = parallax_uncertainty(gmag, release=release)
    return _scaling_for_proper_motions[release]['AlphaStar'] * plx_unc, \
        _scaling_for_proper_motions[release]['Delta'] * plx_unc


def parallax_uncertainty(gmag, release=_default_release):
    z = calc_z_plx(gmag)
    value = np.sqrt(40 + 800 * z + 30 * z * z) * _t_factor[release]
    if release == 'dr3':
        return value / _science_margin
    else:
        return value


def calc_z_plx(gmag):
    gatefloor = np.power(10.0, 0.4 * (_bright_floor_star_plx - 15.0))
    if np.isscalar(gmag):
        result = np.amax((gatefloor, np.power(10.0, 0.4 * (gmag - 15.0))))
    else:
        result = np.power(10.0, 0.4 * (gmag - 15.0))
        indices = (result < gatefloor)
        result[indices] = gatefloor
    return result
