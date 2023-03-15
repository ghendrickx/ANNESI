"""
Check if input parameters result in a physically sound estuarine design.

Author: Gijs G. Hendrickx
"""
import numpy as np


def _tidal_prism(tidal_range, depth, width, min_width, friction, convergence):
    """Determination of the tidal prism based on the analytical solutions on the tidal damping by van Rijn (2011).

    van Rijn, L.C. (2011). Analytical and numerical analysis of tides and salinities in estuaries; part I: tidal wave
        propagation in convergent estuaries. Ocean Dynamics, 61(11):1719-1741.
        doi:https://doi.org/10.1007/s10236-011-0453-0.

    :param tidal_range: tidal range
    :param depth: channel depth
    :param width: channel width
    :param min_width: minimum channel width
    :param friction: channel friction
    :param convergence: convergence

    :type tidal_range: float
    :type depth: float
    :type width: float
    :type min_width: float
    :type friction: float
    :type convergence: float

    :return: tidal prism
    :rtype: float
    """
    # constants
    gravity = 9.81
    length = 2e5
    period = 12 * 3600

    # tidal wave
    velocity = 1 / (2 * np.sqrt(2)) * np.sqrt(gravity / depth) * tidal_range
    k = 1 / (period * np.sqrt(gravity * depth))

    # friction parameter
    m = 8 / (3 * np.pi) * friction * velocity / depth

    # damping parameter
    val = -1 + (.5 * convergence / k) ** 2
    mu = k / np.sqrt(2) * np.sqrt(val + np.sqrt(val ** 2 + (m * period) ** 2))

    # tidal damping
    damping = -.5 * convergence + mu

    # tidal prism
    prism = .5 * tidal_range * (
            min_width / damping * (1 - np.exp(-damping * length)) +
            (width - min_width) / (convergence + damping) * (1 - np.exp(-(convergence + damping) * length))
    )

    # return tidal prism and period
    return prism, period


def input_check(
        tidal_range, surge_level, river_discharge,
        channel_depth, channel_width, channel_friction, convergence,
        flat_depth_ratio, flat_width, flat_friction,
        bottom_curvature, meander_amplitude, meander_length,
):
    """Validity check of all input parameters to ensure a physical sound simulation before any numerical computations
    are started. This check also includes a type-check of the input parameters, where all parameters are floats.

    Certain checks make use of empirical relations from data-fitting. The following sources are used:
    Leuven, J.R.F.W., van Maanen, B., Lexmond, B.R., van der Hoek, B.V., Spruijt, M.J., and Kleinhans, M.G. (2018).
        Dimensions of fluvial-tidal meanders: Are they disproportionally large? Geology, 46(10):923-926.
        doi:https://doi.org/10.1130/G45144.1.
    van Rijn, L.C. (2011). Analytical and numerical analysis of tides and salinities in estuaries; part I: tidal wave
        propagation in convergent estuaries. Ocean Dynamics, 61(11):1719-1741.
        doi:https://doi.org/10.1007/s10236-011-0453-0.

    A file is written confirming the validity of the input parameters as received by this method.

    BOUNDARY CONDITIONS
    :param tidal_range: tidal range
    :param surge_level: storm surge level
    :param river_discharge: river discharge

    GEOMETRY
    :param channel_depth: channel depth
    :param channel_width: channel width
    :param channel_friction: channel friction
    :param convergence: estuarine convergence
    :param flat_depth_ratio: flat depth ratio
    :param flat_width: flat width
    :param flat_friction: flat friction
    :param bottom_curvature: channel bottom curvature
    :param meander_amplitude: meander amplitude
    :param meander_length: meander length

    :return: list with error messages
    :rtype: list[str]
    """
    # type-checks
    assert all(isinstance(p, (float, int)) for p in locals().values())

    # error messages
    msg = []

    # channel depth-check
    channel_depth_min = .33 * (3.5 * river_discharge) ** .35
    if not channel_depth > channel_depth_min:
        msg.append(f'channel_depth    : {channel_depth:.1f} must be larger than {channel_depth_min:.1f}.')

    # channel width-check
    channel_width_min = 3.67 * (3.5 * river_discharge) ** .45
    if not channel_width > channel_width_min:
        msg.append(f'channel_width    : {channel_width:.1f} must be larger than {channel_width_min:.1f}.')

    # flat depth-check 1
    if not -1 <= flat_depth_ratio <= 1:
        msg.append(f'flat_depth_ratio : {flat_depth_ratio:.2f} must be between -1.00 and 1.00.')

    # flat depth-check 2
    flat_depth = .5 * flat_depth_ratio * tidal_range
    if not -flat_depth > -channel_depth:
        msg.append(f'flat_depth       : {flat_depth:.2f} must be larger than {-channel_depth:.2f}.')

    # flat width-check
    flat_width_ratio = 1 + flat_width / channel_width
    if not 1 <= flat_width_ratio <= 2:
        msg.append(f'flat_width_ratio : {flat_width_ratio:.2f} must be between 1.00 and 2.00.')

    # bottom curvature-check
    max_bottom_curvature = .6 * channel_depth / (channel_width ** 2)
    if not bottom_curvature <= max_bottom_curvature:
        msg.append(f'bottom_curvature : {bottom_curvature} must be smaller than {max_bottom_curvature}.')

    # meandering-check 1 - based on Leuven et al. (2018)
    total_width = channel_width + flat_width
    max_meander_amplitude = 2.5 * total_width ** 1.1
    if not meander_amplitude <= max_meander_amplitude:
        msg.append(
            f'meander_amplitude: {meander_amplitude:.1f} must be smaller than {max_meander_amplitude:.1f}.'
        )

    # meandering-check 2 - based on Leuven et al. (2018)
    min_meander_length, max_meander_length = 27.044 * meander_amplitude ** .786, 71.429 * meander_amplitude ** .833
    if not min_meander_length <= meander_length <= max_meander_length:
        msg.append(
            f'meander_length   : {meander_length:.1f} must be between {min_meander_length:.1f} and '
            f'{max_meander_length:.1f} (change amplitude [{meander_amplitude}] and/or length [{meander_length}]).'
        )

    # flow velocity-check - based on van Rijn (2011)
    velocity_max = 2
    tidal_prism, tidal_period = _tidal_prism(
        tidal_range, channel_depth, channel_width, channel_width_min, channel_friction, convergence
    )
    channel_cross_section = channel_width * channel_depth
    velocity = river_discharge / channel_cross_section + 2 * tidal_prism / (tidal_period * channel_cross_section)
    if not velocity <= velocity_max:
        msg.append(f'flow velocity    : {velocity:.2f} must be smaller than {velocity_max:.2f}.')

    # return check-results
    return msg
