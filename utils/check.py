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
    length = 8e4
    # gravitational acceleration
    g = 9.81
    # tidal wave
    velocity = 1 / (2 * np.sqrt(2)) * np.sqrt(g / depth) * tidal_range
    period = 12 * 3600
    k = 1 / (period * np.sqrt(g * depth))
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
    return prism, period
