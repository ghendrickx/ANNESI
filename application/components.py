"""
Components used in the online application; stored separately for readability of both files.

Author: Gijs G. Hendrickx
"""
import numpy as np
import pandas as pd

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class AppEstuary:
    """Estuary object."""
    _length = 8e4
    _ocean_extent = 1e3

    _min_channel_depth = None
    _min_channel_width = None

    def __init__(
            self, channel_depth, channel_width, tidal_range, river_discharge, channel_friction,
            convergence=None, bottom_curvature=None, flat_width_ratio=None, flat_depth_ratio=None, flat_friction = None,
            meander_amplitude=None, meander_length=None, step_size=1
    ):
        """Compressed Estuary-class applicable for the use in the online application.

        :param channel_depth: channel depth [m]
        :param channel_width: channel width [m]
        :param tidal_range: tidal range [m]
        :param river_discharge: river discharge [m3 s-1]
        :param channel_friction: channel friction (Manning) [s m-1/3]
        :param bottom_curvature: channel bottom curvature [m-1], defaults to None
        :param convergence: convergence [m-1], defaults to None
        :param flat_width_ratio: flat width ratio [-], defaults to None
        :param flat_depth_ratio: flat depth ratio [-], defaults to None
        :param flat_friction: flat friction (Manning) [s m-1/3], defaults to None
        :param meander_amplitude: meander amplitude [m], defaults to None
        :param meander_length: meander wave length [m], defaults to None
        :param step_size: grid step size [m], defaults to 1

        :type channel_depth: float
        :type channel_width: float
        :type tidal_range: float
        :type river_discharge: float
        :type convergence: float, optional
        :type bottom_curvature: float, optional
        :type flat_width_ratio: float, optional
        :type flat_depth_ratio: float, optional
        :type meander_amplitude: float, optional
        :type meander_length: float, optional
        :type step_size: float, optional
        """
        self._step_size = step_size

        self.channel_depth = channel_depth
        self.channel_width = channel_width

        self.tidal_range = tidal_range
        self.river_discharge = river_discharge

        self.flat_depth_ratio = 0 if flat_depth_ratio is None else flat_depth_ratio
        self.flat_width_ratio = 1 if flat_width_ratio is None else flat_width_ratio

        self.channel_friction = channel_friction
        self.flat_friction = channel_friction if flat_friction is None else flat_friction

        self.bottom_curvature = 0 if bottom_curvature is None else bottom_curvature

        self.convergence = 0 if convergence is None else convergence

        self.meander_amplitude = 0 if meander_amplitude is None else meander_amplitude
        self.meander_length = meander_length

    @property
    def length(self):
        """Length of estuary in landward direction.

        :return: estuary length [m]
        :rtype: float
        """
        return self._length

    @property
    def ocean_extent(self):
        """Ocean extent of estuary profile.

        :return: ocean extent [m]
        :rtype: float
        """
        return self._ocean_extent

    @property
    def total_width(self):
        """Total estuary width: summation of channel and flat width.

        :return: total width [m]
        :rtype: float
        """
        return self.flat_width_ratio * self.channel_width

    @property
    def min_channel_depth(self):
        """Minimal channel depth as a function of the river discharge; modified from Leuven et al. (2018).

        Leuven, J.R.F.W., Verhoeve, S.L., van Dijk, W.M., Selakovic, S., and Kleinhans, M.G. (2018). Empirical
            assessment tool for bathymetry, flow velocity and salinity in estuaries based on tidal amplitude and
            remotely-sensed imagery. Remote Sensing, 10(12):1915. doi:10.3390/rs10121915

        :return: minimum channel depth [m]
        :rtype: float
        """
        if self._min_channel_depth is None:
            self._min_channel_depth = .33 * (3.5 * self.river_discharge) ** .35

        return self._min_channel_depth

    @property
    def min_channel_width(self):
        """Minimal channel width as a function of the river discharge; modified from Leuven et al. (2018).

        Leuven, J.R.F.W., Verhoeve, S.L., van Dijk, W.M., Selakovic, S., and Kleinhans, M.G. (2018). Empirical
            assessment tool for bathymetry, flow velocity and salinity in estuaries based on tidal amplitude and
            remotely-sensed imagery. Remote Sensing, 10(12):1915. doi:10.3390/rs10121915

        :return: minimum channel width [m]
        :rtype: float
        """
        if self._min_channel_width is None:
            self._min_channel_width = 3.67 * (3.5 * self.river_discharge) ** .45

        return self._min_channel_width

    @property
    def min_total_width(self):
        """Minimum total width as a function of the river discharge and the flat width ratio.

        :return: minimum total width [m]
        :rtype: float
        """
        return self.flat_width_ratio * self.min_channel_width

    @property
    def flat_depth(self):
        """Flat depth as a function of the tidal range and the flat depth ratio.

        :return: flat depth [m]
        :rtype: float
        """
        return .5 * self.flat_depth_ratio * self.tidal_range

    @property
    def _half_channel_width(self):
        """Half the channel width [working parameter].

        :return: half channel width [m]
        :rtype: float
        """
        return .5 * self.channel_width

    @property
    def _half_total_width(self):
        """Half the total width [working parameter].

        :return: half total width [m]
        :rtype: float
        """
        return .5 * self.total_width

    def _meander_wave(self, x):
        """Meandering translation of the y-coordinates based on the meander details.

        :param x: x-coordinates
        :type x: iterable

        :return: y-coordinates
        :rtype: iterable
        """
        if self.meander_amplitude == 0 or self.meander_length == 0:
            return 0
        return self.meander_amplitude * (np.cos((2 * np.pi / self.meander_length) * x) - 1)

    def centre_line(self):
        """Centre-line of the estuary.

        :return: (x,y)-coordinates
        :rtype: tuple
        """
        num_of_points = int((self._ocean_extent + self._length) / self._step_size + 1)
        x = np.linspace(-self._ocean_extent, self._length, num_of_points)
        y = self._meander_wave(x)
        return x, y

    def _converging_banks(self, x):
        """Converging banks [working parameter].

        :param x: x-coordinates
        :type x: iterable

        :return: y-coordinates
        :rtype: iterable
        """
        return .5 * (self.min_total_width + (self.total_width - self.min_total_width) * np.exp(-self.convergence * x))

    def _converging_channel(self, x):
        """Converging channel [working parameter].

        :param x: x-coordinates
        :type x: iterable

        :return: y-coordinates
        :rtype: iterable
        """
        return .5 * (self.min_channel_width + (self.channel_width - self.min_channel_width)) * np.exp(-self.convergence * x)

    def _channel_profile(self, x, y):
        """Channel cross-sectional profile.

        :param x: x-coordinates
        :param y: y-coordinates

        :type x: iterable
        :type y: iterable

        :return: z-coordinates
        :rtype: iterable
        """
        def linear_bottom():
            return self.channel_depth - (self.channel_depth - self.min_channel_depth) / self._length * x

        def curved_bottom(curvature):
            return curvature * (1/12 * self._converging_channel(x) ** 2 - y ** 2)

        return -(linear_bottom() + curved_bottom(self.bottom_curvature))

    def profile(self, x, y):
        """Cross-sectional profile.

        :param x: x-coordinates
        :param y: y-coordinates

        :type x: iterable
        :type y: iterable

        :return: z-coordinates
        :rtype: iterable
        """
        z = self._channel_profile(x, y)
        z[(y > self._half_channel_width) | (y < -self._half_channel_width)] = -self.flat_depth
        return z

    def lateral_reformat(self, x, y):
        """Reformat (x,y)-coordinates to represent changes over the longitudinal direction (i.e. x-direction).

        :param x: x-coordinates
        :param y: y-coordinates

        :type x: iterable
        :type y: iterable

        :return: (x,y)-coordinates
        :rtype: tuple
        """
        # convergence
        y *= (2 / self.total_width) * self._converging_banks(x)

        # meandering
        if self.meander_amplitude > 0 and self.meander_length > 0:
            angle = np.arctan(-self.meander_amplitude * 2 * np.pi / self.meander_length * np.sin(
                (2 * np.pi / self.meander_length) * x
            ))
            rotation = 1j * y * np.exp(1j * angle)
            y = np.imag(rotation) + self._meander_wave(x)
            x += np.real(rotation)

        return x, y

    @property
    def land_boundaries(self):
        """Land boundaries os Estuary, expressed as line-segments.

        :return: land boundaries
        :rtype: pandas.DataFrame
        """
        num_of_points = int((self._ocean_extent + self._length) / self._step_size + 1)
        x = np.linspace(-self._ocean_extent, self._length, num_of_points)
        xy = pd.DataFrame({'x': x})

        _, xy['NORTH_BANK'] = self.lateral_reformat(xy.x,  self._half_total_width * np.ones_like(x))
        _, xy['SOUTH_BANK'] = self.lateral_reformat(xy.x, -self._half_total_width * np.ones_like(x))

        return xy

    @property
    def bathymetry(self):
        """Bathymetry of Estuary, expressed by (x,y,z)-coordinates (z-direction positive upwards).

        :return: bathymetry
        :rtype: pandas.DataFrame
        """
        x_points = int((self._ocean_extent + self._length) / self._step_size + 1)
        y_points = int(self.total_width / self._step_size + 1)

        x = np.linspace(-self._ocean_extent, self._length, x_points)
        y = np.linspace(-self._half_total_width, self._half_total_width, y_points)

        xy = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        xyz = pd.DataFrame(data=xy, columns=['x', 'y'])

        xyz['z'] = self.profile(xyz.x, xyz.y)
        xyz[['x', 'y']] = xyz.apply(lambda r: pd.Series(self.lateral_reformat(r.x, r.y)), axis=1)

        return xyz


class EstuaryType:
    """Classification of the estuary based on the freshwater Froude number and the mixing number, as proposed by Geyer
    & MacCready (2014).

    Geyer, W.R., and MacCready, P. (2014). The estuarine circulation. Annual Review of Fluid Mechanics, 46(1):175-197.
        doi:https://doi.org/10.1146/annurev-fluid-010313-141302.
    """
    _gravity = 9.81
    _contraction = 7.6e-4
    _salinity = 30
    _tidal_frequency = 2 * np.pi / 44700

    def __init__(self, estuary):
        """
        :param estuary: estuary definition
        :type estuary: AppEstuary
        """
        self._estuary = estuary

    def __str__(self):
        """String-representation.

        :return: estuary type
        :rtype: str
        """
        return str(self.classification)

    @property
    def mixing(self):
        """Mixing-parameter as defined by Geyer & MacCready (2014).

        Geyer, W.R., and MacCready, P. (2014). The estuarine circulation. Annual Review of Fluid Mechanics,
            46(1):175-197. doi:https://doi.org/10.1146/annurev-fluid-010313-141302.

        :return: mixing-parameter
        :rtype: float
        """
        velocity = 1/(2*np.sqrt(2)) * np.sqrt(self._gravity / self._estuary.channel_depth) * self._estuary.tidal_range
        fricion = self._gravity * (self._estuary.channel_friction ** 2) / (self._estuary.channel_depth ** (1/3))
        buoyancy_freq = np.sqrt(self._contraction * self._gravity * self._salinity / self._estuary.channel_depth)
        return np.sqrt(
            (fricion * velocity ** 2) / (self._tidal_frequency * buoyancy_freq * self._estuary.channel_depth ** 2)
        )

    @property
    def froude(self):
        """Freshwater Froude number as defined by Geyer & MacCready (2014).

        Geyer, W.R., and MacCready, P. (2014). The estuarine circulation. Annual Review of Fluid Mechanics,
            46(1):175-197. doi:https://doi.org/10.1146/annurev-fluid-010313-141302.

        :return: freshwater Froude number
        :rtype: float
        """
        internal_wave = np.sqrt(self._contraction * self._salinity * self._gravity * self._estuary.channel_depth)
        return self._estuary.river_discharge / (
                self._estuary.channel_width * self._estuary.channel_depth * internal_wave
        )

    @property
    def classification(self):
        """Classification of estuary based on the freshwater Froude number and the mixing-parameter following Geyer &
        MacCready (2014).

        Geyer, W.R., and MacCready, P. (2014). The estuarine circulation. Annual Review of Fluid Mechanics,
            46(1):175-197. doi:https://doi.org/10.1146/annurev-fluid-010313-141302.

        :return: estuary type
        :rtype: str
        """
        p = Point(self.mixing, self.froude)

        # salt wedge
        if Polygon([(.19, 7e-2), (.3, 1), (1, 1), (.8, 7e-2)]).contains(p):
            return 'salt wedge'

        # time-dependent salt wedge
        if Polygon([(.8, 7e-2), (1, 1), (2, 1), (2, 7e-2)]).contains(p):
            return 'time-dependent salt wedge'

        # strongly stratified
        if Polygon([(.1, 2e-3), (.19, 7e-2), (.75, 7e-2), (.45, 2e-3)]).contains(p):
            return 'strongly stratified'

        # partially mixed
        if Polygon([(.8, 2e-3), (1.4, 7e-2), (.75, 7e-2), (.45, 2e-3)]).contains(p):
            return 'partially mixed'

        # fjord

        # bay

        # SIPS
        if Polygon([(1.4, 7e-2), (2, 7e-2), (2, 1e-2), (1, 1e-4), (.5, 1e-4)]).contains(p):
            return 'strain-induced periodic stratification (SIPS)'

        # well-mixed
        if Polygon([(1, 1e-4), (2, 1e-2), (2, 1e-4)]).contains(p):
            return 'well-mixed'

        return 'undefined'


def _type_check(param, type_):
    """Check the type of the parameter.

    :param param: parameter
    :param type_: type

    :type param: typ
    :type type_: type
    """
    if not type(param) == type_:
        try:
            param = type_(param)
        except ValueError:
            pass
    if not isinstance(param, type_):
        msg = f'{param} should be of type {type_}, {type(param)} given.'
        raise TypeError(msg)


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


def input_check(
        tidal_range, surge_level, river_discharge,
        channel_depth, channel_width, channel_friction, convergence,
        flat_depth_ratio, flat_width, flat_friction,
        bottom_curvature, meander_amplitude, meander_length,
):
    """Validity check of all input parameters to ensure a physical sound simulation before any numerical computations
    are started. The meandering-related checks are based on Leuven et al. (2018).

    This check also includes a type-check of the input parameters, where all parameters are of :type: float.

    BOUNDARY CONDITIONS
    :param tidal_range: tidal range [m]
    :param surge_level: storm surge level [m]
    :param river_discharge: river discharge [m3 s-1]

    GEOMETRY
    :param channel_depth: channel depth [m]
    :param channel_width: channel width [m]
    :param channel_friction: channel friction [s m-1/3]
    :param convergence: estuarine convergence [m-1]
    :param flat_depth_ratio: flat depth ratio [-]
    :param flat_width: flat width [m]
    :param flat_friction: flat friction [s m-1/3]
    :param bottom_curvature: channel bottom curvature [m-1]
    :param meander_amplitude: meander amplitude [m]
    :param meander_length: meander length [m]

    Leuven, J.R.F.W., van Maanen, B., Lexmond, B.R., van der Hoek, B.V., Spruijt, M.J., and Kleinhans, M.G. (2018).
        Dimensions of fluvial-tidal meanders: Are they disproportionally large? Geology, 46(10):923-926.
        doi:https://doi.org/10.1130/G45144.1.
    """
    # type-checks
    [_type_check(p, float) for p in locals().values()]

    # input parameters
    params = locals().copy()

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
    if not flat_depth > -channel_depth:
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
    max_meander_amplitude = 2.5 * (channel_width + flat_width) ** 1.1
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

    # flow velocity-check
    velocity_max = 2
    tidal_prism, tidal_period = _tidal_prism(
        tidal_range, channel_depth, channel_width, channel_width_min, channel_friction, convergence
    )
    channel_cross_section = channel_width * channel_depth
    velocity = river_discharge / channel_cross_section + 2 * tidal_prism / (tidal_period * channel_cross_section)
    if not velocity <= velocity_max:
        msg.append(f'flow velocity    : {velocity:.2f} must be smaller than {velocity_max:.2f}.')

    # raise ValueError if any checks have not passed
    if msg:
        raise ValueError(msg)
