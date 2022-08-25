"""
Neural network fitted to DFM-simulations: Frontend. The neural network's backend is in '_backend.py'.

Author: Gijs G. Hendrickx
"""
import numpy as np
import pandas as pd
import logging

import torch

from utils.check import input_check
from utils.files_dirs import DirConfig
from utils.data_conv import Import, Export

from src._backend import MLP, InputData, DEVICE, _NNData

LOG = logging.getLogger(__name__)


class NeuralNetwork(_NNData):
    """The interface of a neural network, which defaults to a neural network trained on a large data set of hydrodynamic
    simulations using Delft3D Flexible Mesh (DFM). The DFM-simulations encompass idealised estuaries, and wide-ranging
    sets of parameters are evaluated.
    """
    _reduced_output_vars = None
    _de_norm = {'L': 8e4, 'V': 30}

    def __init__(self, neural_network=None):
        """Loads trained neural network, which is stored inside the package, by default. When a neural network is
        provided, this default network is overruled, and the provided neural network is used. It is assumed that this
        neural network has been trained upfront; if so, please make sure that the neural network is trained using the
        correct data:

        >>> input_variables = NeuralNetwork.get_input_vars()
        >>> output_variables = NeuralNetwork.get_output_vars())

        In case the neural network has not been trained upfront, there is the option to train it to the default data
        set; use the 'fit_custom_neural_network'-method. In that case, make sure that the input and output dimensions
        of the untrained neural network agree with this training data set:

        >>> input_dimension = len(NeuralNetwork.get_input_vars())
        >>> output_dimension = len(NeuralNetwork.get_output_vars())

        In case a custom-made neural network design is provided, please use PyTorch to define this neural network
        (see https://pytorch.org/docs/stable/index.html). For the structure of the custom-made neural network, one may
        get inspired by the definition of the default neural network: 'neural_network._backend.MLP'.

        :param neural_network: (trained) neural network, defaults to None
        :type neural_network: torch.nn.Module, optional
        """
        self._nn = neural_network

    def __repr__(self):
        """Object representation."""
        return f'NeuralNetwork(neural_network={self.nn})'

    def __str__(self):
        """String representation."""
        return f'ANNESI: Artificial neural network for estuarine salt intrusion'

    @property
    def nn(self):
        """Neural network. When none is specified during the initiation, the default neural network is loaded and used.

        :return: (trained) neural network
        :rtype: torch.nn.Module
        """
        if self._nn is None:
            # default neural network
            nn = MLP(input_dim=len(self.input_vars), output_dim=len(self.output_vars))

            # import neural network
            self._nn = Import(self.wd).from_pkl(nn)

        # return neural network
        return self._nn

    @property
    def output(self):
        """Output variables, which may be a selection of the available output parameters. The available output
        parameters are defined by:

        >>> NeuralNetwork.get_output_vars()

        :return: output variables
        :rtype: list
        """
        return self.output_vars if self._reduced_output_vars is None else self._reduced_output_vars

    @output.setter
    def output(self, out):
        """Set output variables of interest, which must be a selection of the available output parameters. The available
        output parameters are defined by:

        >>> NeuralNetwork.get_output_vars()

        When the output variables are set to None, this is considered as resetting the output definition, which returns
        all available output variables.

        :param out: selection of output parameters
        :type out: iterable[str], None
        """
        # reset to default, i.e. all output variables
        if out is None:
            self._reduced_output_vars = None

        def validation(key):
            """Validate output definition.

            :param key: output variable
            :type key: str

            :return: valid output definition
            :rtype: bool
            """
            if key in self.output_vars:
                # valid output definition
                return True

            # invalid output definition
            LOG.critical(f'Unavailable output variable: \"{key}\": Skipped.\n\tChoose one of {self.output_vars}')
            return False

        # single output definition
        if isinstance(out, str):
            out = [out]

        # validated custom output
        self._reduced_output_vars = [v for v in out if validation(v)]

    def _de_normalise(self, output):
        """De-normalise output data.

        :param output: normalised output data
        :type output: pandas.DataFrame

        :return: de-normalised output data
        :rtype: pandas.DataFrame
        """
        return output * self._de_norm

    def single_predict(
            self, tidal_range, surge_level, river_discharge, channel_depth, channel_width, channel_friction,
            convergence, flat_depth_ratio, flat_width, flat_friction, bottom_curvature, meander_amplitude,
            meander_length
    ):
        """Predict output of a single set of input parameters. This single predictor provides more guidance in the type
        of input; the `predict`- and `predict_from_file`-methods enable predictions based on multiple sets of input
        parameters, i.e. based on an input data set.

        :param tidal_range: tidal range
        :param surge_level: storm surge level
        :param river_discharge: river discharge
        :param channel_depth: channel depth
        :param channel_width: channel width
        :param channel_friction: channel friction
        :param convergence: convergence
        :param flat_depth_ratio: flat depth ratio
        :param flat_width: flat width
        :param flat_friction: flat friction
        :param bottom_curvature: bottom curvature
        :param meander_amplitude: meander amplitude
        :param meander_length: meander length

        :type tidal_range: float
        :type surge_level: float
        :type river_discharge: float
        :type channel_depth: float
        :type channel_width: float
        :type channel_friction: float
        :type convergence: float
        :type flat_depth_ratio: float
        :type flat_width: float
        :type flat_friction: float
        :type bottom_curvature: float
        :type meander_amplitude: float
        :type meander_length: float

        :return: neural network-based estimate of output
        :rtype: pandas.DataFrame
        """
        # extract input data
        input_data = [v for k, v in locals().items() if k in self._input_vars]

        # physical input check
        msg = input_check(*input_data)
        if len(msg) > 0:
            LOG.critical(f'Input is considered physically invalid; use output with caution!')

        # normalise data
        norm_input = InputData.normalise([input_data])

        # use neural network
        x = torch.tensor(norm_input).float().to(DEVICE)
        y = self.nn(x)
        df = pd.DataFrame(data=y.detach().cpu(), columns=self._output_vars, dtype=float)

        # return output estimation
        out = self._de_normalise(df)
        return out[self.output]

    def predict(self, data, scan='full'):
        """Predict output.

        The scanning method is based on the `scan`-argument, which can have one of three values:
         1. 'full'      :   apply the input check and raise an error if there is an invalid model configuration.
         2. 'skip'      :   apply the input check and remove invalid model configurations from the data.
         3. 'ignore'    :   ignore the input check and predict for all model configurations, invalid or valid.

        :param data: input data
        :param scan: method of scanning the input data, defaults to 'full'

        :type data: pandas.DataFrame
        :type scan: str

        :return: prediction
        :rtype: pandas.DataFrame
        """
        # scanning data set: physical input check
        if scan == 'full':
            data.apply(lambda row: physical_input_check(False, *row[self.input_vars], grid_limits=False), axis=1)

        elif scan == 'skip':

            def check(*args):
                """Perform input check and return a warning when it is not passed."""
                msg_ = _input_check(*args, grid_limits=False)
                if msg_:
                    LOG.warning(msg_)
                    return None
                return args

            data = data.apply(lambda row: check(*[row[p] for p in self.input_vars]), axis=1, result_type='broadcast')
            data.dropna(inplace=True)

        elif scan == 'ignore':
            pass

        else:
            msg = f'Scanning option {scan} not included; see documentation for help.'
            raise NotImplementedError(msg)

        # normalise data
        norm_data = InputData.normalise(data[self.input_vars])

        # use neural network
        x = torch.tensor(norm_data).float().to(DEVICE)
        y = self.nn(x)

        # store as pandas.DataFrame
        df = pd.DataFrame(data=y.detach().cpu(), columns=self.output_vars, dtype=float)

        # return selected output
        out = self._de_normalise(df)
        return out[self.output]

    def predict_from_file(self, file_name, directory=None, scan='full', **kwargs):
        """Predict output based on input data from a file.

        The scanning method is based on the `scan`-argument, which can have one of three values:
         1. 'full'      :   apply the input check and raise an error if there is an invalid model configuration.
         2. 'skip'      :   apply the input check and remove invalid model configurations from the data.
         3. 'ignore'    :   ignore the input check and predict for all model configurations, invalid or valid.

        :param file_name: file name
        :param directory: directory, defaults to None
        :param scan: method of scanning the input data, defaults to 'full'
        :param kwargs: pandas.read_csv key-worded arguments

        :type file_name: str
        :type directory: DirConfig, str, list[str], tuple[str], optional
        :type scan: str, optional

        :return: prediction
        :rtype: pandas.DataFrame
        """
        # load data
        file = DirConfig(directory).config_dir(file_name)
        data = pd.read_csv(file, **kwargs)

        # predict output
        self.predict(data, scan)

    def estimate(
            self, tidal_range=None, surge_level=None, river_discharge=None, channel_depth=None, channel_width=None,
            channel_friction=None, convergence=None, flat_depth_ratio=None, flat_width=None, flat_friction=None,
            bottom_curvature=None, meander_amplitude=None, meander_length=None,
            parameter_samples=3, statistics=True, include_input=False,
    ):
        """Provides an estimate with incomplete input data. The rough estimate is based on assessing a range values of
        the undefined parameter(s). Based on the resulting spreading, a rough estimate is provided. A parameter can also
        be given as a list (or tuple) containing the minimum and maximum values of a range of interest. Between these
        two provided extremes, a number of samples are evaluated (equal to :param parameter_samples:).

        The size of the range assessed per undefined parameter is defined by :param parameter_samples:, which iterates
        through the minimum and maximum of the undefined parameter. Multiple values are assessed due to the expected
        non-linear response. However, the number of iterations may rise quickly when more parameters remain undefined:
        The number of iterations required equals :param parameter_samples: to the power of the number of undefined
        parameters. As an example, with :param parameter_samples: equal to 3 (default), and three undefined parameters,
        the total number of iterations equals 3 ** 3, or 27.

        :param tidal_range: tidal range, defaults to None
        :param surge_level: storm surge level, defaults to None
        :param river_discharge: river discharge, defaults to None
        :param channel_depth: channel depth, defaults to None
        :param channel_width: channel width, defaults to None
        :param channel_friction: channel friction, defaults to None
        :param convergence: convergence, defaults to None
        :param flat_depth_ratio: flat depth ratio, defaults to None
        :param flat_width: flat width, defaults to None
        :param flat_friction: flat friction, defaults to None
        :param bottom_curvature: bottom curvature, defaults to None
        :param meander_amplitude: meander amplitude, defaults to None
        :param meander_length: meander length, defaults to None
        :param parameter_samples: number of samples per incompletely defined parameter, defaults to 3
        :param statistics: return statistics of estimate, defaults to True
        :param include_input: include input in returned data, defaults to False

        :type tidal_range: float, iterable, optional
        :type surge_level: float, iterable, optional
        :type river_discharge: float, iterable, optional
        :type channel_depth: float, iterable, optional
        :type channel_width: float, iterable, optional
        :type channel_friction: float, iterable, optional
        :type convergence: float, iterable, optional
        :type flat_depth_ratio: float, iterable, optional
        :type flat_width: float, iterable, optional
        :type flat_friction: float, iterable, optional
        :type bottom_curvature: float, iterable, optional
        :type meander_amplitude: float, iterable, optional
        :type meander_length: float, iterable, optional
        :type parameter_samples: int, optional
        :type statistics: bool, optional
        :type include_input: bool, optional

        :return: neural network-based estimate of output, or its statistics
        :rtype: pandas.DataFrame
        """

        def check(*args):
            """Execute physical input check and return None if the model configuration fails this check. This will
            eventually eliminate the model configuration from the data set.

            :param args: input parameters
            :type args: bool, str, float

            :return: input parameters, None
            :rtype: list, NoneType
            """
            # model configuration check: physical soundness
            msg = input_check(*args)

            # model configuration check: failed
            if msg:
                LOG.info(f'Physical input check failed: {args}')
                return None

            # model configuration check: passed
            return args

        # determine estimate when all input parameters are provided
        if all(isinstance(v, (float, int)) for k, v in locals().items() if k in self.input_vars):
            return self.single_predict(**{k: v for k, v in locals().items() if k in self.input_vars})

        # define input space
        scaler = InputData.get_scaler()
        arrays = dict()
        for i, var in enumerate(self.input_vars):
            # no range defined
            if locals()[var] is None:
                arrays[var] = np.linspace(scaler.data_min_[i], scaler.data_max_[i], parameter_samples)
            # single value defined
            elif isinstance(locals()[var], (float, int)):
                arrays[var] = locals()[var]
            # range defined
            else:
                min_var = max(min(locals()[var]), scaler.data_min_[i])
                max_var = min(max(locals()[var]), scaler.data_max_[i])
                arrays[var] = np.linspace(min_var, max_var, parameter_samples)
                # warning message: range outside training data
                if min(locals()[var]) < scaler.data_min_[i] or max(locals()[var]) > scaler.data_max_[i]:
                    LOG.warning(f'Defined range exceeds training data; \"{var}\" range used: {min_var, max_var}')

        # create model configurations
        df = pd.DataFrame(
            data=np.array(np.meshgrid(*[v for v in arrays.values()])).T.reshape(-1, len(arrays)),
            columns=self.input_vars
        )

        # check data and predict output
        df[self.output] = self.predict(df, scan='skip')

        # return statistics of estimation
        if include_input:
            return df
        return df[self.output].describe() if statistics else df[self.output]

    @classmethod
    def get_input_vars(cls):
        """Get class-level defined input variables

        :return: input variables
        :rtype: tuple
        """
        return cls._input_vars

    @classmethod
    def get_output_vars(cls):
        """Get class-level defined output variables

        :return: output variables
        :rtype: tuple
        """
        return cls._output_vars

    def save_as(self, f_type, file_name=None, directory=None):
        """Save neural network as one of the available export-formats:
         1. *.pkl   :   for usage within Python, using PyTorch.
         2. *.onnx  :   for integration of neural network in a website.

        :param f_type: file-type
        :param file_name: file-name, defaults to None
        :param directory: directory, defaults to None

        :type f_type: str
        :type file_name: str, optional
        :type directory: DirConfig, str, iterable[str], optional
        """
        # check available export-formats
        f_types = ('pkl', 'onnx')
        if f_type not in f_types:
            msg = f'NeuralNetwork can only be saved as {f_types}, {f_type} has been specified.'
            raise NotImplementedError(msg)

        # internally save neural network
        if directory == 'internal-save':
            msg = 'About to save neural network internally. This might overwrite an existing version. Continue? [y/n]'
            if input(msg) == 'y':
                directory = self.wd
            else:
                msg = 'Internally saving neural network aborted.'
                raise KeyboardInterrupt(msg)

        # export neural network
        export = Export(directory)
        if f_type == 'pkl':
            export.to_pkl(self.nn, file_name=file_name)
        elif f_type == 'onnx':
            export.to_onnx(self.nn, file_name=file_name)
